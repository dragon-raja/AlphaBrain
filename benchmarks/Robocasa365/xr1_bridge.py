"""Private inherited-pipe bridge between frozen simulator and XR-1 runtimes.

Only the child spawned here can access the pipe. No network listener, remote
pickle endpoint, or modification of the original simulation environment.
"""
import argparse
from collections import deque
from multiprocessing.connection import Connection, Pipe
import os
from pathlib import Path
import subprocess


class XR1Bridge:
    def __init__(self, python, checkpoint, upstream, log):
        self.connection, child = Pipe(duplex=True)
        self.log = Path(log).open('xb')
        command=[str(python),'-m','benchmarks.Robocasa365.xr1_bridge','--fd',str(child.fileno()),
                 '--checkpoint',str(checkpoint),'--upstream',str(upstream)]
        environment = os.environ.copy()
        if environment.get('XR1_MODEL_PYTHONPATH'):
            environment['PYTHONPATH'] = environment['XR1_MODEL_PYTHONPATH']
        self.process=subprocess.Popen(command,pass_fds=(child.fileno(),),stdout=self.log,stderr=self.log,env=environment)
        child.close()
        try:
            reply=self.receive()
            if reply.get('status')!='READY':
                raise RuntimeError('Model bridge failed admission')
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(180):
            raise TimeoutError('XR-1 child did not answer within 180 seconds')
        result=self.connection.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def infer(self, observations, instruction, seed, capture=False):
        self.connection.send(dict(observations=list(observations),instruction=instruction,
                                  seed=int(seed),capture=capture))
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                self.connection.send(None)
                self.process.wait(timeout=15)
            except (OSError, EOFError, subprocess.TimeoutExpired):
                self.process.terminate(); self.process.wait(timeout=15)
        self.connection.close(); self.log.close()


def serve(fd, checkpoint, upstream):
    connection=Connection(fd)
    try:
        from .xr1_policy import XR1Policy
        policy=XR1Policy(checkpoint,upstream)
        connection.send(dict(status='READY',pid=os.getpid()))
        while True:
            request=connection.recv()
            if request is None:
                break
            policy.history=deque(request['observations'],maxlen=7)
            result=policy.infer(request['instruction'],request['seed'],capture=request['capture'])
            connection.send(result)
    except BaseException as exc:
        connection.send(dict(error=repr(exc)))
        raise
    finally:
        connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd',type=int,required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--upstream',required=True)
    args=parser.parse_args();serve(args.fd,args.checkpoint,args.upstream)


if __name__=='__main__':
    main()
