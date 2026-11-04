"""CPU-only fresh-environment surrogate for the real resident IPC driver."""
import argparse
import os

from benchmarks.LIBERO import search_simulator


def session(config, initial, connection):
    state=initial
    connection.send(dict(status='READY',physics='cpu-only'))
    while True:
        message=connection.recv()
        if message['kind']=='close':
            connection.send(dict(ok=True));return
        if message['kind']=='step':state+=1
        if message['kind'] not in ('step','query_pose'):raise RuntimeError('Injected invalid command')
        connection.send(dict(state=state,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config');p.add_argument('--socket');p.add_argument('--resident',action='store_true')
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    search_simulator.serve_session=session
    search_simulator.serve_resident(args.config,args.socket)
