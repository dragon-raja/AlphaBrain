"""Opt-in warm interpreter transport; every trajectory gets a fresh environment.

No observation/physics reset cache. A failed session closes its process and
poisons the pool. The caller retains per-call center authorization and budgets.
"""
from contextlib import contextmanager
from multiprocessing.connection import Listener
from pathlib import Path
import subprocess
import tempfile
import time

from AlphaBrain.research.dsol.execution.manifest import digest, file_digest
from .search_live import simulator_environment


class SimulatorSessionPool:
    def __init__(self, log_root, *, max_configs=2):
        self.root = Path(log_root)
        self.max_configs = max_configs
        self.servers = {}
        self.failed = False
        self.closed = False
        if type(max_configs) is not int or not 1<=max_configs<=2:
            raise ValueError('One or two bound simulator configurations required')

    @contextmanager
    def session(self, config_path, config, initial, output):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/search_live.py:simulator')
        if self.failed or self.closed:
            raise RuntimeError('Failed simulator pool requires explicit recovery')
        key = digest(dict(config=config, file_sha256=file_digest(config_path)))
        try:
            server = self.servers.get(key)
            if server is None:
                if len(self.servers) >= self.max_configs:
                    raise RuntimeError('Resident simulator configuration capacity exceeded')
                self.root.mkdir(parents=True,exist_ok=True)
                server = _Server(config_path,config,self.root/(key+'.log'))
                self.servers[key] = server
            with server.session(initial) as request:
                yield request
        except BaseException:
            self.failed = True
            self.close()
            raise

    def close(self):
        self.closed = True
        for server in self.servers.values():
            server.close()


class _Server:
    def __init__(self, config_path, config, log):
        self.child = None
        self.connection = None
        self.temporary = tempfile.TemporaryDirectory(prefix='dsol-resident-sim-')
        self.sequence = 0
        self.busy = False
        self.closed = False
        address = str(Path(self.temporary.name)/'sim.sock')
        try:
            with Listener(address,family='AF_UNIX') as listener:
                command=[config['sim_python'],'-u','-m','benchmarks.LIBERO.search_simulator',
                         '--config',str(config_path),'--socket',address,'--resident','--execute']
                with Path(log).open('xb') as stderr:
                    self.child=subprocess.Popen(command,env=simulator_environment(config),
                        cwd=config['source'],stdout=subprocess.DEVNULL,stderr=stderr)
                deadline=time.monotonic()+180
                while self.connection is None:
                    if self.child.poll() is not None:
                        raise RuntimeError('Resident simulator exited before handshake')
                    if time.monotonic()>=deadline:
                        raise TimeoutError('Resident simulator handshake timeout')
                    listener._listener._socket.settimeout(min(1.,deadline-time.monotonic()))
                    try:self.connection=listener.accept()
                    except TimeoutError:pass
            if self.receive()!=dict(status='IDLE',sequence=0):
                raise RuntimeError('Resident simulator initial handshake changed')
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(180):
            raise TimeoutError('Resident simulator response timeout')
        value=self.connection.recv()
        if not isinstance(value,dict) or value.get('error'):
            raise RuntimeError('Resident simulator rejected request')
        return value

    def request(self, payload):
        self.connection.send(payload)
        return self.receive()

    @contextmanager
    def session(self, initial):
        if self.closed or self.busy or self.child.poll() is not None:
            raise RuntimeError('Resident simulator is unavailable; no restart')
        if type(initial) is not int or initial<0:
            raise ValueError('Explicit initial index required')
        self.busy=True
        try:
            self.sequence+=1
            reply=self.request(dict(kind='session',sequence=self.sequence,initial=initial))
            if reply.get('status')!='READY':
                raise RuntimeError('Fresh environment did not initialize')
            yield self.request
            if self.request(dict(kind='close'))!=dict(ok=True):
                raise RuntimeError('Environment close acknowledgement changed')
            if self.receive()!=dict(status='IDLE',sequence=self.sequence):
                raise RuntimeError('Environment did not finish before pool reuse')
            self.busy=False
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:return
        self.closed=True
        if self.connection is not None:
            if not self.busy:
                try:self.connection.send(dict(kind='shutdown',sequence=self.sequence))
                except (OSError,EOFError):pass
            self.connection.close()
        if self.child is not None:
            try:self.child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
        self.temporary.cleanup()
