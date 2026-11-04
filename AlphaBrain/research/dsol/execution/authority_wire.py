"""Bounded JSON messages over local pipes or an existing strict SSH alias."""
import json
import os
from pathlib import Path
import re
import select
import shlex
import socket
import subprocess
import time

from .manifest import digest, file_digest
from .queue import _durable_write


def identity():
    return dict(host=socket.gethostname(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                namespace=os.readlink('/proc/self/ns/pid'))


def require_local(path):
    """Live authorization may not use a cached network filesystem."""
    path = Path(path).resolve()
    filesystem = subprocess.check_output(['stat', '-f', '-c', '%T', str(path)], text=True).strip()
    if filesystem not in ('ext2/ext3', 'xfs', 'btrfs', 'overlayfs', 'tmpfs'):
        raise RuntimeError('Authority/worker live state requires a local filesystem: '+filesystem)


def fingerprint():
    root = Path(__file__).parent
    return digest({name: file_digest(root/name) for name in
                   ('authority_wire.py', 'authority_worker.py', 'manifest.py', 'queue.py')})


def checkpoint(ledger):
    """Immutable evidence only; remote nodes never read it to obtain a grant."""
    plan, state = ledger.read()
    value = dict(schema='AUTHORITY_CHECKPOINT_V1', authority=identity(), plan=plan, ledger=state)
    stamp = digest(value)
    destination = ledger.authority_mirror / (stamp+'.json')
    if destination.exists():
        if digest(json.loads(destination.read_text())) != stamp:
            raise ValueError('Immutable authority checkpoint changed')
    else:
        _durable_write(destination, value)
    _durable_write(ledger.root/'checkpoint.json', dict(sha256=stamp, path=str(destination)))
    return stamp


class Wire:
    def __init__(self, reader, writer, timeout=10):
        self.reader, self.writer, self.timeout = reader, writer, timeout
        self.buffer = b''

    def send(self, value):
        data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()+b'\n'
        if len(data) > 1024*1024:
            raise ValueError('Oversized protocol message')
        deadline = time.monotonic()+self.timeout
        while data:
            left = deadline-time.monotonic()
            if left <= 0 or not select.select([], [self.writer], [], left)[1]:
                raise TimeoutError('Authority channel write timeout')
            count = os.write(self.writer, data[:4096])
            data = data[count:]

    def receive(self):
        deadline = time.monotonic()+self.timeout
        while b'\n' not in self.buffer:
            left = deadline-time.monotonic()
            if left <= 0 or not select.select([self.reader], [], [], left)[0]:
                raise TimeoutError('Authority channel read timeout')
            data = os.read(self.reader, 4096)
            if not data:
                raise ConnectionError('Authority channel disconnected')
            self.buffer += data
            if len(self.buffer) > 1024*1024:
                raise ValueError('Oversized protocol message')
        return self._message()

    def receive_ready(self):
        """Nonblocking frame read; retain partial JSON without stalling peers."""
        if b'\n' not in self.buffer:
            if not select.select([self.reader], [], [], 0)[0]:
                return None
            data = os.read(self.reader, 4096)
            if not data:
                raise ConnectionError('Authority channel disconnected')
            self.buffer += data
            if len(self.buffer) > 1024*1024:
                raise ValueError('Oversized protocol message')
            if b'\n' not in self.buffer:
                return None
        return self._message()

    def _message(self):
        line, self.buffer = self.buffer.split(b'\n', 1)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('Protocol object required')
        if value.get('kind') == 'ERROR':
            raise RuntimeError('CPU worker rejected request: '+value['error'])
        return value


def ssh_command(target):
    """Strict SSH argv, optionally with a hash-bound public endpoint override.

    Authentication still comes from the existing alias. No credential paths,
    commands, forwarding, or host-key-check overrides are accepted here.
    """
    alias=target['alias']
    if not isinstance(alias,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',alias):
        raise ValueError('A fixed SSH alias is required')
    args=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
          '-o','ConnectTimeout=5','-o','ServerAliveInterval=2','-o','ServerAliveCountMax=2']
    route=target.get('public_endpoint')
    if route is not None:
        import ipaddress
        required={'hostname','port','host_key_alias','known_hosts_file','known_hosts_sha256'}
        if set(route)!=required or type(route['port']) is not int or not 1<=route['port']<=65535:
            raise ValueError('Explicit public endpoint fields required')
        host=str(ipaddress.ip_address(route['hostname']));port=route['port']
        if route['host_key_alias']!=f'[{host}]:{port}':raise ValueError('Endpoint host-key identity differs')
        path=Path(route['known_hosts_file'])
        if not path.is_absolute() or file_digest(path)!=route['known_hosts_sha256']:
            raise ValueError('Public known-hosts file changed')
        args+=['-o','HostName='+host,'-p',str(port),'-o','HostKeyAlias='+route['host_key_alias'],
               '-o','UserKnownHostsFile='+str(path)]
    return args+[alias]


class Channel:
    def __init__(self, target, *, prepare=False, agent=None, expected_source=None, environment=None,
                 hello_timeout=None):
        if hello_timeout is not None and (type(hello_timeout) not in (int, float) or not 10 <= hello_timeout <= 600):
            raise ValueError('Initial CPU handshake must have a bounded 10..600 second deadline')
        self.hello_timeout = hello_timeout
        self.target = target
        self.detached = False
        self.closed = False
        self.expected_source = expected_source or fingerprint()
        command = [target['python'], '-B', '-m',
                   agent or 'AlphaBrain.research.dsol.execution.authority_worker',
                   '--root', target['root'], '--simulate']
        if agent:
            command += ['--binding', target['binding']]
        if prepare:
            command.append('--prepare')
        overrides = dict(environment or {})
        if set(overrides) - {'PYTHONPATH', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'TOKENIZERS_PARALLELISM', 'JAX_PLATFORMS', 'HF_HUB_OFFLINE',
                             'TRANSFORMERS_OFFLINE', 'OPENPI_DATA_HOME', 'PYTHONDONTWRITEBYTECODE',
                             'TMPDIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH'}:
            raise ValueError('Unapproved worker environment keys')
        if any(overrides.get(k, '1') != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
            raise ValueError('Model workers must remain offline')
        environment = dict(os.environ, PYTHONPATH=target['source'], CUDA_VISIBLE_DEVICES='',
                           WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
        environment.update(overrides)
        if target['transport'] == 'ssh':
            remote = ['env', 'CUDA_VISIBLE_DEVICES=', 'WANDB_MODE=disabled', 'HF_HUB_OFFLINE=1',
                      'PYTHONPATH='+target['source'], 'PYTHONDONTWRITEBYTECODE=1', *command]
            if overrides:
                remote = remote[:6] + [k+'='+v for k,v in overrides.items()] + remote[6:]
            command = ssh_command(target)+[shlex.join(remote)]
        elif target['transport'] != 'local':
            raise ValueError('Unsupported authority transport')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, env=environment)
        self.wire = Wire(self.process.stdout.fileno(), self.process.stdin.fileno(), timeout=10)

    def __enter__(self):
        previous = self.wire.timeout
        try:
            # PREPARE can validate large immutable shared inputs before HELLO.
            # This allowance never extends live execution grants/heartbeats.
            self.wire.timeout = previous if self.hello_timeout is None else self.hello_timeout
            self.hello = self.wire.receive()
            if self.hello.get('kind') != 'HELLO' or self.hello.get('source') != self.expected_source:
                raise RuntimeError('Worker source or handshake differs')
            return self
        except BaseException:
            self.close()
            raise
        finally:
            self.wire.timeout = previous

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Only the transport process created by this channel is stopped.
            # Remote uncertainty is retained, never inferred to be cancelled.
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()

    def __exit__(self, *args):
        if not self.detached or args[0] is not None:
            self.close()
