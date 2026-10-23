"""Explicit campaign permit, cross-host stop and restart deadline checks."""
import json
from pathlib import Path
import time
import threading
from contextlib import contextmanager
from .manifest import file_digest
from .queue import _durable_write


class CampaignGuard:
    def __init__(self,manifest,host,stage,*,clock=time.time):
        self.path=Path(manifest);self.config=json.loads(self.path.read_text())
        self.host,self.stage,self.clock=host,stage,clock
        self.root=Path(self.config['control']);self.digest=file_digest(self.path)
        if host not in self.config['hosts'] or stage not in ('admission','capacity','formal'):
            raise ValueError('Unknown campaign host/stage')

    def check(self,*,dispatch=False):
        # Missing/unreadable shared state is an error, not evidence of no failure.
        if not self.root.is_dir():raise RuntimeError('Campaign control is unavailable')
        if list(self.root.glob('failure-*.json')):raise RuntimeError('Campaign failed; no automatic retry')
        permit=json.loads((self.root/'permit.json').read_text())
        now=self.clock()
        if (permit.get('authorization')!='EXPLICIT_CAMPAIGN_EXECUTION'
                or permit['manifest_sha256']!=self.digest or permit['stage']!=self.stage
                or not permit['issued_at']<=now<permit['finish_until']
                or not permit['issued_at']<permit['dispatch_until']<permit['finish_until']):
            raise RuntimeError('Missing, changed, stale or expired campaign permit')
        deadline=self.config['restart_deadline_unix']
        if (permit['dispatch_until']>deadline-21600 or permit['finish_until']>deadline-14400):
            raise RuntimeError('Restart drain/backup reserve was not retained')
        if dispatch and now>=permit['dispatch_until']:return False
        if now-permit['issued_at']>120:
            for host in self.config['hosts']:
                done=self.root/f'done-{host}-{self.stage}.json'
                if done.exists():
                    value=json.loads(done.read_text())
                    if (value.get('manifest_sha256')==self.digest and value.get('stage')==self.stage
                            and value.get('status') in ('COMPLETE','INTERRUPTED')):continue
                heartbeat=json.loads((self.root/f'heartbeat-{host}.json').read_text())
                if (heartbeat['manifest_sha256']!=self.digest or heartbeat['stage']!=self.stage
                        or not -5<=now-heartbeat['unix_time']<=60):
                    raise RuntimeError('Campaign coordinator heartbeat missing/stale')
        return True

    def heartbeat(self):
        _durable_write(self.root/f'heartbeat-{self.host}.json',dict(manifest_sha256=self.digest,
            host=self.host,stage=self.stage,unix_time=self.clock()))

    @contextmanager
    def lease(self,interval=10):
        """Keep peers informed during long source/weight validation as well as polling."""
        stopped=threading.Event();errors=[]
        self.heartbeat()
        def renew():
            while not stopped.wait(interval):
                try:self.heartbeat()
                except BaseException as error:
                    errors.append(error);break
        thread=threading.Thread(target=renew,name='campaign-heartbeat',daemon=True)
        thread.start()
        try:
            yield
            if errors:raise RuntimeError('Campaign heartbeat write failed') from errors[0]
        finally:
            stopped.set();thread.join(timeout=max(1,interval*2))

    def fail(self,error):
        path=self.root/f'failure-{self.host}.json'
        if not path.exists():_durable_write(path,dict(host=self.host,stage=self.stage,
            manifest_sha256=self.digest,error=repr(error),unix_time=self.clock()))
