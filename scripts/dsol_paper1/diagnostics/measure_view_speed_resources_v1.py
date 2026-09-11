#!/usr/bin/env python3
"""Read-only resource samples during the predeclared concurrency benchmark."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/full64-extension-v2/early-speed-gate-v1')
def main():
    deadline=time.time()+7600
    path=ROOT/'resource-samples.csv'
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['utc','mode','gpu','utilization_percent','memory_used_mib','power_watts'])
        writer.writeheader()
        while time.time()<deadline:
            state=json.loads((ROOT/'status.json').read_text())
            if state['status'] in ('PASS_COMPLETE_SPEED_MEASUREMENT','STOPPED_REQUIRES_REVIEW'): break
            if state['status']=='BENCHMARKING':
                output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                for line in output.splitlines():
                    gpu,util,memory,power=[value.strip() for value in line.split(',')]
                    writer.writerow(dict(utc=datetime.now(timezone.utc).isoformat(),mode=state['mode']['label'],gpu=gpu,
                        utilization_percent=util,memory_used_mib=memory,power_watts=power))
                stream.flush()
            time.sleep(2)
    print('RESOURCE_SAMPLING_COMPLETE',path)


if __name__ == "__main__":
    main()
