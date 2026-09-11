#!/usr/bin/env python3
"""Recover JSON serialization only; preserve the frozen analyzer and ledgers."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import scripts.dsol_paper1.analysis.analyze_render_protocol_bridge_v1 as analyzer
from scripts.dsol_paper1.render_bridge_common_v1 import ROOT, sha, verify_release


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def main():
    verify_release()
    target=ROOT/'analysis-recovered-v1.json'
    receipt=ROOT/'analysis-recovery-v1.json'
    if target.exists() or receipt.exists() or (ROOT/'summary_zh.md').exists():
        raise RuntimeError('Recovery outputs already exist; inspect instead of overwriting')
    damaged=ROOT/'analysis.json'
    old_digest=sha(damaged)
    frozen_digest=sha(Path(analyzer.__file__))
    original_writer=analyzer.write_new
    def recovered_writer(path,value):
        if Path(path)!=damaged:
            raise ValueError('Unexpected analyzer output')
        # Serialize before opening: type errors cannot leave another partial file.
        payload=json.dumps(value,default=json_value,allow_nan=False,sort_keys=True,indent=2)+'\n'
        with target.open('x') as stream: stream.write(payload)
    analyzer.write_new=recovered_writer
    try: analyzer.main()
    finally: analyzer.write_new=original_writer
    assert sha(damaged)==old_digest and sha(Path(analyzer.__file__))==frozen_digest
    verify_release()
    result=dict(status='PASS_SERIALIZATION_RECOVERY',utc=datetime.now(timezone.utc).isoformat(),
                analysis=str(target),analysis_sha256=sha(target),
                preserved_partial_analysis_sha256=old_digest,frozen_analyzer_sha256=frozen_digest,
                recovery_script_sha256=sha(Path(__file__)),new_gpu_calls=0,
                scope='NumPy scalar JSON conversion only; no changes to estimators, release, or raw data')
    with receipt.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
