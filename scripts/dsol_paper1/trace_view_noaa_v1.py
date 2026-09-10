#!/usr/bin/env python3
"""Opt-in diagnostic rendering protocol; never modifies the frozen evaluator."""
import hashlib
import json
from pathlib import Path
import sys

def main():
    import trace_view_repeatability_v1 as trace
    # Configure the runtime before importing its robosuite bindings.
    from audit_libero_hdf5_restore import _configure_runtime
    spec_path = Path(sys.argv[sys.argv.index('--spec') + 1])
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    spec = json.loads(spec_path.read_text())
    _configure_runtime(Path('/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus'),
                       Path(spec['hdf5']).parent.parent, output.parent/'noaa-libero-config')
    from robosuite.utils.binding_utils import MjRenderContext
    original = MjRenderContext._set_mujoco_context_and_buffers
    observed = []
    def initialize(context):
        context.model.vis.quality.offsamples = 0
        result = original(context)
        assert context.con.offSamples == 0
        observed.append(int(context.con.offSamples))
        return result
    MjRenderContext._set_mujoco_context_and_buffers = initialize
    try:
        trace.main()
        assert observed, 'No render context was instrumented'
        trace.write_json(output/'render-protocol.json', {
            'protocol': 'diagnostic-noaa-v1', 'offsamples': observed,
            'observation_protocol_changed': True, 'scientific_ledger': False,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    finally:
        MjRenderContext._set_mujoco_context_and_buffers = original

if __name__ == '__main__':
    main()
