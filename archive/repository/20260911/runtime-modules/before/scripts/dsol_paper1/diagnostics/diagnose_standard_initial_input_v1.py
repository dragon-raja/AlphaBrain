"""Read-only-to-experiment diagnosis of settled versus refreshed observations."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

from pathlib import Path
import tempfile
import numpy as np
import scripts.dsol_paper1.standard_initial_aa0_v1 as control
from scripts.dsol_paper1.standard_initialization_v1 import initialize, load_initial
from scripts.dsol_paper1.audit_libero_hdf5_restore import _configure_runtime
def main():
    r=control.read(control.ROOT/'release.json')
    config=Path(tempfile.mkdtemp(prefix='dsol-initial-diagnosis-'))
    _configure_runtime(control.core.RUNTIME,Path(r['states'][0]['hdf5']).parent.parent,config)
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference,install_camera_pose
    from scripts.dsol_paper1.scan_libero_hdf5_views import _restore_reference
    from evaluate_pi05_libero_plus_views import agentview_camera_calibration
    from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import masked_policy_observation
    for ci in [0,1,65]:
        spec=control.core.protocol_spec_at(r['protocol'],ci*32)
        with control.core.render_protocol(0):
            e=OffScreenRenderEnv(bddl_file_name=spec['bddl_file'],camera_names=('agentview','robot0_eye_in_hand'),
                                camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            obs,_=initialize(e,spec,load_initial(spec))
            reference=capture_camera_reference(e,camera_name='agentview',table_plane_z=control.read(r['catalog']['path'])['table_plane_z'])
            if spec['pose'] is not None:
                install_camera_pose(e,reference,spec['pose']);e.env._update_observables(force=True);obs=e.env._get_observations()
            with np.load(Path(spec['initial_asset_record']).parent/'policy_inputs.npz') as f:
                expected=f['external_images'][ci]
            def report(label,ob):
                request,_,_=masked_policy_observation(ob,prompt=spec['prompt'],resize_size=224,
                    eval_seed=spec['environment_seed'],camera_calibration=agentview_camera_calibration(e),sensor_control='both')
                actual=request['observation/image']
                print(ci,label,'equal',np.array_equal(actual,expected),'mean_abs',float(np.abs(actual.astype(float)-expected).mean()),
                      'actual',control.core.array_identity(actual),flush=True)
            report('rollout',obs)
            e.env._update_observables(force=True);report('forced',e.env._get_observations())
            _restore_reference(e,reference)
            if spec['pose'] is not None:install_camera_pose(e,reference,spec['pose'])
            e.env._update_observables(force=True);report('restore-reference',e.env._get_observations())
            e.close()


if __name__ == "__main__":
    main()
