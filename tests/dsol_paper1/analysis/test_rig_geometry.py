import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from AlphaBrain.research.dsol.acquisition.camera_rig import orbit_rig
from AlphaBrain.research.dsol.analysis.rig_geometry import audit_mounts, camera_opencv


def sample():
    p=np.array([[-1., .35, 1.], [-1., -.35, 1.]])
    r=np.broadcast_to(Rotation.from_euler('y', -90, degrees=True).as_matrix(), (2,3,3))
    params=np.array([[0.,0.,1.], [30.,15.,1.2], [-20.,-10.,.8]])
    positions=[]; quaternions=[]
    for a,e,s in params:
        g=orbit_rig(p,r,azimuth_deg=a,elevation_deg=e,radius_scale=s)
        positions.append(np.vstack([g['positions'], [0.,0.,0.]]))
        q=Rotation.from_matrix(g['rotations']).as_quat()[:,[3,0,1,2]]
        quaternions.append(np.vstack([q,[1.,0.,0.,0.]]))
    return np.array(positions),np.array(quaternions),params


def test_saved_mount_audit_preserves_rigid_pair_and_untouched_wrist():
    p,q,params=sample(); result=audit_mounts(p,q,[0,1],params)
    np.testing.assert_allclose(result['baseline'],.7)
    assert result['checks']['central_ray_pivot_error_m'] < 1e-12
    assert result['checks']['untouched_mount_positions']


def test_audit_rejects_wrong_baseline_or_wrist():
    p,q,params=sample(); p[1,0,0] += .1
    with pytest.raises(ValueError): audit_mounts(p,q,[0,1],params)
    p,q,params=sample(); p[1,2,0] += .1
    with pytest.raises(ValueError): audit_mounts(p,q,[0,1],params)


def test_native_to_opencv_keeps_optical_direction():
    r=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
    c=camera_opencv(np.array([1.,2.,3.]),r)
    np.testing.assert_allclose(c[:3,2],-r[:,2])
    np.testing.assert_allclose(c[:3,3],[1.,2.,3.])
