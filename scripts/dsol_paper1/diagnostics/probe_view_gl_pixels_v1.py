#!/usr/bin/env python3
"""Render one captured pose repeatedly; isolate GL state from physics and policy."""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'vla_shared'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',default='goal_wine_rack'); parser.add_argument('--call',type=int,default=22)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--label',required=True)
    parser.add_argument('--offsamples',type=int)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--modes',default='baseline,no_dither,no_multisample,neither')
    args=parser.parse_args(); out=args.root/'render-fix-window-v1'/args.label; out.mkdir()
    spec=json.loads((args.root/(args.task+'-spec.json')).read_text())
    runtime=Path('/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus')
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths
    _configure_runtime(runtime,Path(spec['hdf5']).parent.parent,out/'libero-config')
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference,install_camera_pose
    import h5py
    import mujoco
    from OpenGL import GL
    with h5py.File(spec['hdf5'],'r') as h:
        data=h['data']; demo=data[spec['demo_name']]
        xml,_=_rewrite_model_paths(_decode(demo.attrs['model_file']),runtime)
        bddl=Path(_decode(data.attrs['bddl_file_name'])).name
    with np.load(args.root/(args.task+'-live-a')/f'call-{args.call:04d}.npz',allow_pickle=False) as capture:
        state=capture['physical_flat'].copy()
    env=OffScreenRenderEnv(bddl_file_name=str(runtime/'libero/libero/bddl_files'/spec['suite']/bddl),
        camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,camera_widths=256,render_gpu_device_id=args.gpu)
    try:
        env.seed(spec['environment_seed']); env.reset(); env.reset_from_xml_string(xml); env.set_init_state(state)
        reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=json.loads(Path(spec['catalog']).read_text())['table_plane_z'])
        if spec.get('pose') is not None: install_camera_pose(env,reference,spec['pose'])
        sim=env.env.sim; context=sim._render_context_offscreen; context.gl_ctx.make_current()
        if args.offsamples is not None:
            sim.model.vis.quality.offsamples=args.offsamples
            context.con.free()
            context._set_mujoco_context_and_buffers()
        flags=context.scn.flags.copy(); results={}
        for mode in args.modes.split(','):
            context.scn.flags[:]=flags
            (GL.glDisable if mode in ('no_dither','neither') else GL.glEnable)(GL.GL_DITHER)
            (GL.glDisable if mode in ('no_multisample','neither') else GL.glEnable)(GL.GL_MULTISAMPLE)
            if mode=='no_shadow': context.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
            if mode=='no_reflection': context.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
            images=[]
            for i in range(args.repeats):
                (GL.glDisable if mode in ('no_dither','neither') else GL.glEnable)(GL.GL_DITHER)
                (GL.glDisable if mode in ('no_multisample','neither') else GL.glEnable)(GL.GL_MULTISAMPLE)
                sim.render(256,256,camera_name='agentview')
                (GL.glDisable if mode in ('no_dither','neither') else GL.glEnable)(GL.GL_DITHER)
                (GL.glDisable if mode in ('no_multisample','neither') else GL.glEnable)(GL.GL_MULTISAMPLE)
                images.append(sim.render(256,256,camera_name='robot0_eye_in_hand').copy())
                GL.glFinish()
            images=np.stack(images)
            np.savez_compressed(out/(mode+'.npz'),images=images,physical_flat=np.asarray(env.get_sim_state()))
            hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in images]
            delta=np.abs(images.astype(np.int16)-images[0].astype(np.int16))
            results[mode]={'distinct_frames':len(set(hashes)),'sha256':hashes,
                'max_channel_difference':int(delta.max()),'changed_channel_values':int(np.count_nonzero(delta)),
                'dither':bool(GL.glIsEnabled(GL.GL_DITHER)),'multisample':bool(GL.glIsEnabled(GL.GL_MULTISAMPLE))}
        receipt={'status':'PASS_STATIC_RENDER_PROBE','gpu':args.gpu,'task':args.task,'call':args.call,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'renderer':GL.glGetString(GL.GL_RENDERER).decode(),'version':GL.glGetString(GL.GL_VERSION).decode(),
            'offsamples':int(context.con.offSamples),'results':results}
        (out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:{x:y for x,y in v.items() if x!='sha256'} for k,v in results.items()}),flush=True)
    finally: env.close()


if __name__=='__main__': main()
