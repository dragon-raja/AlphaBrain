"""Build a bounded, outcome-blind observation/label diagnostic panel.

Uses existing rendered training records only. No rendering, inference or label
editing. Natural state pairs are canonical-view comparisons, not image swaps.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

from AlphaBrain.common.pair_records import read_record
from .artifacts import read, sha, write


def temporal_pair(records, frame_key, minimum_gap):
    """First eligible episode is chosen by caller; choose quarter/three-quarter frames."""
    ordered=sorted(records,key=lambda r:r[frame_key])
    low,high=ordered[0][frame_key],ordered[-1][frame_key]
    selected=[min(ordered,key=lambda r:(abs(r[frame_key]-(low+fraction*(high-low))),r[frame_key]))
              for fraction in (.25,.75)]
    if selected[1][frame_key]-selected[0][frame_key]<minimum_gap:
        raise ValueError('Insufficient temporal separation')
    return selected


def modality_vector(table, index, mapping, names):
    return np.concatenate([np.asarray(table[mapping[k]['original_key']][index],np.float32)
        [mapping[k]['start']:mapping[k]['end']] for k in names])


def robocasa_source(dataset, row):
    import pyarrow.parquet as pq
    dataset=Path(dataset);episode=int(row['episode_index']);frame=int(row['frame_index'])
    info=read(dataset/'meta/info.json');mod=read(dataset/'meta/modality.json')
    path=dataset/info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode)
    table=pq.read_table(path).to_pydict()
    if int(table['frame_index'][frame])!=frame or int(table['episode_index'][frame])!=episode:
        raise ValueError('Source frame identity mismatch')
    state=modality_vector(table,frame,mod['state'],('end_effector_position_relative','end_effector_rotation_relative','base_position','base_rotation','gripper_qpos'))
    indices=np.minimum(np.arange(frame,frame+50),len(table['frame_index'])-1)
    actions=np.stack([modality_vector(table,int(i),mod['action'],('end_effector_position','end_effector_rotation','gripper_close','base_motion','control_mode')) for i in indices])
    tasks={j['task_index']:j['task'] for j in map(json.loads,(dataset/'meta/tasks.jsonl').read_text().splitlines())}
    instruction=tasks[table['annotation.human.task_description'][frame]]
    return state,actions,instruction,dict(parquet=str(path),parquet_sha256=sha(path),
        padded_action_steps=int(np.sum(indices==len(table['frame_index'])-1))-int(frame+50>=len(table['frame_index'])))


def _save(output, benchmark, records, images, states, actions):
    payload=output/f'{benchmark}.npz'
    np.savez_compressed(payload,images=np.stack(images),states=np.stack(states),actions=np.stack(actions))
    return dict(records=records,payload=str(payload),payload_sha256=sha(payload),
        source_scope='Existing training records; in-sample diagnostic, NOT heldout generalization',
        views=['canonical','broad_a','broad_b'])


def build_robocasa(cfg,output):
    root=Path(cfg['robocasa_training']);release=root/'matched-data-release-20260917'
    manifest=read(release/'canonical_matched.json')
    ledger=list(map(json.loads,Path(manifest['source_window_ledger']).read_text().splitlines()))
    populations=read(cfg['robocasa_population']);tasks=sorted({j['task'] for j in populations['jobs']})
    exposure={arm:{r['source_window_id']:r for r in map(json.loads,(release/f'{arm}.jsonl').read_text().splitlines())}
        for arm in ('canonical_matched','broad_exterior_matched')}
    records=[];images=[];states=[];actions=[]
    for task in tasks:
        groups=defaultdict(list)
        for row in ledger:
            if row['dataset_identity'].split('/')[1]==task:groups[row['episode_index']].append(row)
        selected=None
        for episode in sorted(groups):
            try:selected=temporal_pair(groups[episode],'frame_index',50);break
            except ValueError:continue
        if selected is None:raise ValueError('No separated source-window pair: '+task)
        for phase,row in enumerate(selected):
            source=row['source_window_id'];can=exposure['canonical_matched'][source]['exposures']
            broad=exposure['broad_exterior_matched'][source]['exposures'];views=[can[0],*broad]
            assert len(can)==2 and len(broad)==2
            assert can[0]['view_id']==can[1]['view_id']=='canonical'
            assert len({v['image_files']['wrist'] for v in views})==1
            dataset=manifest['data_dirs'][row['dataset_index']]['path']
            state,action,prompt,source_info=robocasa_source(dataset,row)
            imgs=np.stack([np.stack([np.array(Image.open(v['image_files'][key]).convert('RGB')) for key in ('left','right','wrist')]) for v in views])
            images.append(imgs);states.append(state);actions.append(action)
            records.append(dict(task=task,phase=phase,episode_id=int(episode),frame=int(row['frame_index']),
                sample_id=source,prompt=prompt,view_ids=[v['view_id'] for v in views],
                source=row,source_info=source_info,image_sources=[v['image_files'] for v in views],
                image_hashes=[{k:sha(v) for k,v in row['image_files'].items()} for row in views]))
    return _save(output,'robocasa',records,images,states,actions)


def build_libero(cfg,output):
    import h5py
    root=Path(cfg['libero_training']);manifest=read(root/'manifest.json')
    records=[];images=[];states=[];actions=[]
    for shard_meta in sorted(manifest['shards'],key=lambda s:s['task_id']):
        shard=root/shard_meta['path'];meta=read(shard/'manifest.json');groups=defaultdict(list)
        for row in map(json.loads,(shard/'records.jsonl').read_text().splitlines()):
            if row['split']=='train':groups[row['episode_id']].append(row)
        selected=None
        for episode in sorted(groups):
            try:selected=temporal_pair(groups[episode],'frame',10);break
            except ValueError:continue
        if selected is None:raise ValueError('No separated LIBERO source pair')
        with (shard/'pairs.bin').open('rb') as stream,h5py.File(meta['source_hdf5'],'r') as h5:
            for phase,row in enumerate(selected):
                stored=read_record(stream,offset=row['offset']);h=stored['header'];im=stored['images']
                demo=h5['data'][h['demo']];idx=np.minimum(np.arange(h['frame'],h['frame']+10),len(demo['actions'])-1)
                expected=np.asarray(demo['actions'])[idx];actual=np.asarray(h['action_chunk'],np.float32)
                if not np.allclose(expected,actual,atol=1e-7,rtol=1e-7):raise ValueError('LIBERO action label mismatch')
                assert h['sample_id']==row['sample_id'] and h['frame']==row['frame']
                image=np.stack([np.stack([im[key],im['wrist']]) for key in ('canonical','broad_a','broad_b')])
                images.append(image);states.append(np.asarray(h['robot_state'],np.float32));actions.append(actual)
                records.append(dict(task=shard_meta['task_id'],phase=phase,episode_id=episode,frame=h['frame'],
                    sample_id=h['sample_id'],prompt=h['language_instruction'],view_ids=['canonical',row['pose_a_id'],row['pose_b_id']],
                    source=dict(shard=str(shard/'pairs.bin'),offset=row['offset'],records_sha256=sha(shard/'records.jsonl')),
                    source_info=dict(hdf5=meta['source_hdf5'],demo=h['demo'],action_label_check='PASS'),
                    camera_poses=[h['canonical_camera_to_world_opencv'],h['camera_a_to_world_opencv'],h['camera_b_to_world_opencv']],
                    pose_parameters=[None,h['pose_a'],h['pose_b']]))
    return _save(output,'libero',records,images,states,actions)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    result=dict(protocol=cfg,config_sha256=sha(args.config),source_sha256=sha(__file__),
        no_outcome_selection=True,no_new_rendering=True,no_new_rollouts=True,benchmarks={})
    for name,builder in [('robocasa',build_robocasa),('libero',build_libero)]:
        result['benchmarks'][name]=builder(cfg,out)
    write(out/'panel.json',result)
    print(json.dumps({k:len(v['records']) for k,v in result['benchmarks'].items()}))


if __name__=='__main__':main()
