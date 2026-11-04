"""Split contextual VLM image tokens using the real processor layout.

This is a portable preparation primitive, not a model-specific runtime adapter.
It never treats sequence order as a spatial grid, duplicates a camera, includes
padding/text tokens, or silently accepts temporal history as a single image.
Whether image tokens are instruction-conditioned depends on the VLM's causal
ordering and must be declared by the caller, not inferred by this function.
"""
import numpy as np


def split_image_tokens(hidden, input_ids, image_grid_thw, camera_ids, *,
                       image_token_id, spatial_merge_size, attention_mask):
    """Return copied per-camera (tokens, THW grid) from a batch-one VLM result.

    ``image_grid_thw`` is the processor's pre-merge patch grid in actual image
    input order. Each image must occupy one contiguous placeholder-token block.
    Caller binds query/input/history/model/layer identities in the feature packet.
    Output can use the existing explicit 2x2 pooling and native feature contract.
    """
    hidden=np.asarray(hidden)
    ids=np.asarray(input_ids)
    grids=np.asarray(image_grid_thw)
    mask=np.asarray(attention_mask)
    if (hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[2]<1
            or hidden.dtype!=np.float32 or not np.isfinite(hidden).all()
            or ids.shape!=hidden.shape[:2] or not np.issubdtype(ids.dtype,np.integer)
            or mask.shape!=ids.shape or not np.isin(mask,[0,1]).all()):
        raise ValueError('Finite float32 batch-one hidden states and aligned IDs/mask required')
    if (not isinstance(camera_ids,tuple) or not camera_ids
            or any(not isinstance(c,str) or not c for c in camera_ids)
            or len(set(camera_ids))!=len(camera_ids)
            or grids.shape!=(len(camera_ids),3)
            or not np.issubdtype(grids.dtype,np.integer) or np.any(grids<=0)
            or type(image_token_id) is not int or type(spatial_merge_size) is not int
            or spatial_merge_size<1):
        raise ValueError('Real distinct camera identities and processor grid required')
    if np.any(grids[:,0]!=1) or np.any(grids[:,1:]%spatial_merge_size):
        raise ValueError('Single current image and exactly divisible spatial merge required')
    rows=grids[:,1:]//spatial_merge_size
    if np.any(rows<2):raise ValueError('At least 2x2 real spatial tokens per image required')
    image_positions=np.flatnonzero(ids[0]==image_token_id)
    if (len(image_positions)!=int(np.prod(rows,axis=1).sum())
            or not mask[0,image_positions].all()):
        raise ValueError('Image token counts or padding disagree with processor grids')
    result=[];offset=0
    for camera,(height,width) in zip(camera_ids,rows):
        count=int(height*width);positions=image_positions[offset:offset+count]
        if not np.array_equal(positions,np.arange(positions[0],positions[0]+count)):
            raise ValueError('Each image requires one contiguous token block in input order')
        result.append(dict(camera_id=camera,grid=(1,int(height),int(width)),
            tokens=hidden[:,positions,:].copy()))
        offset+=count
    return tuple(result)
