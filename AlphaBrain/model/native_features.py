"""Read-only eager method capture and explicit spatial pooling.

No model imports, instantiation, downloads or extra forward passes. CUDA graph
internals require a method called outside graph replay, not a stale Python hook.
"""
import math
import torch
from torch.nn import functional as F


def pool_spatial_grid(tokens,grid):
    if (not isinstance(grid,tuple) or len(grid)!=3 or grid[0]!=1
            or any(type(n) is not int or n<1 for n in grid) or min(grid[1:])<2
            or tokens.ndim!=3 or tokens.shape[0]!=1 or math.prod(grid)!=tokens.shape[1]
            or not torch.isfinite(tokens).all()):raise ValueError('Explicit finite single-frame spatial grid required')
    x=tokens.detach().float().reshape(1,grid[1],grid[2],tokens.shape[-1]).permute(0,3,1,2)
    return F.adaptive_avg_pool2d(x,(2,2)).permute(0,2,3,1).reshape(1,-1).contiguous()


class NativeVisionCapture:
    """One fresh eager native vision result; serialized instance, single use."""
    def __init__(self,model,*,shape):
        if (not isinstance(shape,tuple) or len(shape)!=3 or any(type(n) is not int or n<1 for n in shape)):
            raise ValueError('Explicit native view/token/channel shape required')
        self.model,self.shape=model,shape;self.result=None;self.used=False;self.calls=0;self.active=False

    def __enter__(self):
        name='_dsol_native_vision_capture'
        if self.used or hasattr(self.model,name):raise RuntimeError('Serialized single-use native capture required')
        original=self.model.extract_vision_feature
        self.previous=vars(self.model).get('extract_vision_feature');self.had='extract_vision_feature' in vars(self.model)
        def observed(*args,**kwargs):
            self.calls+=1
            if self.calls!=1:raise ValueError('Native vision must execute once per scored call')
            output=original(*args,**kwargs)
            if not isinstance(output,torch.Tensor) or tuple(output.shape)!=self.shape or not torch.isfinite(output).all():
                raise ValueError('Native vision token contract differs')
            self.result=output.detach().clone()
            return output
        self.model.extract_vision_feature=observed;setattr(self.model,name,True)
        self.used=self.active=True;return self

    def __exit__(self,kind,value,tb):
        if self.had:self.model.extract_vision_feature=self.previous
        else:delattr(self.model,'extract_vision_feature')
        delattr(self.model,'_dsol_native_vision_capture');self.active=False
        if kind is not None:self.result=None
        return False

    def features(self):
        if self.active or not self.used or self.calls!=1 or self.result is None:
            raise ValueError('No fresh successful native vision result')
        return self.result.clone()
