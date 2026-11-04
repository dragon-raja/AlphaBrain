"""Read-only RLDX pre-language visual capture, without importing RLDX.

This is a candidate tap, not production admission. The caller must establish
processor frame/camera order and verify native outputs on the actual checkpoint.
Joint cognition queries are never split or reinterpreted as camera patches.
"""
import inspect

import torch
from torch.nn import functional as F


class RLDXVisualCapture:
    """Capture one existing get_image_features call and pool a declared frame.

    Images must be in frame-major, then camera order, as in the reviewed RLDX
    processor. Temporal effects inside the native visual tower are preserved;
    selecting the current frame does not remove its temporal receptive field.
    """

    def __init__(self, model, *, cameras, frames, selected_frame, channels, merge_size):
        if (not isinstance(cameras, tuple) or not cameras
                or any(not isinstance(c, str) or not c for c in cameras)
                or len(set(cameras)) != len(cameras)
                or any(type(v) is not int or v < 1 for v in (frames, channels, merge_size))
                or type(selected_frame) is not int or not 0 <= selected_frame < frames):
            raise ValueError('Explicit camera/frame/channel/merge contract required')
        self.model, self.cameras = model, cameras
        self.frames, self.selected_frame = frames, selected_frame
        self.channels, self.merge_size = channels, merge_size
        self.active = self.used = False
        self.calls = 0
        self.result = None

    def _capture(self, bound, output):
        grid = bound.arguments['image_grid_thw']
        if (bound.arguments.get('num_frames', 1) != self.frames
                or bound.arguments.get('num_views', 1) != len(self.cameras)
                or not isinstance(grid, torch.Tensor)
                or grid.dtype not in (torch.int32, torch.int64)
                or tuple(grid.shape) != (self.frames * len(self.cameras), 3)):
            raise ValueError('Native frame/view/grid contract differs')
        sizes = grid.detach().cpu().tolist()
        if (not isinstance(output, tuple) or len(output) != 2
                or not isinstance(output[0], (tuple, list))
                or len(output[0]) != len(sizes)):
            raise ValueError('Native per-image feature boundaries missing')
        result = {}
        for i, (tokens, (t, h, w)) in enumerate(zip(output[0], sizes)):
            if (t != 1 or h % self.merge_size or w % self.merge_size
                    or min(h // self.merge_size, w // self.merge_size) < 2
                    or not isinstance(tokens, torch.Tensor)
                    or tuple(tokens.shape) != ((h // self.merge_size)*(w // self.merge_size), self.channels)
                    or not torch.isfinite(tokens).all()):
                raise ValueError('Finite per-frame merged spatial tokens required')
            if i // len(self.cameras) == self.selected_frame:
                # Qwen merger outputs a raster grid of merged patch blocks.
                spatial = tokens.detach().float().T.reshape(
                    1, self.channels, h // self.merge_size, w // self.merge_size)
                pooled = F.adaptive_avg_pool2d(spatial, (2, 2))
                result[self.cameras[i % len(self.cameras)]] = (
                    pooled.permute(0, 2, 3, 1).reshape(1, 4, self.channels).contiguous().clone())
        self.result = result

    def __enter__(self):
        marker = '_dsol_rldx_visual_capture'
        if self.used or hasattr(self.model, marker):
            raise RuntimeError('Serialized single-use capture required')
        original = self.model.get_image_features
        signature = inspect.signature(original)
        self.had = 'get_image_features' in vars(self.model)
        self.previous = vars(self.model).get('get_image_features')

        def observed(*args, **kwargs):
            self.calls += 1
            if self.calls != 1:
                raise ValueError('Exactly one native image encoding per request required')
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            output = original(*args, **kwargs)
            self._capture(bound, output)
            return output  # Preserve object identity, deepstack outputs and RNG.

        self.model.get_image_features = observed
        setattr(self.model, marker, True)
        self.active = self.used = True
        return self

    def __exit__(self, kind, value, traceback):
        if self.had:
            self.model.get_image_features = self.previous
        else:
            delattr(self.model, 'get_image_features')
        delattr(self.model, '_dsol_rldx_visual_capture')
        self.active = False
        if kind is not None:
            self.result = None
        return False

    def features(self):
        if self.active or self.calls != 1 or self.result is None:
            raise ValueError('No complete fresh native capture')
        return {k: v.clone() for k, v in self.result.items()}
