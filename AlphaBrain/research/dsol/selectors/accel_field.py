"""Small current-observation baseline for a fixed, aligned camera catalog.

No queried-target images, task IDs, success labels or evaluation preprocessing.
This discrete-field diagnostic assumes known task-reference camera coordinates;
it is not yet an unknown-reference, continuous SE(3) navigation policy.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AccelFieldRidge:
    image_mean: np.ndarray
    components: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    use_image: bool

    def raw_features(self, images, coordinates):
        p = (np.asarray(coordinates) - self.coordinate_mean) / self.coordinate_scale
        parts = [p, p * p]
        if self.use_image:
            z = (np.asarray(images) - self.image_mean) @ self.components.T
            parts.extend([z, (z[..., :, None] * p[..., None, :]).reshape(len(p), -1)])
        return np.concatenate(parts, axis=-1)

    def predict(self, images, coordinates):
        x = (self.raw_features(images, coordinates) - self.feature_mean) / self.feature_scale
        return self.target_mean + x @ self.weights


def fit_accel_field(images, coordinates, target_fields, *, use_image=True, image_dim=8, alpha=10.):
    """Fit only on explicitly supplied training rows; each output is one pose.

    Targets are within-field standardized for relative shape, not probabilities.
    All starts must be equally represented per training initial in the caller.
    """
    images, coordinates, y = (np.asarray(a, dtype=np.float64) for a in (images, coordinates, target_fields))
    if images.ndim != 2 or coordinates.shape != (len(images), 3) or y.ndim != 2 or len(y) != len(images):
        raise ValueError('Expected aligned rows: image features, 3D coordinates, target fields')
    if not all(np.isfinite(a).all() for a in (images, coordinates, y)) or alpha <= 0:
        raise ValueError('Finite data and positive regularization required')
    im = images.mean(0)
    components = np.linalg.svd(images - im, full_matrices=False)[2][:image_dim] if use_image else np.empty((0, images.shape[1]))
    ps = coordinates.std(0)
    ps[ps < 1e-8] = 1.
    model = AccelFieldRidge(im, components, coordinates.mean(0), ps, None, None, None, None, use_image)
    x = model.raw_features(images, coordinates)
    model.feature_mean = x.mean(0)
    model.feature_scale = x.std(0)
    model.feature_scale[model.feature_scale < 1e-8] = 1.
    x = (x - model.feature_mean) / model.feature_scale
    y = y - y.mean(1, keepdims=True)
    y = y / np.maximum(y.std(1, keepdims=True), 1e-8)
    model.target_mean = y.mean(0)
    model.weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - model.target_mean))
    return model
