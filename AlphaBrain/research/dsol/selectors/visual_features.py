"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

import numpy as np
from ..data.artifacts import sha
from .specification import CONTRACT


def encode_assets(root, out, weights):
    import torch
    from torchvision.models import resnet18

    assert sha(weights) == CONTRACT["encoder_sha256"]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def encode(images):
        output = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                x = torch.from_numpy(images[start : start + 16]).permute(0, 3, 1, 2).float() / 255
                output.append(model((x - mean) / std).numpy())
        return np.concatenate(output)

    external, context = [], []
    for i in range(32):
        with np.load(root / f"assets/aa0-a/{i:02d}/policy_inputs.npz") as z:
            e = encode(z["external_images"])
            w = encode(z["wrist_images"][:1])
        external.append(e)
        context.append(np.concatenate([e[0], w[0]]))
        if (i + 1) % 8 == 0:
            print("CPU image features", i + 1, "/32", flush=True)
    external, context = np.stack(external), np.stack(context)
    np.savez_compressed(out / "embeddings.npz", external=external, context=context)
    return external, context
