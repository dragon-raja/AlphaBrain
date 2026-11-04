"""Shared diagnostic plot for legacy seven-dimensional action clients."""
from typing import Sequence
import numpy as np

def visualize_epoch(
    self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
) -> None:
    from matplotlib import pyplot as plt
    images = [self._resize_image(image) for image in images]
    ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

    img_strip = np.concatenate(np.array(images[::3]), axis=1)

    # set up plt figure
    figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # plot actions
    pred_actions = np.array(
        [
            np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
            for a in predicted_raw_actions
        ]
    )
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        # actions have batch, horizon, dim, in this example we just take the first action for simplicity
        axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time in one episode")

    axs["image"].imshow(img_strip)
    axs["image"].set_xlabel("Time in one episode (subsampled)")
    plt.legend()
    plt.savefig(save_path)
