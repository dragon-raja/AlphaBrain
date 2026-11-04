"""Explicit GR1 task construction; import never creates a simulator.

An isolated source view binds assets BEFORE upstream object registries import.
The convenience upstream Gym creator drops layout/style arguments, so this
adapter constructs the registered class directly and verifies the actual scene.
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import os
import sys
import xml.etree.ElementTree as ET


def prepare_source_view(source, destination, assets):
    """Create links in a new owned directory, leaving source/assets untouched."""
    source, destination, assets = Path(source).resolve(), Path(destination), Path(assets).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Source view already exists; audit before reuse")
    if not (source / "robocasa/models/__init__.py").is_file() or not assets.is_dir():
        raise ValueError("Pinned task source and composed assets required")
    destination.mkdir(parents=True)

    def link_tree(src, dst, depth):
        for child in sorted(src.iterdir()):
            if child.name == "__pycache__":
                continue
            target = dst / child.name
            if (depth, child.name) in ((0, "robocasa"), (1, "models")):
                target.mkdir()
                link_tree(child, target, depth + 1)
            elif depth == 2 and child.name == "assets":
                target.symlink_to(assets, target_is_directory=True)
            else:
                target.symlink_to(child, target_is_directory=child.is_dir())

    link_tree(source, destination, 0)
    return {"source": str(source), "view": str(destination.resolve()), "assets": str(assets)}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    layout: int
    style: int
    seed: int
    asset_root: str
    active_head: bool
    control_hz: int
    horizon: int

    def validate(self):
        if (
            not isinstance(self.task, str)
            or not self.task
            or type(self.layout) is not int
            or self.layout not in (0, 2, 4, 5)
            or type(self.style) is not int
            or not 0 <= self.style < 12
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.active_head) is not bool
            or any(type(v) is not int or v <= 0 for v in (self.control_hz, self.horizon))
            or not Path(self.asset_root).is_absolute()
        ):
            raise ValueError("Explicit task, layout/style, seed, clock and asset binding required")


def process_fingerprint():
    """Python hash seed affects upstream list(set(...)) object/task generation.

    Record actual hash outputs, not just an environment variable that could
    have been changed after interpreter startup. Task seed alone is insufficient.
    """
    return dict(
        python=list(sys.version_info[:3]),
        hash_seed=os.environ.get("PYTHONHASHSEED"),
        hash_probe=[hash(x) for x in ("GR1_REPLAY_V1", "object_category", "target_container")],
    )


def create_task(
    spec,
    *,
    guard,
    controller_config,
    render=False,
    allowed_empty_categories=(),
    expected_process_fingerprint=None,
):
    """Explicit construction includes upstream CPU settling; no policy calls.

    Rendering requires an external authorized backend; this preparation factory
    only accepts render=False. Future GPU binding must be separately admitted.
    """
    spec.validate()
    guard()
    if (
        expected_process_fingerprint is not None
        and process_fingerprint() != expected_process_fingerprint
    ):
        raise ValueError("Task interpreter/hash seed differs before construction")
    if render is not False:
        raise ValueError("GPU/render admission is not available in this CPU factory")
    import robocasa
    from robosuite.environments.base import REGISTERED_ENVS

    if Path(robocasa.models.assets_root).resolve() != Path(spec.asset_root).resolve():
        raise ValueError(
            "Import task from the asset-bound source view, before registries initialize"
        )
    # Upstream may silently drop a distractor when its enabled asset category
    # has no files. Missing assets must not change our task distribution.
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES

    missing = [
        f"{registry}/{category}"
        for category, registries in OBJ_CATEGORIES.items()
        for registry, metadata in registries.items()
        if registry in ("objaverse", "sketchfab", "lightwheel") and not metadata.mjcf_paths
    ]
    if (
        not isinstance(allowed_empty_categories, tuple)
        or len(set(allowed_empty_categories)) != len(allowed_empty_categories)
        or any(not isinstance(x, str) for x in allowed_empty_categories)
    ):
        raise ValueError("Explicit frozen upstream asset exception identities required")
    if set(missing) != set(allowed_empty_categories):
        raise FileNotFoundError("Empty enabled object categories: " + ", ".join(sorted(missing)))
    cls = REGISTERED_ENVS[spec.task]
    if spec.layout not in cls.VALID_LAYOUTS:
        raise ValueError("Layout does not belong to this task")
    cfg = copy.deepcopy(controller_config)
    if cfg.get("type") != "BASIC" or cfg.get("control_delta") is not False:
        raise ValueError("Explicit absolute BASIC operation controller required")
    if spec.active_head:
        from robosuite.models.robots import GR1
        from .head_layout import restore_head_actuation

        reference = GR1(idn=0).get_xml()

        class HeadTask(cls):
            def edit_model_xml(self, xml):
                xml = super().edit_model_xml(xml)
                return restore_head_actuation(xml, reference, naming_prefix="robot0_")[0]

            def _pre_action(self, action, policy_step=False):
                super()._pre_action(action, policy_step=policy_step)
                control = getattr(self, "search_head_control", None)
                if control is not None:
                    control.apply()

        cls = HeadTask
    env = None
    try:
        env = cls(
            robots="GR1ArmsAndWaistFourierHands",
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            camera_names=["egoview"],
            camera_widths=1280,
            camera_heights=800,
            seed=spec.seed,
            layout_and_style_ids=[(spec.layout, spec.style)],
            randomize_cameras=False,
            translucent_robot=False,
            control_freq=spec.control_hz,
            horizon=spec.horizon,
            ignore_done=False,
        )
        if (env.layout_id, env.style_id) != (spec.layout, spec.style):
            raise ValueError("Constructed scene differs from manifest")
        guard()
        return env
    except BaseException:
        if env is not None:
            env.close()
        raise


def task_inventory(env):
    """Actual constructed model files/objects, not all unused kitchen assets."""
    xml = env.sim.model.get_xml()
    refs = []
    for el in ET.fromstring(xml).iter():
        name = el.get("file")
        if name:
            path = Path(name)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Unresolved compiled asset: " + name)
            refs.append(str(path.resolve()))
    return dict(
        layout=int(env.layout_id),
        style=int(env.style_id),
        process_fingerprint=process_fingerprint(),
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        files=sorted(set(refs)),
        objects=sorted(env.objects),
        action_dim=int(env.action_dim),
        nq=int(env.sim.model.nq),
        nu=int(env.sim.model.nu),
    )
