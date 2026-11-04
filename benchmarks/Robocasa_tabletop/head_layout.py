"""Static GR1 head contract; targets are NOT raw MuJoCo motor controls.

No simulator construction. Mechanical limits do not establish a collision-safe
search domain. A separately admitted joint controller must track these targets.
"""

from dataclasses import dataclass
import copy
import math
import xml.etree.ElementTree as ET

HEAD_NAMES = ("head_yaw", "head_roll", "head_pitch")


def restore_head_actuation(task_xml, reference_xml, *, naming_prefix):
    """Restore only the three removed head hinges and their native actuators.

    Both XML inputs must already use the same explicit naming prefix. This is
    a pure transformation: it does not construct a simulator or choose gains,
    a search range, or a controller. Existing task joints/actuators retain their
    relative order. Joint state must still be remapped by name after compilation.
    Unsupported coupled head tendons/equalities fail instead of being guessed.
    """
    if not isinstance(naming_prefix, str):
        raise ValueError("Explicit common naming prefix required")
    task, reference = ET.fromstring(task_xml), ET.fromstring(reference_xml)
    for tree in (task, reference):
        compiler = tree.find("compiler")
        if compiler is None or compiler.get("angle") != "radian":
            raise ValueError("Both sources must explicitly use radians")
        if tree.find("worldbody") is None or tree.find("actuator") is None:
            raise ValueError("Worldbody and actuator sections required")
    names = tuple(naming_prefix + n for n in HEAD_NAMES)
    for tree in (task, reference):
        for section in ("tendon", "equality"):
            node = tree.find(section)
            if node is not None and any(
                value in names for item in node.iter() for value in item.attrib.values()
            ):
                raise ValueError("Coupled head topology requires a separate adapter")
    added = []
    for name in names:
        if any(j.get("name") == name for j in task.find("worldbody").iter("joint")):
            raise ValueError("Head is already articulated; do not apply twice")
        matches = [
            (parent, child)
            for parent in reference.find("worldbody").iter("body")
            for child in parent
            if child.tag == "joint" and child.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError("Unique reference head joint required")
        parent, joint = matches[0]
        bodies = [
            b for b in task.find("worldbody").iter("body") if b.get("name") == parent.get("name")
        ]
        if len(bodies) != 1 or not parent.get("name"):
            raise ValueError("Head body identity differs")
        for key in ("pos", "quat", "axisangle", "euler", "xyaxes", "zaxis"):
            if bodies[0].get(key) != parent.get(key):
                raise ValueError("Head mount transform differs from reference")
        motors = [a for a in reference.find("actuator") if a.get("joint") == name]
        if len(motors) != 1 or motors[0].tag != "motor" or not motors[0].get("name"):
            raise ValueError("Exactly one named native motor per head joint required")
        motor = motors[0]
        if any(
            a.get("joint") == name or a.get("name") == motor.get("name")
            for a in task.find("actuator")
        ):
            raise ValueError("Conflicting existing head actuator")
        bodies[0].insert(0, copy.deepcopy(joint))
        task.find("actuator").append(copy.deepcopy(motor))
        for sensor in reference.findall("./sensor/*"):
            if sensor.get("joint") != name:
                continue
            if any(s.get("name") == sensor.get("name") for s in task.findall("./sensor/*")):
                raise ValueError("Conflicting existing head sensor")
            node = task.find("sensor")
            if node is None:
                node = ET.SubElement(task, "sensor")
            node.append(copy.deepcopy(sensor))
        added.append({"joint": name, "body": parent.get("name"), "actuator": motor.get("name")})
    return ET.tostring(task, encoding="unicode"), tuple(added)


@dataclass(frozen=True)
class HeadLayout:
    limits: tuple
    axes: tuple
    actuator_names: tuple
    actuator_kind: str

    def named_targets(self, yaw_pitch, *, roll_hold):
        if len(yaw_pitch) != 2:
            raise ValueError("Exactly yaw/pitch radians required")
        values = (float(yaw_pitch[0]), float(roll_hold), float(yaw_pitch[1]))
        if any(
            not math.isfinite(v) or not low <= v <= high
            for v, (low, high) in zip(values, self.limits)
        ):
            raise ValueError("Head target outside mechanical limits; no clipping")
        return dict(zip(HEAD_NAMES, values))


def read_head_layout(path):
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise ValueError("Explicit radian source required")
    joints = list(root.find("worldbody").iter("joint"))
    actuators = list(root.find("actuator"))
    limits = []
    axes = []
    names = []
    kinds = []
    for name in HEAD_NAMES:
        matches = [j for j in joints if j.get("name") == name]
        motors = [a for a in actuators if a.get("joint") == name]
        if len(matches) != 1 or len(motors) != 1:
            raise ValueError("Unique joint/actuator binding required")
        j, a = matches[0], motors[0]
        bound = tuple(map(float, j.attrib["range"].split()))
        axis = tuple(map(float, j.attrib["axis"].split()))
        if (
            len(bound) != 2
            or len(axis) != 3
            or not all(math.isfinite(v) for v in bound + axis)
            or bound[0] >= bound[1]
            or not math.isclose(sum(v * v for v in axis), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("Invalid joint axis or limits")
        limits.append(bound)
        axes.append(axis)
        names.append(a.attrib["name"])
        kinds.append(a.tag)
    if len(set(kinds)) != 1:
        raise ValueError("Mixed actuator modes need an explicit controller adapter")
    return HeadLayout(tuple(limits), tuple(axes), tuple(names), kinds[0])


def extend_initial_joint_state(original, target_joint_names, *, added_head_positions):
    """Map by name, never shifted numeric indices when head DOFs are restored.

    Joint-only preflight, not a replacement for full qpos/qvel/RNG replay.
    No source joint may disappear or be silently overwritten.
    """
    names = tuple(target_joint_names)
    if (
        len(set(names)) != len(names)
        or not set(original) <= set(names)
        or set(added_head_positions) != set(names) - set(original)
        or not set(added_head_positions) <= set(HEAD_NAMES)
    ):
        raise ValueError("Only explicit added head joints may differ")
    merged = {**original, **added_head_positions}
    if any(not math.isfinite(float(v)) for v in merged.values()):
        raise ValueError("Finite joint positions required")
    return tuple(float(merged[name]) for name in names)


def extend_generalized_joint_state(original, target_layout, *, added_head_positions):
    """Preserve named position AND velocity blocks when adding head hinges.

    target_layout is an explicitly inspected sequence of (name, nq, nv).
    Free-joint qpos and qvel have different widths; never reuse qpos indices
    for velocity. This maps joint state only, not actuator/RNG/cache replay.
    Added head hinges start with explicit position and zero velocity.
    """
    layout = tuple(target_layout)
    names = tuple(row[0] for row in layout)
    if (
        len(set(names)) != len(names)
        or not set(original) <= set(names)
        or set(added_head_positions) != set(names) - set(original)
        or not set(added_head_positions) <= set(HEAD_NAMES)
    ):
        raise ValueError("Only explicitly added head joints may differ")
    positions, velocities = [], []
    for name, nq, nv in layout:
        if (
            not isinstance(name, str)
            or not name
            or type(nq) is not int
            or type(nv) is not int
            or nq < 1
            or nv < 1
        ):
            raise ValueError("Explicit positive position/velocity widths required")
        if name in original:
            block = original[name]
            if set(block) != {"qpos", "qvel"}:
                raise ValueError("Both named position and velocity blocks required")
            qpos, qvel = tuple(block["qpos"]), tuple(block["qvel"])
        else:
            if (nq, nv) != (1, 1):
                raise ValueError("New head joints must be scalar hinges")
            qpos, qvel = (added_head_positions[name],), (0.0,)
        if len(qpos) != nq or len(qvel) != nv:
            raise ValueError("Joint topology changed; cannot reuse initial state")
        qpos, qvel = tuple(map(float, qpos)), tuple(map(float, qvel))
        if not all(math.isfinite(v) for v in qpos + qvel):
            raise ValueError("Finite position and velocity required")
        positions.extend(qpos)
        velocities.extend(qvel)
    return {"qpos": tuple(positions), "qvel": tuple(velocities)}
