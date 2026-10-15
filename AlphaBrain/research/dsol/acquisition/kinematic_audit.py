"""Pure geometry checks; flags are not a physical installation certificate."""
import xml.etree.ElementTree as ET
import numpy as np


def camera_ancestry(xml, camera_name):
    root = ET.fromstring(xml)
    parent = {child: node for node in root.iter() for child in node}
    cameras = [c for c in root.find('worldbody').iter('camera') if c.get('name') == camera_name]
    if len(cameras) != 1:
        raise ValueError('Expected exactly one named camera')
    actuators = {}
    actuator_node = root.find('actuator')
    for item in ([] if actuator_node is None else actuator_node):
        actuators.setdefault(item.get('joint'), []).append(dict(item.attrib))
    chain = []
    node = parent[cameras[0]]
    while node.tag == 'body':
        joints = [dict(j.attrib, actuators=actuators.get(j.get('name'), [])) for j in node.findall('joint')]
        chain.append(dict(body=node.get('name'), joints=joints))
        node = parent[node]
    return dict(camera=dict(cameras[0].attrib), chain=chain)


def inside_primitive(points, kind, size, margin=1e-5):
    """Points in geom-local frame; strict interior with a numerical margin.

    The test has zero sensor size and no motion path. None means unsupported,
    not collision-free. Planes and meshes are intentionally handled elsewhere.
    """
    p, s = np.asarray(points, float), np.asarray(size, float)
    if kind == 'box': return np.all(np.abs(p) < s-margin, axis=-1)
    if kind == 'sphere': return np.linalg.norm(p, axis=-1) < s[0]-margin
    if kind == 'ellipsoid': return np.linalg.norm(p/(s-margin), axis=-1) < 1
    if kind == 'cylinder': return (np.linalg.norm(p[:, :2], axis=-1) < s[0]-margin) & (np.abs(p[:, 2]) < s[1]-margin)
    if kind == 'capsule':
        q=p.copy(); q[:, 2] -= np.clip(q[:, 2], -s[1], s[1])
        return np.linalg.norm(q, axis=-1) < s[0]-margin
    return None


def bounded_joint_path(start, target, limits, max_step):
    """Kinematic waypoint proposal only; not a motor or collision controller."""
    a,b,limits = map(lambda x: np.asarray(x,float), (start,target,limits))
    if a.ndim != 1 or b.shape != a.shape or limits.shape != (len(a),2):
        raise ValueError('Incompatible joint dimensions')
    if not np.isfinite(max_step) or max_step <= 0 or not all(np.isfinite(x).all() for x in (a,b,limits)):
        raise ValueError('Invalid joint parameters')
    if np.any(limits[:,0] >= limits[:,1]) or any(np.any((x<limits[:,0]) | (x>limits[:,1])) for x in (a,b)):
        raise ValueError('Joint limit violation')
    count=max(1,int(np.ceil(np.max(np.abs(b-a))/max_step)))
    return np.linspace(a,b,count+1)
