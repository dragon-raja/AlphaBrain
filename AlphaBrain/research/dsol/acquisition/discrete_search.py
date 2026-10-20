"""Queryable finite-camera graph replay; not physical motion validation.

Only query(node) may reveal a score. No outcome labels or unvisited image
features are accepted by the search API. Navigation uses relative pose alone.
"""

import heapq
import numpy as np


def camera_graph(poses, neighbors=6, translation_scale=.1, rotation_scale=15.):
    c = np.asarray(poses, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (4, 4) or not np.isfinite(c).all():
        raise ValueError("Expected finite candidate camera transforms")
    if not 1 <= neighbors < len(c) or min(translation_scale, rotation_scale) <= 0:
        raise ValueError("Invalid graph parameters")
    translation = np.linalg.norm(c[:, None, :3, 3] - c[None, :, :3, 3], axis=-1)
    cosine = (np.einsum('nij,mij->nm', c[:, :3, :3], c[:, :3, :3]) - 1) / 2
    rotation = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    distance = translation / translation_scale + rotation / rotation_scale
    np.fill_diagonal(distance, np.inf)
    edges = np.zeros(distance.shape, dtype=bool)
    for i in range(len(c)):
        edges[i, np.argsort(distance[i], kind='stable')[:neighbors]] = True
    edges |= edges.T
    seen, pending = {0}, [0]
    while pending:
        for j in np.flatnonzero(edges[pending.pop()]):
            if int(j) not in seen:
                seen.add(int(j)); pending.append(int(j))
    if len(seen) != len(c):
        raise ValueError("Disconnected fixed graph; revise protocol, not silently add edges")
    np.fill_diagonal(distance, 0)
    return edges, distance, translation, rotation


def _route(edges, distance, allowed, source, target):
    queue = [(0., source, [source])]
    seen = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == target:
            return path
        if node in seen:
            continue
        seen.add(node)
        for j in np.flatnonzero(edges[node]):
            j = int(j)
            if j in allowed and j not in seen:
                heapq.heappush(queue, (cost + distance[node, j], j, path + [j]))
    raise ValueError("Visited subgraph disconnected")


def visited_route(edges, distance, allowed, source, target):
    """Public routing interface; no new nodes or scores are acquired."""
    return _route(edges, distance, allowed, source, target)


def search(edges, distance, start, budgets, query, *, random_seed=None):
    """Best-score frontier expansion or random frontier expansion.

    Scores are minimized. Both policies select the best queried node, retaining
    the incumbent on ties. The start is one query. Travel through already
    visited nodes and final return to the selected endpoint are recorded.
    """
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets or budgets[0] < 1 or budgets[-1] > len(edges):
        raise ValueError("Invalid distinct-node query budgets")
    if not 0 <= start < len(edges):
        raise ValueError("Invalid start")
    rng = None if random_seed is None else np.random.default_rng(random_seed)
    scores, visited, path, results = {}, [], [int(start)], {}
    current = best = int(start)
    while len(visited) < budgets[-1]:
        value = float(query(current))
        if not np.isfinite(value):
            raise ValueError("Nonfinite queried score")
        scores[current] = value
        visited.append(current)
        if value < scores[best]:
            best = current
        allowed = set(visited)
        if len(visited) in budgets:
            final_path = path + _route(edges, distance, allowed, current, best)[1:]
            results[len(visited)] = dict(selected=best, visited=list(visited), path=final_path)
        if len(visited) == budgets[-1]:
            break
        frontier = [(i, int(j)) for i in visited for j in np.flatnonzero(edges[i]) if int(j) not in allowed]
        if not frontier:
            raise ValueError("No unvisited frontier")
        if rng is None:
            anchor, nxt = min(frontier, key=lambda ij: (scores[ij[0]], distance[ij], visited.index(ij[0]), ij[1]))
        else:
            anchor, nxt = frontier[int(rng.integers(len(frontier)))]
        path.extend(_route(edges, distance, allowed, current, anchor)[1:])
        path.append(nxt)
        current = nxt
    return results
