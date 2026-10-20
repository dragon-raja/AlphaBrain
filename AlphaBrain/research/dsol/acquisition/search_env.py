"""Budgeted cached-camera search, with a Gymnasium-shaped dependency-free core.

Actions are STOP (0) or acquire candidate (index + 1). Only the unvisited
frontier is available. Acquisition includes navigation through visited nodes.
This is a static-scene catalog replay, not a robot head or a physics simulator.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data.search_replay import QueryResult
from .discrete_search import camera_graph, visited_route


class QuerySource(Protocol):
    size: int
    feature_dim: int
    score_samples: int

    def query(self, candidate: int) -> QueryResult: ...


@dataclass(frozen=True)
class SearchConfig:
    query_budget: int = 8
    neighbors: int = 6
    translation_scale_m: float = .1
    rotation_scale_deg: float = 15.
    query_cost: float = .01
    motion_cost: float = .001

    def validate(self, count):
        if (not isinstance(self.query_budget, int) or not 1 <= self.query_budget <= count
                or not isinstance(self.neighbors, int) or not 1 <= self.neighbors < count
                or not all(np.isfinite(v) for v in (
                    self.translation_scale_m, self.rotation_scale_deg, self.query_cost, self.motion_cost))
                or min(self.translation_scale_m, self.rotation_scale_deg) <= 0
                or min(self.query_cost, self.motion_cost) < 0):
            raise ValueError("Invalid finite search configuration")


def unit_vector(value):
    value = np.asarray(value, dtype=float)
    return value / max(float(np.linalg.norm(value)), 1e-12)


class SearchEnv:
    """No success labels, task IDs, canonical images, or unqueried RGB access.

    reset(start=...) charges one initial query. Its fixed cost is reported in
    info, but excluded from rewards since the agent cannot choose to avoid it.
    Finite query budget exhaustion is terminal, not a time-limit truncation.
    """

    def __init__(self, poses, source: QuerySource, wrist, score_scale, config=SearchConfig()):
        self.poses = np.array(poses, dtype=float, copy=True)
        config.validate(source.size)
        if self.poses.shape != (source.size, 4, 4) or not np.isfinite(self.poses).all():
            raise ValueError("Candidate pose count/shape mismatch")
        rot = self.poses[:, :3, :3]
        if (not np.allclose(self.poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(np.swapaxes(rot, -1, -2) @ rot, np.eye(3), atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.det(rot), 1, atol=1e-5, rtol=0)):
            raise ValueError("Rigid camera transforms required")
        if (not np.isfinite(score_scale) or score_scale <= 0 or source.feature_dim < 1
                or source.score_samples < 1):
            raise ValueError("Positive development score scale and source dimensions required")
        wrist = np.asarray(wrist, dtype=float)
        if wrist.ndim != 1 or not np.isfinite(wrist).all():
            raise ValueError("Finite wrist feature vector required")
        self.source, self.config, self.scale = source, config, float(score_scale)
        self.wrist = unit_vector(wrist)
        self.edges, self.distance, self.translation, self.rotation = camera_graph(
            self.poses, config.neighbors, config.translation_scale_m, config.rotation_scale_deg)
        self.n = source.size
        self._started = False

    def reset(self, *, start, seed=None):
        if not isinstance(start, (int, np.integer)) or not 0 <= start < self.n:
            raise ValueError("Invalid starting camera")
        # The replay dynamics are deterministic; baseline sampling owns its RNG.
        self.start = self.current = self.best = int(start)
        self.visited, self.path = [], [int(start)]
        self.values = np.zeros((self.n, 2))
        self.visual = np.zeros((self.n, self.source.feature_dim))
        self.travel = self.translation_m = self.rotation_deg = self.total_reward = 0.
        self._acquire(int(start))
        self.initial_score = self.values[start, 0]
        self.done = self.config.query_budget == 1
        self.reason = "budget" if self.done else None
        self._started = True
        return self.observation(), self.info()

    def _acquire(self, node):
        value = self.source.query(node)
        visual = np.asarray(value.visual, dtype=float)
        if (visual.shape != (self.source.feature_dim,) or not np.isfinite(visual).all()
                or not np.isfinite([value.accel_mean, value.accel_std]).all()
                or min(value.accel_mean, value.accel_std) < 0
                or value.score_samples != self.source.score_samples):
            raise ValueError("Invalid query response; stop this episode")
        self.values[node] = [value.accel_mean, value.accel_std]
        self.visual[node] = unit_vector(visual)
        self.visited.append(node)
        if value.accel_mean < self.values[self.best, 0]:
            self.best = node

    def action_masks(self):
        if not self._started:
            raise RuntimeError("reset() required")
        mask = np.zeros(self.n + 1, dtype=bool)
        if not self.done:
            mask[0] = True
            mask[1:] = self.edges[self.visited].any(0)
            mask[1 + np.array(self.visited)] = False
        return mask

    def _proposal_route(self, target):
        # Shortest known route to a frontier endpoint; never crosses unseen nodes.
        allowed = set(self.visited)
        routes = []
        for anchor in self.visited:
            if self.edges[anchor, target]:
                route = visited_route(self.edges, self.distance, allowed, self.current, anchor) + [target]
                cost = sum(self.distance[a, b] for a, b in zip(route[:-1], route[1:]))
                routes.append((cost, route))
        if not routes:
            raise ValueError("Target is not on visited frontier")
        return min(routes, key=lambda x: (x[0], x[1]))[1]

    def _move(self, route):
        for a, b in zip(route[:-1], route[1:]):
            self.travel += self.distance[a, b]
            self.translation_m += self.translation[a, b]
            self.rotation_deg += self.rotation[a, b]
        self.path.extend(route[1:])
        self.current = route[-1]

    def _finish(self, reason):
        self._move(visited_route(self.edges, self.distance, set(self.visited), self.current, self.best))
        self.done, self.reason = True, reason

    def step(self, action):
        if not self._started or self.done:
            raise RuntimeError("Episode is not active")
        if (not isinstance(action, (int, np.integer)) or not 0 <= action <= self.n
                or not self.action_masks()[action]):
            raise ValueError("Action must be STOP or an unvisited frontier candidate")
        previous_best, previous_travel = self.values[self.best, 0], self.travel
        if action == 0:
            self._finish("stop")
            query_charge = 0.
        else:
            target = int(action) - 1
            route = self._proposal_route(target)
            # Query failure propagates; no silent retry or fabricated transition.
            try:
                self._acquire(target)
            except Exception:
                self.done, self.reason = True, "query_error"
                raise
            self._move(route)
            query_charge = self.config.query_cost
            if len(self.visited) == self.config.query_budget:
                self._finish("budget")
        improvement = (previous_best - self.values[self.best, 0]) / self.scale
        movement_charge = self.config.motion_cost * (self.travel - previous_travel)
        reward = improvement - query_charge - movement_charge
        self.total_reward += reward
        info = self.info()
        info["reward_terms"] = dict(improvement=improvement, query_cost=query_charge,
                                    motion_cost=movement_charge)
        return self.observation(), float(reward), self.done, False, info

    def observation(self):
        if not self._started:
            raise RuntimeError("reset() required")
        # Relative to the starting camera, not canonical/world pose coordinates.
        origin = self.poses[self.start]
        geometry = np.zeros((self.n, 12))
        geometry[:, :3] = (self.poses[:, :3, 3] - origin[:3, 3]) @ origin[:3, :3]
        geometry[:, :3] /= self.config.translation_scale_m
        geometry[:, 3:] = (origin[:3, :3].T @ self.poses[:, :3, :3]).reshape(self.n, 9)
        flags = np.zeros((self.n, 4))
        for order, node in enumerate(self.visited):
            flags[node, :2] = [1, (order + 1) / self.config.query_budget]
        flags[self.current, 2] = flags[self.best, 3] = 1
        return {
            "geometry": geometry.astype(np.float32),
            "visual": self.visual.astype(np.float32),
            "scores": (self.values / self.scale).astype(np.float32),
            "flags": flags.astype(np.float32),
            "wrist": self.wrist.astype(np.float32),
            "global": np.array([
                len(self.visited) / self.config.query_budget,
                (self.config.query_budget - len(self.visited)) / self.config.query_budget,
                self.initial_score / self.scale, self.values[self.best, 0] / self.scale,
                self.travel,
            ], dtype=np.float32),
            "action_mask": self.action_masks().astype(np.int8),
        }

    def info(self):
        return dict(
            queries=len(self.visited), scoring_members=len(self.visited) * self.source.score_samples,
            visited=list(self.visited), path=list(self.path), selected=self.best,
            translation_m=float(self.translation_m), rotation_deg=float(self.rotation_deg),
            motion_units=float(self.travel), initial_query_cost=self.config.query_cost,
            total_reward=float(self.total_reward), stop_reason=self.reason,
            initial_accel=float(self.initial_score), best_accel=float(self.values[self.best, 0]),
        )


def baseline_action(env, method, rng=None):
    """All policies share action permissions, query/return costs, and final scorer."""
    if method == "stay":
        return 0
    frontier = np.flatnonzero(env.action_masks()[1:])
    if not len(frontier):
        return 0
    if method == "random":
        if rng is None:
            raise ValueError("Explicit random generator required")
        # Match historical uniform sampling of frontier edges (not unique nodes).
        edges = [(i, int(j)) for i in env.visited for j in np.flatnonzero(env.edges[i])
                 if j not in env.visited]
        return edges[int(rng.integers(len(edges)))][1] + 1
    if method == "scan":
        # Deterministic pose order; no candidate name or unqueried score.
        geometry = env.observation()["geometry"]
        return int(min(frontier, key=lambda j: tuple(geometry[j]))) + 1
    if method == "local_accel":
        edges = [(i, int(j)) for i in env.visited for j in np.flatnonzero(env.edges[i])
                 if j not in env.visited]
        return min(edges, key=lambda ij: (
            env.values[ij[0], 0], env.distance[ij], env.visited.index(ij[0]), ij[1]))[1] + 1
    raise ValueError("Unknown baseline: " + method)


def rollout_baseline(env, start, method, seed=0):
    env.reset(start=start)
    rng = np.random.default_rng(seed)
    while not env.done:
        env.step(baseline_action(env, method, rng))
    return env.info()


def permute_observation(observation, order):
    """Episode-local action slots conceal stable catalog/canonical indices."""
    order = np.asarray(order)
    count = len(observation["geometry"])
    if (order.shape != (count,) or order.dtype.kind not in "iu"
            or not np.array_equal(np.sort(order), np.arange(count))):
        raise ValueError("A complete candidate permutation is required")
    result = {key: value.copy() for key, value in observation.items()}
    for key in ("geometry", "visual", "scores", "flags"):
        result[key] = result[key][order]
    result["action_mask"][1:] = observation["action_mask"][1:][order]
    return result


def catalog_action(action, order):
    """Convert one episode-local action slot back to the logged catalog index."""
    if not isinstance(action, (int, np.integer)) or not 0 <= action <= len(order):
        raise ValueError("Invalid action slot")
    return 0 if action == 0 else int(order[action - 1]) + 1


def gymnasium_env(core, starts):
    """Optional bridge; never install Gymnasium or import it during core imports.

    The trainer must apply action_masks() to logits (e.g. MaskablePPO).
    An ordinary unmasked PPO is not an accepted runner for this action contract.
    """
    import gymnasium as gym

    starts = tuple(starts)
    if not starts or any(not isinstance(s, (int, np.integer)) or not 0 <= s < core.n for s in starts):
        raise ValueError("Explicit valid start distribution required")
    if core.config.query_budget < 2:
        raise ValueError("Gym training episodes require at least two query slots")
    shapes = {
        "geometry": (core.n, 12), "visual": (core.n, core.source.feature_dim),
        "scores": (core.n, 2), "flags": (core.n, 4), "wrist": core.wrist.shape,
        "global": (5,), "action_mask": (core.n + 1,),
    }

    class Adapter(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.order = None
            self.action_space = gym.spaces.Discrete(core.n + 1)
            self.observation_space = gym.spaces.Dict({
                key: gym.spaces.MultiBinary(core.n + 1) if key == "action_mask"
                else gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                for key, shape in shapes.items()
            })

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            start = (options or {}).get("start")
            if start is None:
                start = starts[int(self.np_random.integers(len(starts)))]
            if start not in starts:
                raise ValueError("Start outside configured distribution")
            self.order = self.np_random.permutation(core.n)
            observation, info = core.reset(start=start)
            return permute_observation(observation, self.order), info

        def step(self, action):
            if self.order is None:
                raise RuntimeError("reset() required")
            observation, reward, terminated, truncated, info = core.step(catalog_action(action, self.order))
            return permute_observation(observation, self.order), reward, terminated, truncated, info

        def action_masks(self):
            if self.order is None:
                raise RuntimeError("reset() required")
            mask = core.action_masks()
            mask[1:] = mask[1:][self.order]
            return mask

    return Adapter()
