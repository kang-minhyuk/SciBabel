from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ArmState:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


class EpsilonGreedyBandit:
    """Simple in-memory epsilon-greedy bandit keyed by route (src->tgt)."""

    def __init__(self, actions: list[str], epsilon: float = 0.2) -> None:
        self.actions = actions
        self.epsilon = epsilon
        self._state: dict[str, dict[str, ArmState]] = defaultdict(dict)

    def choose(self, route_key: str) -> str:
        self._ensure_arms(route_key)
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        arm_map = self._state[route_key]
        return max(arm_map, key=lambda action: arm_map[action].mean_reward)

    def update(self, route_key: str, action: str, reward: float) -> None:
        self._ensure_arms(route_key)
        arm = self._state[route_key][action]
        arm.pulls += 1
        arm.total_reward += reward

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for route, arm_map in self._state.items():
            out[route] = {}
            for action, arm in arm_map.items():
                out[route][action] = {
                    "pulls": float(arm.pulls),
                    "mean_reward": arm.mean_reward,
                }
        return out

    def _ensure_arms(self, route_key: str) -> None:
        if route_key in self._state and self._state[route_key]:
            return
        for action in self.actions:
            self._state[route_key][action] = ArmState()
