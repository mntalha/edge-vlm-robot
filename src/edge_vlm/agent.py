"""Policies and the closed-loop fetch agent.

Architecture (the same split NVIDIA uses for GR00T-style robots):

* **System 2 – slow, deliberate**: the VLM looks at the camera + command and
  picks the next *skill* (a few Hz is plenty).
* **System 1 – fast, reactive**: skill controllers (Nav2 path following,
  obstacle avoidance, grasping) run at 20-100+ Hz and are *not* VLM calls.

Here System 1 is abstracted by :class:`~edge_vlm.sim.FetchWorld.step`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np

from .actions import Action, check_preconditions, parse_action
from .domain import SEARCH_ROOMS, USER_ROOM, build_prompt, parse_command
from .sim.world import FetchWorld


@dataclass
class AgentState:
    image: np.ndarray
    command: str
    room: str
    holding: Optional[str]
    visited: List[str]
    visible: Optional[List[str]] = None  # ground truth, expert only

    @property
    def prompt(self) -> str:
        return build_prompt(self.command, self.room, self.holding, self.visited)


class Policy(Protocol):
    def act(self, state: AgentState) -> str: ...


def expert_action(state: AgentState) -> Action:
    """Privileged rule-based teacher. Uses ground-truth ``visible`` objects."""
    if state.holding is not None:
        return Action("handover") if state.room == USER_ROOM else Action("return")
    target, hint = parse_command(state.command)
    if target and state.visible and target in state.visible:
        return Action("pick", target)
    if hint and hint not in state.visited:
        return Action("navigate", hint)
    for room in SEARCH_ROOMS:
        if room not in state.visited:
            return Action("navigate", room)
    return Action("return")  # searched everywhere: go back and report


class ExpertPolicy:
    def act(self, state: AgentState) -> str:
        return expert_action(state).to_text()


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    failure: Optional[str]
    trajectory: List[Dict[str, object]] = field(default_factory=list)


class FetchAgent:
    """Closed loop: observe -> VLM picks skill -> validate -> execute."""

    def __init__(self, policy: Policy, max_steps: int = 10):
        self.policy = policy
        self.max_steps = max_steps

    def run(self, world: FetchWorld, command: Optional[str] = None) -> EpisodeResult:
        command = command or world.command
        visited: List[str] = [world.robot_room]
        traj: List[Dict[str, object]] = []
        for step in range(self.max_steps):
            obs = world.observe()
            state = AgentState(obs.image, command, obs.room, obs.holding, list(visited), world.visible_objects())
            raw = self.policy.act(state)
            action = parse_action(raw)
            rec: Dict[str, object] = {"step": step, "room": obs.room, "holding": obs.holding, "output": raw}
            traj.append(rec)
            if action is None:
                rec["result"] = "unparseable output -> safe stop"
                return EpisodeResult(False, step + 1, "unparseable", traj)
            ok, why = check_preconditions(action, obs.room, obs.holding)
            if not ok:
                rec["result"] = f"rejected by guardrail: {why} -> safe stop"
                return EpisodeResult(False, step + 1, f"guardrail: {why}", traj)
            res = world.step(action)
            rec["result"] = res.message
            if action.skill == "navigate" and action.target not in visited:
                visited.append(action.target)
            if not res.ok and action.skill == "pick":
                return EpisodeResult(False, step + 1, res.message, traj)
            if res.done:
                return EpisodeResult(res.success, step + 1, None if res.success else res.message, traj)
        return EpisodeResult(False, self.max_steps, "timeout", traj)


def evaluate_policy(policy: Policy, n_episodes: int = 100, seed: int = 123, **world_kwargs) -> Dict[str, float]:
    world = FetchWorld(seed=seed, **world_kwargs)
    agent = FetchAgent(policy)
    successes, steps = 0, []
    failures: Dict[str, int] = {}
    for _ in range(n_episodes):
        world.reset()
        r = agent.run(world)
        successes += r.success
        steps.append(r.steps)
        if r.failure:
            key = r.failure.split(":")[0]
            failures[key] = failures.get(key, 0) + 1
    return {
        "success_rate": successes / n_episodes,
        "avg_steps": float(np.mean(steps)),
        **{f"fail_{k}": v for k, v in failures.items()},
    }
