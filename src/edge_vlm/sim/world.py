"""A tiny 'fetch' world: rooms, objects and an egocentric camera.

It is deliberately simple (2D top-down camera, 64x64 RGB) so you can generate
thousands of training images on a laptop CPU, yet it keeps the *structure* of
the real task: the robot must understand a spoken command, look at camera
images to find the right object among distractors, pick it, and bring it back.
Swap this for Isaac Sim / a real camera later without changing the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..actions import Action, check_preconditions
from ..domain import (
    COLORS,
    COMMAND_TEMPLATES_NO_ROOM,
    COMMAND_TEMPLATES_WITH_ROOM,
    OBJECT_KINDS,
    ROOMS,
    SEARCH_ROOMS,
    USER_ROOM,
    object_name,
)

# Floor appearance per room: (base color, alternate color, pattern)
FLOORS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], str]] = {
    "living_room": ((150, 110, 80), (135, 98, 70), "stripes_h"),
    "kitchen": ((205, 205, 205), (170, 170, 170), "checker"),
    "bedroom": ((200, 175, 150), (185, 160, 138), "stripes_v"),
    "office": ((95, 95, 108), (85, 85, 98), "plain"),
}


@dataclass
class WorldObject:
    color: str
    kind: str
    center: Tuple[int, int]

    @property
    def name(self) -> str:
        return object_name(self.color, self.kind)


@dataclass
class Observation:
    image: np.ndarray  # HxWx3 uint8 from the robot camera
    room: str  # from localization (e.g. cuVSLAM / Nav2 AMCL on a real robot)
    holding: Optional[str]  # from gripper sensing


@dataclass
class StepResult:
    ok: bool
    message: str
    done: bool = False
    success: bool = False


@dataclass
class FetchWorld:
    image_size: int = 64
    noise_std: float = 6.0
    min_distractors: int = 2
    max_distractors: int = 4
    p_mention_room: float = 0.6
    p_wrong_hint: float = 0.1
    seed: Optional[int] = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, target: Optional[str] = None, target_room: Optional[str] = None) -> Observation:
        rng = self.rng
        if target is None:
            target = object_name(rng.choice(list(COLORS)), rng.choice(OBJECT_KINDS))
        if target_room is None:
            target_room = str(rng.choice(SEARCH_ROOMS))
        self.target, self.target_room = target, target_room
        others = [object_name(c, k) for c in COLORS for k in OBJECT_KINDS if object_name(c, k) != target]
        tc, tk = target.split()
        hard = [o for o in others if o.startswith(tc + " ") or o.endswith(" " + tk)]

        self.objects: Dict[str, List[WorldObject]] = {}
        for room in ROOMS:
            n = int(rng.integers(self.min_distractors, self.max_distractors + 1))
            names: List[str] = [target] if room == target_room else []
            while len(names) < n + (room == target_room):
                pool = hard if rng.random() < 0.5 else others
                cand = str(rng.choice(pool))
                if cand not in names:
                    names.append(cand)
            slots = rng.choice(9, size=len(names), replace=False)
            cell = self.image_size // 3
            objs = []
            for nm, s in zip(names, slots):
                jitter = rng.integers(-cell // 6, cell // 6 + 1, size=2)
                cx = int((s % 3) * cell + cell // 2 + jitter[0])
                cy = int((s // 3) * cell + cell // 2 + jitter[1])
                c, k = nm.split()
                objs.append(WorldObject(c, k, (cx, cy)))
            self.objects[room] = objs

        self.robot_room = USER_ROOM
        self.holding: Optional[str] = None
        self.done = False
        self.success = False
        self.n_steps = 0
        self.command = self._sample_command()
        return self.observe()

    def _sample_command(self) -> str:
        rng = self.rng
        if rng.random() < self.p_mention_room:
            room = self.target_room
            if rng.random() < self.p_wrong_hint:
                room = str(rng.choice([r for r in SEARCH_ROOMS if r != self.target_room]))
            tpl = str(rng.choice(COMMAND_TEMPLATES_WITH_ROOM))
            return tpl.format(obj=self.target, room=room)
        return str(rng.choice(COMMAND_TEMPLATES_NO_ROOM)).format(obj=self.target)

    # -------------------------------------------------------------- perception
    def visible_objects(self, room: Optional[str] = None) -> List[str]:
        """Ground-truth object list (only the *expert* may use this)."""
        return [o.name for o in self.objects[room or self.robot_room]]

    def render(self, room: Optional[str] = None, noise: bool = True) -> np.ndarray:
        room = room or self.robot_room
        s = self.image_size
        base, alt, pattern = FLOORS[room]
        img = np.empty((s, s, 3), dtype=np.float32)
        img[:] = base
        yy, xx = np.mgrid[0:s, 0:s]
        if pattern == "checker":
            mask = ((yy // 8) + (xx // 8)) % 2 == 1
        elif pattern == "stripes_h":
            mask = (yy // 6) % 2 == 1
        elif pattern == "stripes_v":
            mask = (xx // 6) % 2 == 1
        else:
            mask = np.zeros((s, s), dtype=bool)
        img[mask] = alt
        for obj in self.objects[room]:
            _draw(img, obj, s)
        if noise and self.noise_std > 0:
            img += self.rng.normal(0, self.noise_std, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def observe(self) -> Observation:
        return Observation(self.render(), self.robot_room, self.holding)

    # --------------------------------------------------------------- dynamics
    def step(self, action: Action) -> StepResult:
        """Execute a validated high-level skill. Low-level control is abstracted."""
        if self.done:
            return StepResult(False, "episode finished", True, self.success)
        self.n_steps += 1
        ok, why = check_preconditions(action, self.robot_room, self.holding)
        if not ok:
            return StepResult(False, why)
        if action.skill == "navigate":
            self.robot_room = action.target
            return StepResult(True, f"arrived at {action.target}")
        if action.skill == "return":
            self.robot_room = USER_ROOM
            return StepResult(True, "returned to user")
        if action.skill == "pick":
            for i, obj in enumerate(self.objects[self.robot_room]):
                if obj.name == action.target:
                    self.objects[self.robot_room].pop(i)
                    self.holding = obj.name
                    return StepResult(True, f"picked {obj.name}")
            return StepResult(False, f"no {action.target} here")
        if action.skill == "handover":
            item, self.holding = self.holding, None
            self.done = True
            self.success = item == self.target
            msg = "delivered target!" if self.success else f"delivered wrong item ({item})"
            return StepResult(True, msg, True, self.success)
        return StepResult(False, f"unknown skill {action.skill}")


def _draw(img: np.ndarray, obj: WorldObject, s: int) -> None:
    color = COLORS[obj.color]
    cx, cy = obj.center
    r = max(s // 12, 3)  # ~5px for a 64px image
    yy, xx = np.mgrid[0:s, 0:s]
    dx, dy = xx - cx, yy - cy
    if obj.kind == "cup":
        mask = (abs(dx) <= r - 1) & (abs(dy) <= r - 1)
    elif obj.kind == "ball":
        mask = dx**2 + dy**2 <= r**2
    elif obj.kind == "book":
        mask = (abs(dx) <= r + 2) & (abs(dy) <= r // 2)
    elif obj.kind == "bottle":
        mask = (abs(dx) <= r // 2) & (abs(dy) <= r + 2)
    elif obj.kind == "box":
        outer = (abs(dx) <= r + 1) & (abs(dy) <= r + 1)
        inner = (abs(dx) <= r - 2) & (abs(dy) <= r - 2)
        mask = outer & ~inner
    else:  # pragma: no cover
        raise ValueError(obj.kind)
    img[mask] = color
