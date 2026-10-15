"""Robot skill/action schema, parsing and safety validation.

The VLM never drives motors directly. It emits a *high-level skill* such as
``navigate kitchen`` or ``pick red cup``; this module parses it, checks it is
well-formed and that its preconditions hold, and only then is it handed to
the low-level controllers (Nav2 / grasp policy / sim). This "guardrail"
layer is essential on a real robot because language models can hallucinate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .domain import COLORS, OBJECT_KINDS, ROOMS, SKILLS, USER_ROOM


@dataclass(frozen=True)
class Action:
    skill: str
    target: Optional[str] = None  # room for navigate, "color kind" for pick

    def to_text(self) -> str:
        return f"{self.skill} {self.target}" if self.target else self.skill

    def to_json(self) -> str:
        payload = {"skill": self.skill}
        if self.skill == "navigate":
            payload["room"] = self.target
        elif self.skill == "pick":
            payload["object"] = self.target
        return json.dumps(payload)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.to_text()


def _valid_object(name: Optional[str]) -> bool:
    if not name:
        return False
    parts = name.split()
    return len(parts) == 2 and parts[0] in COLORS and parts[1] in OBJECT_KINDS


def _build(skill: str, target: Optional[str]) -> Optional[Action]:
    skill = skill.strip().lower()
    target = target.strip().lower() if isinstance(target, str) else None
    if skill == "navigate" and target in ROOMS:
        return Action("navigate", target)
    if skill == "pick" and _valid_object(target):
        return Action("pick", target)
    if skill in ("return", "handover"):
        return Action(skill)
    return None


def parse_action(text: str) -> Optional[Action]:
    """Parse model output into an :class:`Action`, or ``None`` if malformed.

    Accepts the compact text form used by the tiny model (``pick red cup``)
    and the JSON form that larger pretrained VLMs are usually prompted for
    (``{"skill": "pick", "object": "red cup"}``).
    """
    if not text:
        return None
    text = text.strip()
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "skill" not in obj:
            return None
        target = obj.get("room") or obj.get("object") or obj.get("target")
        return _build(str(obj["skill"]), target)
    toks = text.lower().split()
    if not toks or toks[0] not in SKILLS:
        return None
    return _build(toks[0], " ".join(toks[1:]) or None)


def check_preconditions(action: Action, room: str, holding: Optional[str]) -> Tuple[bool, str]:
    """Safety/logic checks before executing a skill on the robot."""
    if action.skill == "pick" and holding is not None:
        return False, f"already holding {holding}"
    if action.skill == "handover":
        if holding is None:
            return False, "nothing to hand over"
        if room != USER_ROOM:
            return False, "user is not here"
    if action.skill == "navigate" and action.target == room:
        return False, f"already in {room}"
    return True, "ok"
