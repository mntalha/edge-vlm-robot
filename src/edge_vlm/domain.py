"""The robot's world vocabulary: rooms, objects, colors and command templates.

Keeping the domain in one place lets the simulator, the expert policy, the
tokenizer and the action validator all agree on the same names.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

USER_ROOM = "living_room"
SEARCH_ROOMS: Tuple[str, ...] = ("kitchen", "bedroom", "office")
ROOMS: Tuple[str, ...] = (USER_ROOM,) + SEARCH_ROOMS

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 80, 225),
    "yellow": (235, 210, 30),
}
OBJECT_KINDS: Tuple[str, ...] = ("cup", "ball", "book", "bottle", "box")

COMMAND_TEMPLATES_WITH_ROOM: Tuple[str, ...] = (
    "bring me the {obj} from the {room}",
    "go to the {room} and get the {obj}",
    "can you fetch the {obj} in the {room}",
    "please grab the {obj} from the {room} and bring it here",
)
COMMAND_TEMPLATES_NO_ROOM: Tuple[str, ...] = (
    "bring me the {obj}",
    "find the {obj} and bring it to me",
    "i need the {obj}",
    "please fetch the {obj}",
)

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def object_name(color: str, kind: str) -> str:
    return f"{color} {kind}"


def all_object_names() -> List[str]:
    return [object_name(c, k) for c in COLORS for k in OBJECT_KINDS]


def parse_command(command: str) -> Tuple[Optional[str], Optional[str]]:
    """Rule-based command understanding (used by the *expert*, not the VLM).

    Returns ``(target_object_name, room_hint)``; either may be ``None``.
    """
    toks = words(command)
    target = None
    for i in range(len(toks) - 1):
        if toks[i] in COLORS and toks[i + 1] in OBJECT_KINDS:
            target = object_name(toks[i], toks[i + 1])
            break
    hint = next((t for t in toks if t in SEARCH_ROOMS), None)
    return target, hint


def build_prompt(command: str, room: str, holding: Optional[str], visited: Sequence[str]) -> str:
    """Text part of the VLM input. The camera image is provided separately.

    The robot's *state* (where it is, what it holds, where it already looked)
    comes from localization / gripper sensors / memory, so the VLM only has to
    do what VLMs are good at: ground language in pixels and pick the next skill.
    """
    seen = [r for r in SEARCH_ROOMS if r in visited]
    return (
        f"cmd: {command.lower()} | at: {room} | holding: {holding or 'nothing'} "
        f"| visited: {', '.join(seen) if seen else 'none'}"
    )


SKILLS: Tuple[str, ...] = ("navigate", "pick", "return", "handover")
PROMPT_KEYWORDS: Tuple[str, ...] = ("cmd", "at", "holding", "nothing", "visited", "none")


def domain_words() -> List[str]:
    """Every word the tiny tokenizer should know about."""
    vocab: List[str] = []
    texts = list(COMMAND_TEMPLATES_WITH_ROOM) + list(COMMAND_TEMPLATES_NO_ROOM)
    for t in texts:
        vocab += words(t.replace("{obj}", "").replace("{room}", ""))
    vocab += list(ROOMS) + list(COLORS) + list(OBJECT_KINDS) + list(SKILLS) + list(PROMPT_KEYWORDS)
    vocab += [":", "|", ",", "{", "}", '"', "?", ".", "!"]
    seen = set()
    return [w for w in vocab if not (w in seen or seen.add(w))]
