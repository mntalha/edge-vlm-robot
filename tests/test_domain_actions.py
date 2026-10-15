import pytest

from edge_vlm.actions import Action, check_preconditions, parse_action
from edge_vlm.domain import build_prompt, domain_words, parse_command
from edge_vlm.tokenizer import IGNORE_INDEX, RobotTokenizer


@pytest.mark.parametrize(
    "cmd,target,hint",
    [
        ("Bring me the red cup from the kitchen", "red cup", "kitchen"),
        ("please fetch the blue ball", "blue ball", None),
        ("go to the office and get the yellow box", "yellow box", "office"),
        ("bring me something", None, None),
    ],
)
def test_parse_command(cmd, target, hint):
    assert parse_command(cmd) == (target, hint)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("navigate kitchen", Action("navigate", "kitchen")),
        ("pick red cup", Action("pick", "red cup")),
        ("return", Action("return")),
        ("handover", Action("handover")),
        ('Sure! {"skill": "pick", "object": "Green Book"}', Action("pick", "green book")),
        ('{"skill": "navigate", "room": "bedroom"}', Action("navigate", "bedroom")),
        ("navigate garage", None),
        ("pick purple cup", None),
        ("dance", None),
        ("", None),
        ('{"skill": broken json', None),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_action_roundtrip():
    for a in [Action("navigate", "office"), Action("pick", "blue bottle"), Action("return"), Action("handover")]:
        assert parse_action(a.to_text()) == a
        assert parse_action(a.to_json()) == a


def test_preconditions():
    assert not check_preconditions(Action("pick", "red cup"), "kitchen", "blue ball")[0]
    assert not check_preconditions(Action("handover"), "kitchen", "red cup")[0]
    assert not check_preconditions(Action("handover"), "living_room", None)[0]
    assert not check_preconditions(Action("navigate", "kitchen"), "kitchen", None)[0]
    assert check_preconditions(Action("handover"), "living_room", "red cup")[0]


def test_tokenizer_covers_prompts_and_masks_labels():
    tok = RobotTokenizer.from_domain()
    prompt = build_prompt("Can you fetch the red cup in the kitchen?", "living_room", None, ["living_room"])
    assert tok.unk_id not in tok.encode(prompt)
    ids, labels = tok.encode_example(prompt, "navigate kitchen", num_image_tokens=4)
    assert len(ids) == len(labels)
    assert ids.count(tok.image_id) == 4
    n_target = len(tok.encode("navigate kitchen")) + 1
    assert all(label == IGNORE_INDEX for label in labels[:-n_target])
    assert tok.decode(labels[-n_target:]) == "navigate kitchen"
    assert len(set(domain_words())) == len(domain_words())
