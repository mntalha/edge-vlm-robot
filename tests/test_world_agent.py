import numpy as np

from edge_vlm.actions import Action
from edge_vlm.agent import ExpertPolicy, FetchAgent, evaluate_policy
from edge_vlm.data import action_distribution, generate_samples
from edge_vlm.sim import FetchWorld


def test_world_render_and_uniqueness():
    w = FetchWorld(seed=0)
    for _ in range(20):
        w.reset()
        img = w.render()
        assert img.shape == (64, 64, 3) and img.dtype == np.uint8
        everywhere = [o for r in w.objects for o in w.visible_objects(r)]
        assert everywhere.count(w.target) == 1
        assert w.target in w.visible_objects(w.target_room)


def test_world_is_deterministic_for_seed():
    a, b = FetchWorld(seed=3), FetchWorld(seed=3)
    assert a.command == b.command
    assert np.array_equal(a.render(), b.render())


def test_manual_episode():
    w = FetchWorld(seed=1)
    w.reset(target="red cup", target_room="office")
    assert w.step(Action("navigate", "office")).ok
    assert w.step(Action("pick", "red cup")).ok
    assert w.holding == "red cup"
    assert not w.step(Action("handover")).ok  # not at the user yet
    assert w.step(Action("return")).ok
    r = w.step(Action("handover"))
    assert r.done and r.success


def test_expert_solves_every_episode():
    stats = evaluate_policy(ExpertPolicy(), n_episodes=100, seed=5)
    assert stats["success_rate"] == 1.0
    assert stats["avg_steps"] <= 6


def test_agent_guardrail_stops_bad_output():
    class Bad:
        def act(self, state):
            return "handover"

    r = FetchAgent(Bad()).run(FetchWorld(seed=0))
    assert not r.success and r.failure.startswith("guardrail")


def test_generate_samples_labels_are_valid():
    samples = generate_samples(30, seed=0)
    dist = action_distribution(samples)
    assert dist["pick"] == dist["return"] == dist["handover"] == 30
    assert dist["navigate"] >= 30
