import pytest
import torch

from edge_vlm.agent import FetchAgent
from edge_vlm.data import generate_samples
from edge_vlm.sim import FetchWorld
from edge_vlm.tokenizer import RobotTokenizer
from edge_vlm.train import VLMPolicy, build_model, load_checkpoint, save_checkpoint, train


def tiny(tok):
    return build_model(tok, vision_dim=32, vision_layers=1, vision_heads=2, lm_dim=48, lm_layers=1,
                       lm_heads=4, lm_kv_heads=2, lm_ffn_dim=96)


def test_train_smoke_and_checkpoint_roundtrip(tmp_path):
    tok = RobotTokenizer.from_domain()
    model = tiny(tok)
    samples = generate_samples(10, seed=0)
    hist = train(model, tok, samples, samples[:8], epochs=2, batch_size=16, device=torch.device("cpu"), log_every=1)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    path = tmp_path / "m.pt"
    save_checkpoint(model, tok, path)
    m2, tok2 = load_checkpoint(path)
    policy, policy2 = VLMPolicy(model, tok), VLMPolicy(m2, tok2)
    s = samples[0]
    assert policy.predict(s.image, s.prompt) == policy2.predict(s.image, s.prompt)
    FetchAgent(policy, max_steps=3).run(FetchWorld(seed=0))  # runs end-to-end without crashing


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from edge_vlm.export import export_onnx, verify_onnx

    model = tiny(RobotTokenizer.from_domain())
    paths = export_onnx(model, tmp_path)
    err = verify_onnx(model, paths, seq_len=30)
    assert err["decoder_max_abs_err"] < 1e-3
