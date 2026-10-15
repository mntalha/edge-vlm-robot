import torch

from edge_vlm.hardware import PROFILES, get_profile
from edge_vlm.memory import KNOWN_MODELS, kv_cache_bytes, plan_budget, vision_token_count
from edge_vlm.optimize import WeightOnlyQuantLinear, model_storage_mb, quantization_error, quantize_weights


def test_kv_cache_formula():
    # 2 (K,V) * layers * kv_heads * head_dim * tokens * 2 bytes
    assert kv_cache_bytes(30, 3, 64, 1000) == 2 * 30 * 3 * 64 * 1000 * 2


def test_vision_tokens():
    assert vision_token_count(512, 16, pixel_shuffle=4) == 64
    assert vision_token_count(64, 8, pixel_shuffle=2) == 16


def test_budgets_make_sense():
    nano = get_profile("orin_nano_8gb")
    assert plan_budget(KNOWN_MODELS["SmolVLM-256M"], nano, "fp16").fits
    assert not plan_budget(KNOWN_MODELS["Qwen2.5-VL-7B"], nano, "fp16").fits
    b16 = plan_budget(KNOWN_MODELS["SmolVLM-500M"], nano, "fp16")
    b4 = plan_budget(KNOWN_MODELS["SmolVLM-500M"], nano, "int4")
    assert b4.weights_gb < b16.weights_gb and b4.max_decode_tok_s > 3 * b16.max_decode_tok_s
    assert set(PROFILES) >= {"orin_nano_8gb", "orin_nx_16gb", "agx_orin_64gb", "thor_t5000"}


def test_weight_only_quantization():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 32))
    x = torch.randn(8, 64)
    ref = net(x)
    q8 = quantize_weights(net, bits=8)
    q4 = quantize_weights(net, bits=4, group_size=32)
    assert isinstance(q8[0], WeightOnlyQuantLinear) and isinstance(net[0], torch.nn.Linear)
    err8 = (q8(x) - ref).abs().max().item()
    err4 = (q4(x) - ref).abs().max().item()
    assert err8 < err4 < 0.5
    assert model_storage_mb(q4) < model_storage_mb(q8) < model_storage_mb(net)
    assert max(quantization_error(net, q8).values()) < 0.01
