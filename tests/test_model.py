import math

import torch

from edge_vlm.model import TinyVLM, TinyVLMConfig, apply_rope, pixel_shuffle, rope_tables
from edge_vlm.tokenizer import RobotTokenizer


def small_model(**kw) -> TinyVLM:
    tok = RobotTokenizer.from_domain()
    cfg = TinyVLMConfig(vocab_size=len(tok), image_token_id=tok.image_id, pad_token_id=tok.pad_id,
                        vision_dim=32, vision_layers=1, vision_heads=2, lm_dim=48, lm_layers=2,
                        lm_heads=4, lm_kv_heads=2, lm_ffn_dim=96, **kw)
    torch.manual_seed(0)
    return TinyVLM(cfg).eval()


def test_pixel_shuffle_groups_2x2_neighbourhoods():
    g = 4
    coords = torch.tensor([[r, c] for r in range(g) for c in range(g)], dtype=torch.float).unsqueeze(0)
    out = pixel_shuffle(coords, 2)
    assert out.shape == (1, 4, 8)
    assert out[0, 0].tolist() == [0, 0, 0, 1, 1, 0, 1, 1]
    assert out[0, 1].tolist() == [0, 2, 0, 3, 1, 2, 1, 3]
    assert out[0, 2].tolist() == [2, 0, 2, 1, 3, 0, 3, 1]


def test_rope_preserves_norm():
    cos, sin = rope_tables(16, 8, 10000.0)
    x = torch.randn(1, 2, 16, 8)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_forward_shapes_and_initial_loss():
    m = small_model()
    n = m.cfg.num_image_tokens
    assert n == 16
    ids = torch.randint(6, m.cfg.vocab_size, (3, n + 12))
    ids[:, 1 : 1 + n] = m.cfg.image_token_id
    px = torch.randn(3, 3, 64, 64)
    logits, loss = m(ids, px, labels=ids)
    assert logits.shape == (3, n + 12, m.cfg.vocab_size)
    assert abs(loss.item() - math.log(m.cfg.vocab_size)) < 1.0


def test_merge_embeddings_inserts_visual_tokens_in_order():
    m = small_model()
    n = m.cfg.num_image_tokens
    ids = torch.full((1, n + 3), 7, dtype=torch.long)
    ids[0, 1 : 1 + n] = m.cfg.image_token_id
    img = torch.randn(1, n, m.cfg.lm_dim)
    h = m.merge_embeddings(ids, img)
    assert torch.allclose(h[0, 1 : 1 + n], img[0])
    assert torch.allclose(h[0, 0], m.embed.weight[7])


def test_kv_cache_matches_full_recompute():
    m = small_model()
    tok = RobotTokenizer.from_domain()
    ids = torch.tensor([tok.encode_prompt("cmd: bring me the red cup | at: kitchen", m.cfg.num_image_tokens)])
    px = torch.randn(1, 3, 64, 64)
    a = m.generate(ids, px, max_new_tokens=6, use_cache=True)
    b = m.generate(ids, px, max_new_tokens=6, use_cache=False)
    assert a.tolist() == b.tolist() and len(a) == 6


def test_gqa_cache_is_smaller_than_mha():
    m = small_model()
    ids = torch.randint(6, m.cfg.vocab_size, (1, 10))
    _, past = m.lm_forward(m.embed(ids))
    k, _ = past[0]
    assert k.shape == (1, m.cfg.lm_kv_heads, 10, m.cfg.lm_dim // m.cfg.lm_heads)


def test_param_breakdown_sums():
    b = small_model().param_breakdown()
    assert sum(v for k, v in b.items() if k != "total") == b["total"]
