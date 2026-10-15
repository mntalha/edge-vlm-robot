"""A tiny word-level tokenizer for the robot domain.

Real VLMs use sub-word BPE tokenizers with ~50k-150k entries (the embedding
table alone can be a big chunk of a small model!). For learning we use a
closed vocabulary of ~70 words so the whole model trains in minutes on a CPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .domain import domain_words, words

PAD, BOS, EOS, SEP, UNK, IMAGE = "<pad>", "<bos>", "<eos>", "<sep>", "<unk>", "<image>"
SPECIALS = [PAD, BOS, EOS, SEP, UNK, IMAGE]
IGNORE_INDEX = -100


class RobotTokenizer:
    def __init__(self, vocab: List[str]):
        self.itos = list(vocab)
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        assert all(s in self.stoi for s in SPECIALS), "special tokens missing"

    @classmethod
    def from_domain(cls) -> "RobotTokenizer":
        return cls(SPECIALS + [w for w in domain_words() if w not in SPECIALS])

    # ids of special tokens
    pad_id = property(lambda self: self.stoi[PAD])
    bos_id = property(lambda self: self.stoi[BOS])
    eos_id = property(lambda self: self.stoi[EOS])
    sep_id = property(lambda self: self.stoi[SEP])
    unk_id = property(lambda self: self.stoi[UNK])
    image_id = property(lambda self: self.stoi[IMAGE])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.unk_id) for w in words(text)]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            w = self.itos[int(i)]
            if skip_special and w in SPECIALS:
                continue
            out.append(w)
        return " ".join(out)

    def encode_prompt(self, prompt: str, num_image_tokens: int) -> List[int]:
        """``<bos> <image>*N prompt <sep>`` - the model continues after <sep>."""
        return [self.bos_id] + [self.image_id] * num_image_tokens + self.encode(prompt) + [self.sep_id]

    def encode_example(self, prompt: str, action: str, num_image_tokens: int) -> Tuple[List[int], List[int]]:
        """Return (input_ids, labels). Loss is only computed on the action."""
        prefix = self.encode_prompt(prompt, num_image_tokens)
        target = self.encode(action) + [self.eos_id]
        return prefix + target, [IGNORE_INDEX] * len(prefix) + target

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RobotTokenizer":
        return cls(json.loads(Path(path).read_text()))
