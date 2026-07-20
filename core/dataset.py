"""DACEDataset -- forget / retain prompt sets. Independent of lsse (no import).

Plain-text prompt files, one prompt per line, '#' comments ignored.

Callers:
  - train_dace.py
  - tests/test_dace.py
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List


def _load(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    lines = p.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


class DACEDataset:
    """Forget (explicit concept) + retain prompts for DACE training."""

    def __init__(
        self,
        forget_prompts: List[str],
        retain_prompts: List[str],
        target_concept: str = "nudity",
        seed: int = 42,
    ):
        if not forget_prompts:
            raise ValueError("forget_prompts is empty")
        if not retain_prompts:
            raise ValueError("retain_prompts is empty")
        self.forget_prompts = forget_prompts
        self.retain_prompts = retain_prompts
        self.target_concept = target_concept
        self.seed = seed
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.forget_prompts)

    @classmethod
    def from_files(
        cls,
        forget_file: str,
        retain_file: str,
        target_concept: str = "nudity",
        seed: int = 42,
    ) -> "DACEDataset":
        return cls(
            forget_prompts=_load(forget_file),
            retain_prompts=_load(retain_file),
            target_concept=target_concept,
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            f"DACEDataset(concept='{self.target_concept}', "
            f"forget={len(self.forget_prompts)}, retain={len(self.retain_prompts)})"
        )


import re as _re

_CONCEPT_KW = [r"\bnude\b", r"\bnaked\b", r"\bnudity\b", r"\btopless\b",
               r"\bbare-skinned\b", r"\bbare\b", r"\bexposed\b", r"\bundressing\b",
               r"\bwithout clothes\b", r"\bskin-baring\b", r"\bexplicit\b", r"\bsensual\b"]
_CONCEPT_RE = _re.compile("|".join(_CONCEPT_KW), flags=_re.IGNORECASE)


def neutralize(prompt: str) -> str:
    """Concept-stripped counterpart of a forget prompt (same content, concept word removed)."""
    s = _CONCEPT_RE.sub(" ", prompt)
    s = _re.sub(r"\s+", " ", s).strip()
    return s if s else "a photo"
