"""Pure tensor target constructions for ODACE (controlled target family, plan section 6.3).

All three modes are affine constructions over frozen-teacher noise predictions evaluated at
the SAME latent/timestep; every returned target is detached, so no gradient ever flows into
the teacher. The formulas are bit-for-bit the legacy trainer expressions:

    push               : e_u - eta * (e_c - e_u)          (legacy erase_mode 'negguide')
    anchor             : e_b                              (legacy 'benign_anchor')
    anchor_contrastive : e_b - lam * (e_c - e_b)          (legacy 'benign_neg')

with e_u = uncond teacher output, e_c = concept(forget) teacher output, e_b = benign-anchor
teacher output. anchor_contrastive(lam=0) == anchor exactly. This family is NOT claimed as a
novel equation (it matches SDErasure's affine extrapolation); it exists so ablations share
one implementation.

Callers: core/trainer.py, tests/test_targets.py.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

TARGET_MODES: Tuple[str, ...] = ("push", "anchor", "anchor_contrastive")


def push_target(eps_uncond: torch.Tensor, eps_concept: torch.Tensor,
                eta: float) -> torch.Tensor:
    """ESD-style negative guidance: move past uncond, away from the concept direction."""
    return (eps_uncond - eta * (eps_concept - eps_uncond)).detach()


def anchor_target(eps_benign: torch.Tensor) -> torch.Tensor:
    """Redirect the concept-conditioned output onto the benign-anchor teacher output."""
    return eps_benign.detach()


def anchor_contrastive_target(eps_benign: torch.Tensor, eps_concept: torch.Tensor,
                              lam: float) -> torch.Tensor:
    """Anchor plus lam-scaled repulsion along the (concept - benign) output direction."""
    return (eps_benign - lam * (eps_concept - eps_benign)).detach()


def compute_target(
    mode: str,
    *,
    eps_uncond: Optional[torch.Tensor] = None,
    eps_benign: Optional[torch.Tensor] = None,
    eps_concept: Optional[torch.Tensor] = None,
    eta: Optional[float] = None,
    lam: Optional[float] = None,
    with_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Dispatch on TARGET_MODES; raises on a missing required input instead of guessing.

    Returns (target, diagnostics). diagnostics is {} unless with_diagnostics, then it holds
    float norms useful for training logs (never used in the loss).
    """
    if mode == "push":
        if eps_uncond is None or eps_concept is None or eta is None:
            raise ValueError("push target needs eps_uncond, eps_concept, eta")
        target = push_target(eps_uncond, eps_concept, eta)
        ref = eps_uncond
    elif mode == "anchor":
        if eps_benign is None:
            raise ValueError("anchor target needs eps_benign")
        target = anchor_target(eps_benign)
        ref = eps_benign
    elif mode == "anchor_contrastive":
        if eps_benign is None or eps_concept is None or lam is None:
            raise ValueError("anchor_contrastive target needs eps_benign, eps_concept, lam")
        target = anchor_contrastive_target(eps_benign, eps_concept, lam)
        ref = eps_benign
    else:
        raise ValueError(f"unknown target mode '{mode}'; expected one of {TARGET_MODES}")

    diag: Dict[str, float] = {}
    if with_diagnostics:
        with torch.no_grad():
            diag["target_norm"] = float(target.norm())
            diag["ref_norm"] = float(ref.norm())
            if eps_concept is not None:
                diag["target_minus_concept_norm"] = float((target - eps_concept).norm())
    return target, diag
