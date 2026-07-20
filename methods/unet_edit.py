"""Legacy UNet cross-attention editing helpers (compatibility wrappers).

set_trainable_cross_attn_kv now delegates to layer_selection.select_trainable_cross_attention
(the single source of truth for attn2 detection and band definitions); the historical
behavior is unchanged: freeze everything, unfreeze attn2 K/V (include_q_out=False, the
UCE-style minimal edit) or Q/K/V/out (include_q_out=True, ESD-x-style full cross-attn).
Note the projection scope is recorded per run by the trainer manifest -- every paper-era
ODACE config trains the FULL Q/K/V/out set (xattn_full: true), not K/V alone.

eps_match_losses keeps the legacy ESD-style push objective for old tests/scripts:

    target   = eps_neutral + eta * (eps_neutral - eps_forget_frozen)
    L_forget = MSE(eps_forget, target.detach())
    L_retain = MSE(eps_retain, eps_retain_frozen.detach())

which is algebraically identical to core.targets.push_target(e_u, e_c, eta) =
e_u - eta*(e_c - e_u). New code should use core/targets.py directly.

Callers: core/legacy_trainer_snapshot.py (inlined copy), tests/test_odace.py,
tests/test_layer_selection.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .layer_selection import select_trainable_cross_attention


def set_trainable_cross_attn_kv(unet, include_q_out: bool = False) -> int:
    """Freeze all UNet params, then unfreeze cross-attention (attn2) projections.

    include_q_out=False (default): only to_k & to_v. include_q_out=True: to_q, to_k,
    to_v, to_out. Returns the trainable param count. Thin wrapper over
    layer_selection.select_trainable_cross_attention(scope='all_xattn') -- kept so legacy
    callers and the frozen regression snapshot semantics stay importable.
    """
    projections = ("q", "k", "v", "out") if include_q_out else ("k", "v")
    sel = select_trainable_cross_attention(
        unet, scope="all_xattn", projections=projections, strict=False)
    return sel.trainable_parameter_count


def eps_match_losses(eps_forget, eps_neutral_frozen, eps_forget_frozen,
                     eps_retain, eps_retain_frozen, eta: float = 1.0):
    """Legacy ESD-style negative-guidance output loss (see module docstring)."""
    en = eps_neutral_frozen.detach().float()
    ef = eps_forget_frozen.detach().float()
    target = en + eta * (en - ef)
    L_forget = F.mse_loss(eps_forget, target)
    L_retain = F.mse_loss(eps_retain, eps_retain_frozen.detach().float())
    return L_forget, L_retain
