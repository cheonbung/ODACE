"""ODACE unit tests (synthetic, CPU, fast). Run: pytest odace/tests -v"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.unet_edit import set_trainable_cross_attn_kv, eps_match_losses
from core.dataset import neutralize


class _Attn(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.is_cross_attention = cross
        self.to_q = nn.Linear(8, 8); self.to_k = nn.Linear(8, 8); self.to_v = nn.Linear(8, 8)


class _MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attn(False); self.attn2 = _Attn(True); self.other = nn.Linear(8, 8)


def test_set_trainable_only_cross_kv():
    u = _MockUNet(); n = set_trainable_cross_attn_kv(u)
    assert n > 0
    assert u.attn2.to_k.weight.requires_grad and u.attn2.to_v.weight.requires_grad
    assert not u.attn2.to_q.weight.requires_grad
    assert not u.attn1.to_k.weight.requires_grad
    assert not u.other.weight.requires_grad


def test_set_trainable_full_includes_q():
    u = _MockUNet(); n = set_trainable_cross_attn_kv(u, include_q_out=True)
    assert n > 0
    assert u.attn2.to_q.weight.requires_grad
    assert u.attn2.to_k.weight.requires_grad and u.attn2.to_v.weight.requires_grad
    assert not u.attn1.to_q.weight.requires_grad  # self-attn stays frozen
    assert not u.other.weight.requires_grad


def test_eps_match_zero_when_aligned():
    e = torch.randn(2, 4, 8, 8); r = torch.randn(2, 4, 8, 8)
    Lf, Lr = eps_match_losses(e, e, e, r, r, eta=1.0)
    assert Lf.item() < 1e-6 and Lr.item() < 1e-6


def test_eps_match_eta_grad():
    ef = torch.randn(2, 4, 8, 8, requires_grad=True)
    en = torch.randn(2, 4, 8, 8); eff = torch.randn(2, 4, 8, 8)
    er = torch.randn(2, 4, 8, 8, requires_grad=True); erf = torch.randn(2, 4, 8, 8)
    Lf0, _ = eps_match_losses(ef, en, eff, er, erf, eta=0.0)
    Lf2, _ = eps_match_losses(ef, en, eff, er, erf, eta=2.0)
    Lf2.backward()
    assert ef.grad is not None and Lf0.item() >= 0 and Lf2.item() >= 0


def test_neutralize_strips_concept():
    assert "naked" not in neutralize("a naked woman standing").lower()
    assert "nude" not in neutralize("a photo of a nude person").lower()
    assert "woman" in neutralize("a naked woman standing").lower()
