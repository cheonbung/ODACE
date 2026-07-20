"""Target-family invariants (plan section 8.2 G): legacy-formula equality, lambda=0
anchor identity, stop-gradient. CPU, synthetic tensors."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.targets import (anchor_contrastive_target, anchor_target, compute_target,
                          push_target)
from methods.unet_edit import eps_match_losses


def _e(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 8, 8, generator=g)


def test_push_equals_legacy_expression():
    e_u, e_c = _e(0), _e(1)
    for eta in (0.0, 0.5, 1.0, 3.0):
        legacy = (e_u - eta * (e_c - e_u)).detach()
        assert torch.equal(push_target(e_u, e_c, eta), legacy)


def test_anchor_contrastive_equals_legacy_benign_neg():
    e_b, e_c = _e(2), _e(3)
    for lam in (0.0, 0.5, 1.0, 2.0):
        legacy = (e_b - lam * (e_c - e_b)).detach()
        assert torch.equal(anchor_contrastive_target(e_b, e_c, lam), legacy)


def test_lambda_zero_is_exactly_anchor():
    e_b, e_c = _e(4), _e(5)
    assert torch.equal(anchor_contrastive_target(e_b, e_c, 0.0), anchor_target(e_b))


def test_eps_match_losses_target_is_push_target():
    # unet_edit legacy formula en + eta*(en - ef) == push_target(en, ef, eta)
    e_u, e_c = _e(6), _e(7)
    eta = 2.0
    ef = torch.zeros_like(e_u)         # student pred 0 => L_forget = mean(target^2)
    er = torch.zeros_like(e_u)
    Lf, _ = eps_match_losses(ef, e_u, e_c, er, er, eta=eta)
    expected = push_target(e_u, e_c, eta).pow(2).mean()
    assert torch.allclose(Lf, expected)


def test_no_gradient_flows_into_target():
    e_b = _e(8).requires_grad_(True)
    e_c = _e(9).requires_grad_(True)
    target, _ = compute_target("anchor_contrastive", eps_benign=e_b, eps_concept=e_c,
                               lam=1.0)
    assert not target.requires_grad
    student = torch.zeros_like(target).requires_grad_(True)
    loss = torch.nn.functional.mse_loss(student, target)
    loss.backward()
    assert student.grad is not None
    assert e_b.grad is None and e_c.grad is None


def test_compute_target_dispatch_and_missing_args():
    e_u, e_b, e_c = _e(10), _e(11), _e(12)
    t, _ = compute_target("push", eps_uncond=e_u, eps_concept=e_c, eta=1.0)
    assert torch.equal(t, push_target(e_u, e_c, 1.0))
    t, _ = compute_target("anchor", eps_benign=e_b)
    assert torch.equal(t, anchor_target(e_b))
    with pytest.raises(ValueError):
        compute_target("push", eps_uncond=e_u, eps_concept=e_c)          # eta missing
    with pytest.raises(ValueError):
        compute_target("anchor_contrastive", eps_benign=e_b, lam=1.0)   # concept missing
    with pytest.raises(ValueError):
        compute_target("nonsense", eps_benign=e_b)


def test_diagnostics_present_and_loss_free():
    e_b, e_c = _e(13), _e(14)
    t, diag = compute_target("anchor_contrastive", eps_benign=e_b, eps_concept=e_c,
                             lam=1.0, with_diagnostics=True)
    assert set(diag) >= {"target_norm", "ref_norm", "target_minus_concept_norm"}
    assert all(isinstance(v, float) for v in diag.values())
