"""Batch semantics + selection freezing on the refactored trainer (tiny CPU components):
paper_aligned batch_size really accumulates N examples per optimizer step, and every
parameter outside the trainable selection is BIT-identical after optimizer steps."""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import legacy_cfg_dict, make_tiny_sd1_unet, make_trainer, \
    paper_cfg_dict

PAIRS = [("a nude figure on a beach", "a person on a beach"),
         ("nude portrait painting", "portrait painting"),
         ("a naked statue in a garden", "a statue in a garden")]


def test_microbatch_step_equals_scaled_sum_step():
    """A: one _train_step with batch_size=3. B: same three examples, losses summed/3,
    ONE backward + step. Mathematically identical updates expected (<=1e-6)."""
    base = make_tiny_sd1_unet(seed=3)
    ta = make_trainer(paper_cfg_dict(batch_size=3), unet=copy.deepcopy(base))
    tb = make_trainer(paper_cfg_dict(batch_size=3), unet=copy.deepcopy(base))

    ta._train_step([p[0] for p in PAIRS], [p[1] for p in PAIRS])

    tb.optimizer.zero_grad()
    total = None
    for fp, rp in PAIRS:
        Lf, Lr = tb._example_losses(fp, rp)
        term = (tb.cfg.alpha * Lf + tb.cfg.beta * Lr) / 3
        total = term if total is None else total + term
    total.backward()
    tb.optimizer.step()

    for (na, pa), (nb, pb) in zip(sorted(ta.unet.named_parameters()),
                                  sorted(tb.unet.named_parameters())):
        assert na == nb
        assert torch.allclose(pa, pb, atol=1e-6), f"param {na} diverged"


def test_batch_counters_and_history():
    tr = make_trainer(paper_cfg_dict(batch_size=2))
    s = tr._train_step([PAIRS[0][0], PAIRS[1][0]], [PAIRS[0][1], PAIRS[1][1]])
    assert tr.counters["optimizer_steps"] == 1
    assert tr.counters["examples_forget"] == 2
    assert tr.counters["backward"] == 2
    assert set(s) == {"L_forget", "L_retain", "L_total"}


def test_legacy_mode_is_single_example():
    tr = make_trainer(legacy_cfg_dict())
    assert tr.cfg.batch_size == 1
    tr._train_step([PAIRS[0][0]], [PAIRS[0][1]])
    assert tr.counters["examples_forget"] == 1
    assert tr.counters["optimizer_steps"] == 1


def test_params_outside_selection_bit_identical_after_updates():
    """res16 scope: run 2 optimizer steps; every non-selected parameter must be
    BIT-identical (torch.equal), and the selection must move."""
    tr = make_trainer(paper_cfg_dict(trainable_scope="res16", batch_size=1))
    assert len(tr.selection.module_names) == 5
    selected = set(tr.selection.parameter_names)
    before = {n: p.detach().clone() for n, p in tr.unet.named_parameters()}

    for fp, rp in PAIRS[:2]:
        tr._train_step([fp], [rp])

    changed_inside = 0
    for n, p in tr.unet.named_parameters():
        if n in selected:
            if not torch.equal(before[n], p):
                changed_inside += 1
        else:
            assert torch.equal(before[n], p), f"non-selected param {n} changed"
    assert changed_inside > 0, "no selected parameter moved -- optimizer dead"


def test_train_loop_draws_batch_prompts():
    class _DS:
        forget_prompts = [p[0] for p in PAIRS]
        retain_prompts = [p[1] for p in PAIRS]

    tr = make_trainer(paper_cfg_dict(batch_size=2, num_steps=2))
    hist = tr.train(_DS(), num_steps=2, log_every=1)
    assert len(hist) == 2
    assert tr.counters["examples_forget"] == 4      # 2 steps x batch 2
    assert tr.counters["optimizer_steps"] == 2
