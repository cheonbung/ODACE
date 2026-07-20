"""Scheduler-index alignment of _sample_until (plan section 6.2): with an eps=0 teacher
the DDIM trajectory is a deterministic function of z0, so we can replay it manually and
check exactly WHICH state and WHICH timestep each execution mode returns.

  states[i]   = latent BEFORE the step at tlist[i]  (states[0] = z0)
  legacy_exact  returns (states[k], tlist[k-1])  -- documented off-by-one
  paper_aligned returns (states[k], tlist[k])    -- index-aligned (convention 2)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import ZeroTeacher, fake_encode, legacy_cfg_dict, make_trainer, \
    paper_cfg_dict

DDIM_STEPS = 6
K = 3   # trajectory index to probe (1 <= K <= DDIM_STEPS-1)


def _manual_rollout(scheduler, z0, n_steps, device):
    """states[i] = latent before the step at tlist[i], for the eps=0 teacher."""
    scheduler.set_timesteps(DDIM_STEPS, device=device)
    tlist = scheduler.timesteps
    states = [z0]
    z = z0
    for t in tlist[:n_steps]:
        eps = torch.zeros(z.shape, dtype=torch.float32)
        z = scheduler.step(eps, t, z).prev_sample
        states.append(z)
    return states, tlist


def _probe(cfg_dict, seed=7):
    tr = make_trainer(cfg_dict, teacher=ZeroTeacher())
    device = torch.device("cpu")
    # reproduce the exact initial latent the trainer will draw
    if tr._latent_generator is None:
        torch.manual_seed(seed)
        z0 = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        torch.manual_seed(seed)                      # rewind for the real call
    else:
        tr._latent_generator.manual_seed(seed)
        z0 = torch.randn(1, 4, 64, 64, dtype=torch.float16,
                         generator=torch.Generator().manual_seed(seed))
        tr._latent_generator.manual_seed(seed)       # rewind for the real call
    z0 = z0 * tr.ddim.init_noise_sigma
    c_f = fake_encode(["probe prompt"])
    z, t = tr._sample_until(c_f, K)
    states, tlist = _manual_rollout(tr.ddim, z0, K, device)
    return z, int(t), states, [int(x) for x in tlist]


def test_legacy_mode_returns_offbyone_pair():
    z, t, states, tlist = _probe(legacy_cfg_dict(ddim_steps=DDIM_STEPS))
    assert torch.allclose(z.float(), states[K].float(), atol=1e-6), \
        "legacy z is not the state after K steps"
    assert t == tlist[K - 1], "legacy timestep must be the step it JUST took (off-by-one)"
    assert t != tlist[K]


def test_aligned_mode_returns_matching_pair():
    z, t, states, tlist = _probe(
        paper_cfg_dict(ddim_steps=DDIM_STEPS, trajectory_index_max=DDIM_STEPS - 1))
    assert torch.allclose(z.float(), states[K].float(), atol=1e-6), \
        "aligned z is not the state after K steps"
    assert t == tlist[K], "aligned timestep must be the index-matched NEXT timestep"


def test_modes_differ_explicitly():
    _, t_legacy, _, tlist = _probe(legacy_cfg_dict(ddim_steps=DDIM_STEPS))
    _, t_aligned, _, _ = _probe(
        paper_cfg_dict(ddim_steps=DDIM_STEPS, trajectory_index_max=DDIM_STEPS - 1))
    assert t_legacy != t_aligned
    assert tlist.index(t_aligned) - tlist.index(t_legacy) == 1


def test_trajectory_index_bounds_are_honored():
    tr = make_trainer(paper_cfg_dict(ddim_steps=DDIM_STEPS, trajectory_index_min=2,
                                     trajectory_index_max=4), teacher=ZeroTeacher())
    draws = {tr._prompt_rng.randint(tr.cfg.trajectory_index_min,
                                    tr.cfg.trajectory_index_max) for _ in range(200)}
    assert draws <= {2, 3, 4} and draws == {2, 3, 4}
