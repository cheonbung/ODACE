"""Config schema invariants: unknown keys are errors, legacy migration is explicit and
recorded, t_min/t_max are never silently consumed, batch semantics are honest."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config_schema import ConfigError, load_config, resolve_config
from tests.conftest import legacy_cfg_dict, paper_cfg_dict


# ------------------------------------------------------------------ unknown keys
def test_unknown_key_rejected_paper():
    with pytest.raises(ConfigError, match="unknown config key"):
        resolve_config(paper_cfg_dict(totally_new_option=1))


def test_unknown_key_rejected_legacy():
    with pytest.raises(ConfigError, match="unknown config key"):
        resolve_config(legacy_cfg_dict(sample_giudance=3.0))   # typo must not pass


# ------------------------------------------------------------------ legacy migration
def test_legacy_benign_n1_migration():
    cfg = resolve_config(legacy_cfg_dict())
    assert cfg.execution_mode == "legacy_exact"
    assert cfg.target_mode == "anchor_contrastive"
    assert cfg.target_lambda == 1.0
    assert cfg.anchor_prompt == "a fully clothed person, photograph"
    assert cfg.trainable_projections == ("q", "k", "v", "out")
    assert cfg.trainable_scope == "all_xattn"
    assert cfg.timestep_policy == "legacy_trajectory_index_uniform"
    # legacy declared batch_size=4 was never used -> pinned to 1 and RECORDED
    assert cfg.batch_size == 1 and cfg.effective_batch_size == 1
    assert cfg.ignored_legacy_keys["batch_size"] == 4
    # t_min/t_max recorded as ignored, never silently dropped
    assert cfg.ignored_legacy_keys["t_min"] == 5
    assert cfg.ignored_legacy_keys["t_max"] == 950
    # eta present but unused by anchor_contrastive -> recorded
    assert cfg.ignored_legacy_keys["eta"] == 3.0
    assert cfg.eta is None
    assert any("batch_size" in n for n in cfg.migration_notes)


def test_legacy_negguide_keeps_eta():
    cfg = resolve_config(legacy_cfg_dict(erase_mode="negguide"))
    assert cfg.target_mode == "push"
    assert cfg.eta == 3.0
    assert cfg.target_lambda is None


def test_legacy_real_config_file_loads():
    # the actual repo config must migrate cleanly
    real = Path(__file__).resolve().parents[1] / "configs" / "nudity_odace_benign_n1.yaml"
    cfg = load_config(real)
    assert cfg.execution_mode == "legacy_exact"
    assert cfg.target_mode == "anchor_contrastive"
    assert cfg.target_lambda == 1.0
    assert cfg.batch_size == 1
    assert cfg.config_sha256 and len(cfg.config_sha256) == 64


# ------------------------------------------------------------------ paper mode strictness
def test_paper_forbids_legacy_keys():
    for k, v in (("erase_mode", "benign_neg"), ("xattn_full", True),
                 ("benign_neg_lambda", 1.0), ("t_min", 5), ("t_max", 950)):
        with pytest.raises(ConfigError):
            resolve_config(paper_cfg_dict(**{k: v}))


def test_paper_requires_execution_mode():
    d = paper_cfg_dict()
    del d["execution_mode"]
    with pytest.raises(ConfigError, match="execution_mode"):
        resolve_config(d)


def test_paper_batch_size_meaningful():
    cfg = resolve_config(paper_cfg_dict(batch_size=4, gradient_accumulation_steps=2))
    assert cfg.batch_size == 4
    assert cfg.effective_batch_size == 8


def test_unimplemented_timestep_policy_errors():
    with pytest.raises(ConfigError, match="NOT implemented"):
        resolve_config(paper_cfg_dict(timestep_policy="ddpm_timestep_uniform"))
    with pytest.raises(ConfigError, match="timestep_policy"):
        resolve_config(paper_cfg_dict(timestep_policy="whatever"))


def test_trajectory_index_bounds_validated():
    with pytest.raises(ConfigError, match="trajectory index range"):
        resolve_config(paper_cfg_dict(trajectory_index_min=0))
    with pytest.raises(ConfigError, match="trajectory index range"):
        resolve_config(paper_cfg_dict(trajectory_index_max=6))   # > ddim_steps-1 (5)
    with pytest.raises(ConfigError, match="trajectory index range"):
        resolve_config(paper_cfg_dict(trajectory_index_min=4, trajectory_index_max=2))


def test_target_mode_argument_coupling():
    with pytest.raises(ConfigError, match="target_lambda"):
        resolve_config(paper_cfg_dict(target_mode="push", eta=1.0, anchor_prompt=None))
    d = paper_cfg_dict(target_mode="push", eta=1.0)
    d.pop("target_lambda"); d.pop("anchor_prompt")
    assert resolve_config(d).target_mode == "push"
    with pytest.raises(ConfigError, match="eta"):
        resolve_config(paper_cfg_dict(eta=1.0))                  # eta on anchor_contrastive
    d = paper_cfg_dict(target_mode="anchor")
    d.pop("target_lambda")
    assert resolve_config(d).target_mode == "anchor"
    with pytest.raises(ConfigError, match="requires eta"):
        d2 = paper_cfg_dict(target_mode="push")
        d2.pop("target_lambda"); d2.pop("anchor_prompt")
        resolve_config(d2)


def test_anchor_policy_only_fixed():
    with pytest.raises(ConfigError, match="anchor_policy"):
        resolve_config(paper_cfg_dict(anchor_policy="context_matched"))


# ------------------------------------------------------------------ overrides
def test_cli_override_applied_and_legacy_batch_guard(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(yaml.safe_dump(legacy_cfg_dict()), encoding="utf-8")
    cfg = load_config(p, overrides={"num_steps": 2, "output_dir": "outputs/_x"})
    assert cfg.num_optimizer_steps == 2
    assert cfg.output_dir == "outputs/_x"
    with pytest.raises(ConfigError, match="batch_size override"):
        load_config(p, overrides={"batch_size": 4})
    q = tmp_path / "paper.yaml"
    q.write_text(yaml.safe_dump(paper_cfg_dict()), encoding="utf-8")
    assert load_config(q, overrides={"batch_size": 4}).batch_size == 4
