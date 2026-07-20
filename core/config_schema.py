"""Strict ODACE config schema: unknown keys are errors, legacy keys migrate explicitly.

Two execution modes (plan P1.9):

  legacy_exact   -- reproduces the pre-refactor trainer bit-for-bit: effective batch 1
                    (the legacy loop always trained on 1 prompt/step regardless of the
                    declared batch_size), legacy trajectory-index timestep convention
                    including its off-by-one (z after N DDIM steps paired with the
                    timestep of step N-1), global-RNG seeding. t_min/t_max stay ignored
                    but are RECORDED as ignored, never silently dropped.
  paper_aligned  -- new-semantics runs: batch_size is the real number of examples per
                    optimizer step (micro-batch accumulation), timestep policy is
                    explicit, the trajectory latent/timestep pair is index-aligned, and
                    local RNGs are used. Legacy keys (erase_mode, xattn_full,
                    benign_neg_lambda, benign_prompt, t_min, t_max) are hard errors.

Legacy key migration (recorded in migration_notes / ignored_legacy_keys):
  erase_mode: negguide|benign_anchor|benign_neg -> target_mode: push|anchor|anchor_contrastive
  benign_neg_lambda -> target_lambda      benign_prompt -> anchor_prompt
  xattn_full: true -> trainable_projections [q,k,v,out] ; false -> [k,v]
  batch_size (legacy) -> recorded as legacy_declared_batch_size, effective batch forced 1
  t_min/t_max -> ignored_legacy_keys (legacy trainer never read them)

Callers: train_odace.py, core/trainer.py, tests/test_config_schema.py,
experiments/regression_compare.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from methods.layer_selection import PROJECTIONS, SCOPES

SCHEMA_VERSION = "odace-config-v1"

EXECUTION_MODES = ("legacy_exact", "paper_aligned")
TARGET_MODES = ("push", "anchor", "anchor_contrastive")
TIMESTEP_POLICIES_IMPLEMENTED = ("legacy_trajectory_index_uniform",)
TIMESTEP_POLICIES_KNOWN = TIMESTEP_POLICIES_IMPLEMENTED + (
    "ddpm_timestep_uniform", "separability_weighted", "sderasure_step_selection")

_ERASE_MODE_MAP = {"negguide": "push", "benign_anchor": "anchor",
                   "benign_neg": "anchor_contrastive"}

# keys the legacy configs/nudity_odace*.yaml family may contain -- nothing else
_LEGACY_ALLOWED = {
    "experiment_name", "target_concept", "sd_model_id", "learning_rate", "num_steps",
    "batch_size", "seed", "log_every", "alpha", "beta", "t_min", "t_max",
    "forget_prompts_file", "retain_prompts_file", "ood_aug_file", "output_dir",
    "erase_mode", "benign_prompt", "benign_neg_lambda", "eta", "ddim_steps",
    "sample_guidance", "xattn_full", "execution_mode",
}
# keys that only exist in the legacy schema; presence marks a config as legacy
_LEGACY_ONLY = {"erase_mode", "benign_prompt", "benign_neg_lambda", "xattn_full",
                "t_min", "t_max"}

_PAPER_ALLOWED = {
    "method_name", "experiment_name", "target_concept", "sd_model_id", "execution_mode",
    "target_mode", "target_lambda", "eta", "anchor_policy", "anchor_prompt",
    "trainable_scope", "explicit_layers", "trainable_projections", "timestep_policy",
    "trajectory_index_min", "trajectory_index_max", "ddim_steps", "sample_guidance",
    "learning_rate", "alpha", "beta", "num_steps", "batch_size",
    "gradient_accumulation_steps", "forget_prompts_file", "retain_prompts_file",
    "ood_aug_file", "output_dir", "seed", "log_every", "max_length",
}


class ConfigError(ValueError):
    """Any schema violation: unknown key, missing key, bad value, forbidden combination."""


@dataclass
class ResolvedODACEConfig:
    """Fully validated, migration-applied config. This is what the trainer consumes."""

    schema_version: str
    execution_mode: str
    experiment_name: str
    target_concept: str
    sd_model_id: str
    target_mode: str
    target_lambda: Optional[float]
    eta: Optional[float]
    anchor_policy: str
    anchor_prompt: Optional[str]
    trainable_scope: str
    trainable_projections: Tuple[str, ...]
    explicit_layers: Optional[Tuple[str, ...]]
    timestep_policy: str
    trajectory_index_min: int
    trajectory_index_max: int
    ddim_steps: int
    sample_guidance: float
    learning_rate: float
    alpha: float
    beta: float
    num_optimizer_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    forget_prompts_file: str
    retain_prompts_file: str
    ood_aug_file: Optional[str]
    output_dir: str
    seed: int
    log_every: int
    max_length: int
    # provenance
    source_path: Optional[str] = None
    config_sha256: Optional[str] = None
    raw_config: dict = field(default_factory=dict)
    migration_notes: List[str] = field(default_factory=list)
    ignored_legacy_keys: Dict[str, object] = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def to_manifest_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "experiment_name": self.experiment_name,
            "target_concept": self.target_concept,
            "sd_model_id": self.sd_model_id,
            "target_mode": self.target_mode,
            "target_lambda": self.target_lambda,
            "eta": self.eta,
            "anchor_policy": self.anchor_policy,
            "anchor_prompt": self.anchor_prompt,
            "trainable_scope": self.trainable_scope,
            "trainable_projections": list(self.trainable_projections),
            "explicit_layers": list(self.explicit_layers) if self.explicit_layers else None,
            "timestep_policy": self.timestep_policy,
            "trajectory_index_min": self.trajectory_index_min,
            "trajectory_index_max": self.trajectory_index_max,
            "ddim_steps": self.ddim_steps,
            "sample_guidance": self.sample_guidance,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_optimizer_steps": self.num_optimizer_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "forget_prompts_file": self.forget_prompts_file,
            "retain_prompts_file": self.retain_prompts_file,
            "ood_aug_file": self.ood_aug_file,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "log_every": self.log_every,
            "max_length": self.max_length,
            "source_path": self.source_path,
            "config_sha256": self.config_sha256,
            "migration_notes": self.migration_notes,
            "ignored_legacy_keys": self.ignored_legacy_keys,
        }
        return d


def _require(raw: dict, key: str, typ, ctx: str):
    if key not in raw or raw[key] is None:
        raise ConfigError(f"{ctx}: required key '{key}' missing")
    return _coerce(raw[key], key, typ)


def _coerce(value, key: str, typ):
    if typ is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"key '{key}' must be int, got {value!r}")
        return value
    if typ is str and isinstance(value, str):
        return value
    if typ is bool and isinstance(value, bool):
        return value
    if typ is float:
        raise ConfigError(f"key '{key}' must be a number, got {value!r}")
    raise ConfigError(f"key '{key}' must be {typ.__name__}, got {value!r}")


def _is_legacy(raw: dict) -> bool:
    if raw.get("execution_mode") == "paper_aligned":
        return False
    if raw.get("execution_mode") == "legacy_exact":
        return True
    return bool(_LEGACY_ONLY & set(raw))


def _check_unknown(raw: dict, allowed: set, ctx: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"{ctx}: unknown config key(s) {unknown} -- unknown keys are rejected so a "
            f"typo or unimplemented option can never silently shape a result")


def _validate_common(cfg: ResolvedODACEConfig) -> None:
    if cfg.learning_rate <= 0:
        raise ConfigError("learning_rate must be > 0")
    if cfg.num_optimizer_steps <= 0:
        raise ConfigError("num_steps must be > 0")
    if cfg.batch_size < 1:
        raise ConfigError("batch_size must be >= 1")
    if cfg.gradient_accumulation_steps < 1:
        raise ConfigError("gradient_accumulation_steps must be >= 1")
    if cfg.ddim_steps < 2:
        raise ConfigError("ddim_steps must be >= 2")
    if cfg.target_mode not in TARGET_MODES:
        raise ConfigError(f"target_mode must be one of {TARGET_MODES}")
    if cfg.target_mode == "anchor_contrastive" and cfg.target_lambda is None:
        raise ConfigError("target_mode=anchor_contrastive requires target_lambda")
    if cfg.target_mode == "push" and cfg.eta is None:
        raise ConfigError("target_mode=push requires eta")
    if cfg.target_mode in ("anchor", "anchor_contrastive") and not cfg.anchor_prompt:
        raise ConfigError(f"target_mode={cfg.target_mode} requires anchor_prompt")
    if cfg.anchor_policy != "fixed":
        raise ConfigError(f"anchor_policy '{cfg.anchor_policy}' not implemented; only "
                          f"'fixed' exists (multi/context anchors are a later phase)")
    if cfg.trainable_scope not in SCOPES:
        raise ConfigError(f"trainable_scope must be one of {SCOPES}")
    bad = [p for p in cfg.trainable_projections if p not in PROJECTIONS]
    if bad or not cfg.trainable_projections:
        raise ConfigError(f"trainable_projections must be a non-empty subset of "
                          f"{PROJECTIONS}, got {list(cfg.trainable_projections)}")
    if cfg.timestep_policy not in TIMESTEP_POLICIES_KNOWN:
        raise ConfigError(f"timestep_policy must be one of {TIMESTEP_POLICIES_KNOWN}")
    if cfg.timestep_policy not in TIMESTEP_POLICIES_IMPLEMENTED:
        raise ConfigError(
            f"timestep_policy '{cfg.timestep_policy}' is defined by the protocol but NOT "
            f"implemented yet -- refusing to run rather than silently substituting")
    if not (1 <= cfg.trajectory_index_min <= cfg.trajectory_index_max
            <= cfg.ddim_steps - 1):
        raise ConfigError(
            f"trajectory index range [{cfg.trajectory_index_min}, "
            f"{cfg.trajectory_index_max}] must satisfy 1 <= min <= max <= ddim_steps-1 "
            f"(= {cfg.ddim_steps - 1})")


def _resolve_legacy(raw: dict, ctx: str) -> ResolvedODACEConfig:
    _check_unknown(raw, _LEGACY_ALLOWED, ctx)
    notes: List[str] = []
    ignored: Dict[str, object] = {}

    erase_mode = raw.get("erase_mode", "negguide")
    if erase_mode not in _ERASE_MODE_MAP:
        raise ConfigError(f"{ctx}: unknown erase_mode '{erase_mode}'")
    target_mode = _ERASE_MODE_MAP[erase_mode]
    notes.append(f"erase_mode: {erase_mode} -> target_mode: {target_mode}")

    target_lambda = None
    if target_mode == "anchor_contrastive":
        target_lambda = _coerce(raw.get("benign_neg_lambda", 1.0), "benign_neg_lambda",
                                float)
        notes.append(f"benign_neg_lambda -> target_lambda: {target_lambda}")
    elif "benign_neg_lambda" in raw:
        ignored["benign_neg_lambda"] = raw["benign_neg_lambda"]

    eta = None
    if target_mode == "push":
        eta = _coerce(raw.get("eta", 1.0), "eta", float)
    elif "eta" in raw:
        ignored["eta"] = raw["eta"]
        notes.append("eta present but unused by this target_mode (recorded, not applied)")

    anchor_prompt = None
    if target_mode in ("anchor", "anchor_contrastive"):
        anchor_prompt = str(raw.get("benign_prompt", "a fully clothed person, photograph"))
        notes.append("benign_prompt -> anchor_prompt")
    elif "benign_prompt" in raw:
        ignored["benign_prompt"] = raw["benign_prompt"]

    xattn_full = bool(raw.get("xattn_full", False))
    projections = ("q", "k", "v", "out") if xattn_full else ("k", "v")
    notes.append(f"xattn_full: {xattn_full} -> trainable_projections: {list(projections)}")

    declared_bs = raw.get("batch_size")
    if declared_bs is not None and declared_bs != 1:
        ignored["batch_size"] = declared_bs
        notes.append(
            f"legacy declared batch_size={declared_bs} was NEVER used by the legacy loop "
            f"(effective batch was 1); legacy_exact runs pin batch_size=1 -- do not call "
            f"a batch={declared_bs} rerun a reproduction")
    for k in ("t_min", "t_max"):
        if k in raw:
            ignored[k] = raw[k]
            notes.append(f"{k}={raw[k]} was silently ignored by the legacy trainer; "
                         f"recorded as ignored (legacy timestep convention preserved)")

    ddim_steps = _coerce(raw.get("ddim_steps", 30), "ddim_steps", int)
    cfg = ResolvedODACEConfig(
        schema_version=SCHEMA_VERSION,
        execution_mode="legacy_exact",
        experiment_name=_require(raw, "experiment_name", str, ctx),
        target_concept=str(raw.get("target_concept", "nudity")),
        sd_model_id=_require(raw, "sd_model_id", str, ctx),
        target_mode=target_mode,
        target_lambda=target_lambda,
        eta=eta,
        anchor_policy="fixed",
        anchor_prompt=anchor_prompt,
        trainable_scope="all_xattn",
        trainable_projections=projections,
        explicit_layers=None,
        timestep_policy="legacy_trajectory_index_uniform",
        trajectory_index_min=1,
        trajectory_index_max=ddim_steps - 1,
        ddim_steps=ddim_steps,
        sample_guidance=_coerce(raw.get("sample_guidance", 3.0), "sample_guidance", float),
        learning_rate=_coerce(raw.get("learning_rate", 1e-5), "learning_rate", float),
        alpha=_coerce(raw.get("alpha", 1.0), "alpha", float),
        beta=_coerce(raw.get("beta", 1.0), "beta", float),
        num_optimizer_steps=_coerce(raw.get("num_steps", 400), "num_steps", int),
        batch_size=1,
        gradient_accumulation_steps=1,
        forget_prompts_file=_require(raw, "forget_prompts_file", str, ctx),
        retain_prompts_file=_require(raw, "retain_prompts_file", str, ctx),
        ood_aug_file=raw.get("ood_aug_file"),
        output_dir=str(raw.get("output_dir", "outputs/odace")),
        seed=_coerce(raw.get("seed", 42), "seed", int),
        log_every=_coerce(raw.get("log_every", 25), "log_every", int),
        max_length=77,
        raw_config=dict(raw),
        migration_notes=notes,
        ignored_legacy_keys=ignored,
    )
    _validate_common(cfg)
    return cfg


def _resolve_paper(raw: dict, ctx: str) -> ResolvedODACEConfig:
    forbidden = sorted(_LEGACY_ONLY & set(raw))
    if forbidden:
        raise ConfigError(
            f"{ctx}: legacy key(s) {forbidden} are forbidden in paper_aligned configs -- "
            f"use target_mode/target_lambda/anchor_prompt/trainable_projections/"
            f"timestep_policy instead")
    _check_unknown(raw, _PAPER_ALLOWED, ctx)
    if raw.get("method_name", "odace") != "odace":
        raise ConfigError(f"{ctx}: method_name must be 'odace'")

    target_mode = _require(raw, "target_mode", str, ctx)
    if target_mode not in TARGET_MODES:
        raise ConfigError(f"{ctx}: target_mode must be one of {TARGET_MODES}")
    target_lambda = raw.get("target_lambda")
    if target_lambda is not None:
        target_lambda = _coerce(target_lambda, "target_lambda", float)
        if target_mode != "anchor_contrastive":
            raise ConfigError(f"{ctx}: target_lambda only valid for anchor_contrastive")
    eta = raw.get("eta")
    if eta is not None:
        eta = _coerce(eta, "eta", float)
        if target_mode != "push":
            raise ConfigError(f"{ctx}: eta only valid for target_mode=push")
    anchor_prompt = raw.get("anchor_prompt")
    if anchor_prompt is not None and target_mode == "push":
        raise ConfigError(f"{ctx}: anchor_prompt is meaningless for target_mode=push")

    ddim_steps = _coerce(raw.get("ddim_steps", 30), "ddim_steps", int)
    projections = raw.get("trainable_projections", ["q", "k", "v", "out"])
    if not isinstance(projections, (list, tuple)):
        raise ConfigError(f"{ctx}: trainable_projections must be a list")
    explicit_layers = raw.get("explicit_layers")
    if explicit_layers is not None:
        if not isinstance(explicit_layers, (list, tuple)) or not explicit_layers:
            raise ConfigError(f"{ctx}: explicit_layers must be a non-empty list")
        explicit_layers = tuple(str(x) for x in explicit_layers)

    cfg = ResolvedODACEConfig(
        schema_version=SCHEMA_VERSION,
        execution_mode="paper_aligned",
        experiment_name=_require(raw, "experiment_name", str, ctx),
        target_concept=str(raw.get("target_concept", "nudity")),
        sd_model_id=_require(raw, "sd_model_id", str, ctx),
        target_mode=target_mode,
        target_lambda=target_lambda,
        eta=eta,
        anchor_policy=str(raw.get("anchor_policy", "fixed")),
        anchor_prompt=anchor_prompt,
        trainable_scope=str(raw.get("trainable_scope", "all_xattn")),
        trainable_projections=tuple(str(p) for p in projections),
        explicit_layers=explicit_layers,
        timestep_policy=_require(raw, "timestep_policy", str, ctx),
        trajectory_index_min=_coerce(raw.get("trajectory_index_min", 1),
                                     "trajectory_index_min", int),
        trajectory_index_max=_coerce(raw.get("trajectory_index_max", ddim_steps - 1),
                                     "trajectory_index_max", int),
        ddim_steps=ddim_steps,
        sample_guidance=_coerce(raw.get("sample_guidance", 3.0), "sample_guidance", float),
        learning_rate=_require(raw, "learning_rate", float, ctx),
        alpha=_coerce(raw.get("alpha", 1.0), "alpha", float),
        beta=_coerce(raw.get("beta", 1.0), "beta", float),
        num_optimizer_steps=_require(raw, "num_steps", int, ctx),
        batch_size=_require(raw, "batch_size", int, ctx),
        gradient_accumulation_steps=_coerce(
            raw.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps", int),
        forget_prompts_file=_require(raw, "forget_prompts_file", str, ctx),
        retain_prompts_file=_require(raw, "retain_prompts_file", str, ctx),
        ood_aug_file=raw.get("ood_aug_file"),
        output_dir=_require(raw, "output_dir", str, ctx),
        seed=_coerce(raw.get("seed", 42), "seed", int),
        log_every=_coerce(raw.get("log_every", 25), "log_every", int),
        max_length=_coerce(raw.get("max_length", 77), "max_length", int),
        raw_config=dict(raw),
        migration_notes=[],
        ignored_legacy_keys={},
    )
    _validate_common(cfg)
    return cfg


def resolve_config(raw: dict, source_path: Optional[str] = None,
                   config_sha256: Optional[str] = None) -> ResolvedODACEConfig:
    """Validate + migrate an already-parsed config dict. Raises ConfigError on any issue."""
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("config must be a non-empty mapping")
    ctx = source_path or "<config>"
    if "execution_mode" in raw and raw["execution_mode"] not in EXECUTION_MODES:
        raise ConfigError(f"{ctx}: execution_mode must be one of {EXECUTION_MODES}")
    if _is_legacy(raw):
        cfg = _resolve_legacy(raw, ctx)
    else:
        if "execution_mode" not in raw:
            raise ConfigError(
                f"{ctx}: config has no legacy keys and no execution_mode -- new configs "
                f"must declare execution_mode: paper_aligned explicitly")
        cfg = _resolve_paper(raw, ctx)
    cfg.source_path = source_path
    cfg.config_sha256 = config_sha256
    return cfg


def load_config(path: str | Path,
                overrides: Optional[dict] = None) -> ResolvedODACEConfig:
    """Read YAML, apply CLI overrides BEFORE validation, resolve + validate.

    overrides may only touch keys valid for the config's schema; a legacy_exact config
    rejects a batch_size override != 1 (its reproduction semantics are batch 1).
    """
    p = Path(path)
    raw_bytes = p.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    sha = hashlib.sha256(raw_bytes).hexdigest()
    if overrides:
        legacy = _is_legacy(raw)
        for k, v in overrides.items():
            if v is None:
                continue
            if legacy and k == "batch_size" and v != 1:
                raise ConfigError(
                    "legacy_exact config: batch_size override must be 1 (the legacy "
                    "trainer's effective batch); use a paper_aligned config for real "
                    "batching")
            raw[k] = v
    return resolve_config(raw, source_path=str(p), config_sha256=sha)
