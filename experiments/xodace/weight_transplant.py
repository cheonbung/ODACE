"""E1 -- ODACE weight transplant / restoration: is ODACE's advantage MEDIATED by the 16x16
cross-attention band that the causal patching found?

Stage 1/2 established, in a FROZEN raw SD1.4, that transplanting a matched benign counterfactual's
cross-attn signal erases nudity, and that the 16x16 band (down_blocks.2 + up_blocks.1) alone
recovers 100% of the achievable erasure. That is an ORACLE result: it says nothing yet about the
trained model. benign-neg ODACE is trained to approximate exactly that redirection, so the question
this file answers is:

    does the TRAINED ODACE actually route its erasure through the same band?

Two directions, both needed:

  sufficiency   SD + dW(res16)      raw SD, plus ONLY ODACE's weight change on the 16x16 attn2.
                If it reproduces ODACE's erasure AND coherence, that band carries the method.
  necessity     ODACE - dW(res16)   ODACE, with ONLY that band restored to raw SD.
                If ODACE's advantage disappears, the band is required.

Controls (a band can look causal just by being big or by being where the parameters are):
  SD + dW(res32/res64/res8)  the same construction on every other band. The activation patching
                   ranked the bands 1.00 / 0.22 / 0.11 / 0.10 (res16/32/64/8); if the weight
                   transplant reproduces that ORDER, the oracle circuit and the trained edit agree.
  SD + dW(randel)  THE parameter control. A random-LAYER subset matched to res16's parameter count
                   is impossible: res16 holds 59.6% of dW (26.2M params), more than every other
                   attn2 layer combined (17.7M). So the control instead keeps the parameter budget
                   and the dW values but destroys the STRUCTURE: a uniformly random subset of dW
                   ENTRIES, of exactly res16's size, spread over all 16 layers. If res16 beats it,
                   the cause is where the change sits, not how much of it there is.
  SD + dW(all)     sanity: must be bit-identical to the ODACE checkpoint itself

Every ODACE config sets xattn_full: true, so trainer.py:62's set_trainable_cross_attn_kv(include_q_out
=True) trains the WHOLE cross-attn block -- to_q, to_k, to_v AND to_out (43.96M params across 16 attn2
layers, ~5 tensors each), not K/V alone. Nothing outside attn2 is trainable. That is verified here,
not assumed: a nonzero dW on any non-attn2 weight aborts the build, because every claim below is
premised on the edit being cross-attn-local.

Layer bands (attn2 module name -> feature resolution at 512x512), verified against the live UNet:
  res64 down_blocks.0 / up_blocks.3   res32 down_blocks.1 / up_blocks.2
  res16 down_blocks.2 / up_blocks.1   res8  mid_block

Reads : CompVis/stable-diffusion-v1-4 (unet), models/odace/outputs/<odace>/final
Writes: models/odace/outputs/xplant_<name>/final  (diffusers UNet dir, loadable by xeval kind=odace)
        models/odace/outputs/xplant_manifest.json

Caller: eval/run_xodace_e1.sh. Not imported by anything.

Run (WSL conda lsse):
  python models/odace/experiments/xodace/weight_transplant.py --dry_run    # report dW, write nothing
  python models/odace/experiments/xodace/weight_transplant.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import diffusers
import torch
from diffusers import UNet2DConditionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # models/odace
from methods.layer_selection import BANDS_SD1  # noqa: E402  single source of truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
BASE_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_ODACE = ROOT / "models" / "odace" / "outputs" / "odace_benign_n1" / "final"
OUT_ROOT = ROOT / "models" / "odace" / "outputs"

# attn2 module-name prefix -> band. A parameter belongs to a band iff its name starts with one of
# these AND contains ".attn2.". Defined ONCE in methods/layer_selection.py (imported above).
BANDS: dict[str, tuple[str, ...]] = BANDS_SD1


def is_xattn(name: str) -> bool:
    return ".attn2." in name


def band_of(name: str) -> str | None:
    for band, prefixes in BANDS.items():
        if any(name.startswith(p) for p in prefixes):
            return band
    return None


def layer_id(name: str) -> str:
    """attn2 module path, e.g. 'down_blocks.2.attentions.1.transformer_blocks.0.attn2'."""
    return name.split(".attn2.")[0] + ".attn2"


def delta(sd_base: dict, sd_odace: dict) -> dict[str, torch.Tensor]:
    """dW = W_odace - W_sd, only where it is actually nonzero. Aborts if ODACE moved anything
    outside cross-attn, because every claim below assumes the edit is cross-attn-local."""
    if set(sd_base) != set(sd_odace):
        raise SystemExit("state_dict key mismatch between SD and ODACE")
    dw, off_target = {}, []
    for k, v in sd_odace.items():
        d = v.float() - sd_base[k].float()
        if torch.count_nonzero(d) == 0:
            continue
        if not is_xattn(k):
            off_target.append((k, float(d.norm())))
        dw[k] = d
    if off_target:
        top = sorted(off_target, key=lambda x: -x[1])[:5]
        raise SystemExit(f"ODACE changed NON-cross-attn weights, the transplant premise is void: {top}")
    return dw


def summarize(dw: dict[str, torch.Tensor]) -> dict:
    per_layer: dict[str, dict] = {}
    for k, d in dw.items():
        lid, b = layer_id(k), band_of(k)
        e = per_layer.setdefault(lid, {"band": b, "n_params": 0, "sq": 0.0, "tensors": []})
        e["n_params"] += d.numel()
        e["sq"] += float(d.pow(2).sum())
        e["tensors"].append(k.split(".attn2.")[1])
    for e in per_layer.values():
        e["l2"] = round(e.pop("sq") ** 0.5, 4)
    return per_layer


def build(sd_base: dict, sd_odace: dict, dw: dict[str, torch.Tensor], names: set[str],
          mode: str) -> tuple[dict, int]:
    """mode add: SD, with the listed attn2 layers taken from ODACE. mode restore: ODACE, with the
    listed layers taken from SD. Returns a new state_dict; nothing is mutated in place.

    The transplanted value is COPIED from the source tensor, never rebuilt as base + (odace - base):
    in floating point a + (b - a) is not bit-identical to b, and that 1-ULP drift breaks the
    SD + dW(all) == ODACE sanity check (it did, on all 80 dW tensors, before this was fixed)."""
    out = {k: v.clone() for k, v in sd_base.items()}
    applied = 0
    for k, d in dw.items():
        take_odace = layer_id(k) in names
        if mode == "restore":
            take_odace = not take_odace
        if take_odace:
            out[k] = sd_odace[k].clone()
            applied += d.numel()
    return out, applied


def build_random_entries(sd_base: dict, sd_odace: dict, dw: dict[str, torch.Tensor],
                         budget: int, seed: int) -> tuple[dict, int]:
    """The parameter control: take a uniformly random subset of dW ENTRIES of size `budget`,
    spread over every attn2 layer. Same parameter budget and the same ODACE values as res16, but no
    structure -- so a res16 win cannot be explained by parameter count alone."""
    total = sum(d.numel() for d in dw.values())
    p = budget / total
    g = torch.Generator().manual_seed(seed)
    out = {k: v.clone() for k, v in sd_base.items()}
    applied = 0
    for k in dw:
        mask = torch.rand(sd_base[k].shape, generator=g) < p
        applied += int(mask.sum())
        out[k] = torch.where(mask, sd_odace[k], sd_base[k])
    logger.info(f"random-entry control: {applied:,} entries applied "
                f"(target {budget:,}, p={p:.3f} of dW)")
    return out, applied


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odace", type=Path, default=DEFAULT_ODACE)
    ap.add_argument("--prefix", default="xplant")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", default=None,
                    help="comma list of spec suffixes to build, e.g. sd_p_res16,odace_m_res16. "
                         "sd_p_all is always built -- it is the bit-exactness gate.")
    ap.add_argument("--dry_run", action="store_true", help="report dW only, write nothing")
    args = ap.parse_args()

    logger.info(f"base={BASE_ID} | odace={args.odace}")
    base = UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet")
    odace = UNet2DConditionModel.from_pretrained(args.odace)
    sd_base, sd_odace = base.state_dict(), odace.state_dict()

    dw = delta(sd_base, sd_odace)
    per_layer = summarize(dw)
    total = sum(e["n_params"] for e in per_layer.values())

    by_band: dict[str, dict] = {}
    for lid, e in sorted(per_layer.items()):
        b = by_band.setdefault(e["band"], {"layers": [], "n_params": 0, "l2": 0.0})
        b["layers"].append(lid)
        b["n_params"] += e["n_params"]
        b["l2"] += e["l2"]

    logger.info(f"dW is cross-attn only. {len(per_layer)} attn2 layers changed, {total:,} params")
    for band in ("res64", "res32", "res16", "res8"):
        if band in by_band:
            b = by_band[band]
            logger.info(f"  {band:<6} {len(b['layers']):>2} layers  {b['n_params']:>11,} params "
                        f"({100*b['n_params']/total:>4.1f}%)  sum|dW|2={b['l2']:.3f}")

    band_layers = {b: {l for l, e in per_layer.items() if e["band"] == b} for b in BANDS}
    res16 = band_layers["res16"]
    all_layers = set(per_layer)
    budget = by_band["res16"]["n_params"]

    # a layer-level parameter-matched control cannot exist -- say so, loudly, instead of faking one
    others = total - budget
    logger.info(f"res16 holds {budget:,} of {total:,} dW params ({100*budget/total:.1f}%); every "
                f"other attn2 layer COMBINED holds {others:,}. A parameter-matched random LAYER "
                f"control is impossible -> using the random-ENTRY control instead.")

    specs = [
        (f"{args.prefix}_sd_p_res16", res16, "add",
         "sufficiency: raw SD + ODACE dW on the 16x16 band only"),
        (f"{args.prefix}_odace_m_res16", res16, "restore",
         "necessity: ODACE with the 16x16 band restored to raw SD"),
        (f"{args.prefix}_sd_p_res32", band_layers["res32"], "add",
         "band sweep: 32x32 (activation-patch sufficiency was 0.22)"),
        (f"{args.prefix}_sd_p_res64", band_layers["res64"], "add",
         "band sweep: 64x64 (activation-patch sufficiency was 0.11)"),
        (f"{args.prefix}_sd_p_res8", band_layers["res8"], "add",
         "band sweep: 8x8 mid-block (activation-patch sufficiency was 0.10)"),
        (f"{args.prefix}_sd_p_randel", None, "random_entries",
         "parameter control: a random dW-ENTRY subset of res16's size, spread over all layers"),
        (f"{args.prefix}_sd_p_all", all_layers, "add",
         "sanity: must equal the ODACE checkpoint exactly"),
    ]

    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = want - {name.split(f"{args.prefix}_", 1)[1] for name, *_ in specs}
        if unknown:
            raise SystemExit(f"--only: unknown spec(s) {sorted(unknown)}")
        specs = [s for s in specs
                 if s[0].split(f"{args.prefix}_", 1)[1] in want or s[0].endswith("_sd_p_all")]
        logger.info(f"--only {sorted(want)} -> building {[s[0] for s in specs]}")

    manifest = {
        "_doc": "E1 ODACE weight transplant. add = raw SD + ODACE dW on the listed attn2 layers. "
                "restore = ODACE with those layers reset to raw SD. Register these under "
                "eval/xeval.py REGISTRY kind='odace' with unet_dir=<dir>/final.",
        "base": BASE_ID, "odace": str(args.odace), "seed": args.seed,
        # provenance: pin exactly which code + checkpoint produced these transplant models
        "provenance": {
            "code_sha256": sha256_of(Path(__file__)),
            "odace_ckpt_sha256": sha256_of(args.odace / "diffusion_pytorch_model.safetensors"),
            "odace_config_sha256": sha256_of(args.odace / "config.json"),
            "torch": torch.__version__, "diffusers": diffusers.__version__,
        },
        "dw_total_params": total,
        "bands": {b: {"n_layers": len(v["layers"]), "n_params": v["n_params"]}
                  for b, v in by_band.items()},
        "per_layer": per_layer, "models": {},
    }

    for name, names, mode, doc in specs:
        if mode == "random_entries":
            state, applied = build_random_entries(sd_base, sd_odace, dw, budget, args.seed)
            names = all_layers
        else:
            state, applied = build(sd_base, sd_odace, dw, names, mode)
        frac = 100 * applied / total
        manifest["models"][name] = {
            "mode": mode, "doc": doc, "n_layers": len(names),
            "layers": sorted(names), "params_changed_vs_sd": applied,
            "frac_of_odace_dw": round(frac, 2),
            "unet_dir": f"models/odace/outputs/{name}/final",
        }
        logger.info(f"{name:<26} {mode:<14} {len(names):>2} layers  {applied:>11,} params "
                    f"({frac:>5.1f}% of ODACE dW)  -- {doc}")
        if args.dry_run:
            continue
        out_dir = OUT_ROOT / name / "final"
        model = UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet")
        model.load_state_dict(state)
        model.save_pretrained(out_dir)
        logger.info(f"  wrote {out_dir}")

        if name.endswith("_sd_p_all"):   # the sanity check must be exact, not approximate
            chk = UNet2DConditionModel.from_pretrained(out_dir).state_dict()
            bad = [k for k in chk if not torch.equal(chk[k], sd_odace[k])]
            if bad:
                raise SystemExit(f"SANITY FAILED: SD + dW(all) != ODACE on {len(bad)} tensors")
            logger.info("  sanity ok: SD + dW(all) is bit-identical to the ODACE checkpoint")

    if not args.dry_run:
        # prefix-derived so a second concept's build (e.g. --prefix xplantv for the violence
        # ODACE) cannot overwrite the nudity manifest
        p = OUT_ROOT / f"{args.prefix}_manifest.json"
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"wrote {p}")


if __name__ == "__main__":
    main()
