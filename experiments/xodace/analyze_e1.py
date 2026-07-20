"""E1 analysis -- does the TRAINED ODACE route its erasure through the 16x16 band? (no GPU)

The oracle (Stage 1/2, frozen SD1.4 activation patching) ranked the cross-attn bands by how much of
the achievable nudity erasure each one recovers when a matched benign counterfactual is patched in:

    res16 1.00 | res32 0.22 | res64 0.11 | res8 0.10

If ODACE's training discovered the same pathway, then transplanting ODACE's WEIGHT change one band
at a time into raw SD should reproduce that ordering -- and the res16 transplant alone should
reproduce ODACE. This file lines the two up and reports:

  sufficiency  xplant_sd_p_res16    vs odace_benign_n1   -- does the band alone carry the method?
  necessity    xplant_odace_m_res16 vs odace_benign_n1   -- does removing it undo the method?
  band sweep   spearman(weight-transplant erasure, activation-patch sufficiency) over the 4 bands
  control      xplant_sd_p_randel -- res16's parameter budget, no structure. If it matches res16,
               the story is "enough parameters", not "the right band".

Coherence is reported next to every ASR because a low ASR bought by breaking the image is collapse,
not erasure (CLAUDE.md); ODACE's whole claim is the coherence side.

Reads : models/fcf/{fullset_all,coherence,coco5k_lpips}.json  (written by the eval scripts)
        models/odace/outputs/xplant_manifest.json             (param counts per transplant)
Writes: models/odace/experiments/xodace/outputs/e1_summary.json

Caller: eval/run_xodace_e1.sh (analyze step). Not imported by anything.

Run (no GPU):
  python models/odace/experiments/xodace/analyze_e1.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FCF = ROOT / "models" / "fcf"

ODACE = "odace_benign_n1"
RAW = "raw_v14"

# band -> (transplant key, activation-patch sufficiency from Stage 2 structure_summary.json)
BANDS = {
    "res16": ("xplant_sd_p_res16", 1.00),
    "res32": ("xplant_sd_p_res32", 0.22),
    "res64": ("xplant_sd_p_res64", 0.11),
    "res8": ("xplant_sd_p_res8", 0.10),
}
EXTRA = ["xplant_odace_m_res16", "xplant_sd_p_randel"]


def load(name: str) -> dict:
    p = FCF / name
    return json.loads(p.read_text(encoding="utf-8")).get("models", {}) if p.exists() else {}


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else 0.0


def fmt(v: float | None, n: int = 2) -> str:
    return (f"%.{n}f" % v) if isinstance(v, (int, float)) else "  -  "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "e1_summary.json")
    args = ap.parse_args()

    asr, coh, coco = load("fullset_all.json"), load("coherence.json"), load("coco5k_lpips.json")
    man = json.loads((ROOT / "models" / "odace" / "outputs" / "xplant_manifest.json")
                     .read_text(encoding="utf-8"))["models"]

    def row(k: str) -> dict:
        a, c, u = asr.get(k, {}), coh.get(k, {}), coco.get(k, {})
        return {
            "asr4": a.get("fcf4_p03_mean"), "asr8": a.get("ours8_p03_mean"),
            "ring_person": (c.get("ring_a_bell") or {}).get("person_prob"),
            "i2p_person": (c.get("i2p") or {}).get("person_prob"),
            "coco_clip": u.get("coco_clip"),
            "params": man.get(k, {}).get("params_changed_vs_sd"),
        }

    keys = [RAW, ODACE] + [v[0] for v in BANDS.values()] + EXTRA
    rows = {k: row(k) for k in keys}
    missing = [k for k in keys if rows[k]["asr4"] is None]
    if missing:
        print(f"WARNING: no fullset ASR yet for {missing}")

    print(f"{'model':<24}{'params':>12}{'asr4':>7}{'asr8':>7}{'ring_p':>9}{'CLIP':>8}")
    for k in keys:
        r = rows[k]
        p = f"{r['params']:,}" if r["params"] else "-"
        print(f"{k:<24}{p:>12}{fmt(r['asr4'],1):>7}{fmt(r['asr8'],1):>7}"
              f"{fmt(r['ring_person'],3):>9}{fmt(r['coco_clip']):>8}")

    o, raw = rows[ODACE], rows[RAW]
    res16 = rows["xplant_sd_p_res16"]
    restore = rows["xplant_odace_m_res16"]
    randel = rows["xplant_sd_p_randel"]

    def recovered(r: dict) -> float | None:
        """fraction of ODACE's erasure (from raw SD) that this transplant reproduces."""
        if r["asr4"] is None or raw["asr4"] is None or o["asr4"] is None:
            return None
        span = raw["asr4"] - o["asr4"]
        return round((raw["asr4"] - r["asr4"]) / span, 4) if span else None

    verdict = {}
    if res16["asr4"] is not None:
        verdict["sufficiency"] = {
            "recovered_frac_of_odace_erasure": recovered(res16),
            "asr4": res16["asr4"], "odace_asr4": o["asr4"],
            "ring_person": res16["ring_person"], "odace_ring_person": o["ring_person"],
            "note": "SD + dW(res16) only. ~1.0 with coherence intact => the band carries ODACE.",
        }
    if restore["asr4"] is not None:
        verdict["necessity"] = {
            "asr4": restore["asr4"], "odace_asr4": o["asr4"], "raw_asr4": raw["asr4"],
            "erasure_lost": round(restore["asr4"] - o["asr4"], 2) if o["asr4"] is not None else None,
            "ring_person": restore["ring_person"],
            "note": "ODACE with res16 reset to raw SD. ASR climbing back toward raw => band required.",
        }
    if randel["asr4"] is not None and res16["asr4"] is not None:
        verdict["parameter_control"] = {
            "randel_asr4": randel["asr4"], "res16_asr4": res16["asr4"],
            "randel_params": randel["params"], "res16_params": res16["params"],
            "structure_beats_budget": bool(res16["asr4"] < randel["asr4"]),
            "note": "same parameter budget, no structure. res16 lower => location, not budget.",
        }

    bands, xs, ys = {}, [], []
    for b, (key, patch_suff) in BANDS.items():
        r = rows[key]
        bands[b] = {"transplant": key, "params": r["params"], "asr4": r["asr4"],
                    "ring_person": r["ring_person"], "coco_clip": r["coco_clip"],
                    "weight_recovered": recovered(r), "patch_sufficiency": patch_suff}
        if r["asr4"] is not None:
            xs.append(patch_suff)
            ys.append(recovered(r))
    rho = spearman(xs, ys) if len(xs) >= 3 else None

    print("\nband sweep -- oracle (activation patch) vs trained model (weight transplant):")
    print(f"{'band':<8}{'params':>12}{'patch_suff':>12}{'weight_rec':>12}{'asr4':>7}{'ring_p':>9}")
    for b, v in bands.items():
        p = f"{v['params']:,}" if v["params"] else "-"
        print(f"{b:<8}{p:>12}{fmt(v['patch_sufficiency']):>12}{fmt(v['weight_recovered']):>12}"
              f"{fmt(v['asr4'],1):>7}{fmt(v['ring_person'],3):>9}")
    if rho is not None:
        print(f"\nspearman(patch sufficiency, weight-transplant erasure) = {rho}  "
              f"(the oracle circuit and the trained edit agree if this is high)")

    for name, v in verdict.items():
        print(f"\n{name}: {json.dumps(v, indent=1)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_doc": "E1 ODACE weight transplant. weight_recovered = (asr4_raw - asr4_model) / "
                "(asr4_raw - asr4_odace), i.e. the fraction of ODACE's erasure the transplant "
                "reproduces from raw SD. patch_sufficiency = the Stage-2 activation-patch result "
                "for the same band. High spearman => the trained edit uses the oracle's circuit.",
        "odace_key": ODACE, "models": rows, "bands": bands,
        "spearman_weight_vs_patch": rho, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
