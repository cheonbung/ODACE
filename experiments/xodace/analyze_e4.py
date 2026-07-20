"""E4 analysis -- is the res16 mediation CONCEPT-GENERAL? (violence transplant, no GPU)

E1 showed the nudity ODACE (odace_benign_n1) routes its erasure through the 16x16 cross-attn band:
sufficiency 0.94, necessity ~complete, band order = the activation-patch oracle (spearman 1.0).
E4 repeats the weight-transplant construction on the VIOLENCE ODACE (odace_benign_n1_violence):

  sufficiency  xplantv_sd_p_res16     -- raw SD + violence-ODACE dW on 16x16 only
  necessity    xplantv_odace_m_res16  -- violence ODACE with 16x16 reset to raw SD
  band sweep   xplantv_sd_p_res32/64/8
  control      xplantv_sd_p_randel    -- res16's parameter budget, structure destroyed

There is NO violence activation-patch oracle (the Stage-1/2 patching was nudity-only), so the band
ordering is compared against the NUDITY oracle (res16 1.00 / res32 0.22 / res64 0.11 / res8 0.10).
Agreement therefore means something stronger than in E1: the SAME structural band mediates a
DIFFERENT concept -- the mediation is a property of where SD1.4 composes person-level semantics,
not of the nudity concept.

Caveats carried into the output:
  - violence UDA duplicates I2P in violence_q16.json (official prompts missing; long-standing),
    so asr_violence_mean effectively double-weights I2P. Consistent across models => comparable.
  - coherence = ring person_prob from coherence_violence.json; a low ASR with low person_prob is
    OOD collapse, not erasure (CLAUDE.md).

Reads : models/fcf/violence_q16.json, models/fcf/coherence_violence.json
        models/odace/outputs/xplantv_manifest.json
Writes: models/odace/experiments/xodace/outputs/e4_summary.json

Caller: eval/run_xodace_e4.sh (analyze step). Not imported by anything.

Run (no GPU):
  python models/odace/experiments/xodace/analyze_e4.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FCF = ROOT / "models" / "fcf"

ODACE = "odace_benign_n1_violence"
RAW = "raw_v14"

# band -> (transplant key, NUDITY activation-patch sufficiency -- no violence oracle exists)
BANDS = {
    "res16": ("xplantv_sd_p_res16", 1.00),
    "res32": ("xplantv_sd_p_res32", 0.22),
    "res64": ("xplantv_sd_p_res64", 0.11),
    "res8": ("xplantv_sd_p_res8", 0.10),
}
EXTRA = ["xplantv_odace_m_res16", "xplantv_sd_p_randel"]


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
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "e4_summary.json")
    args = ap.parse_args()

    q16, coh = load("violence_q16.json"), load("coherence_violence.json")
    man = json.loads((ROOT / "models" / "odace" / "outputs" / "xplantv_manifest.json")
                     .read_text(encoding="utf-8"))["models"]

    def row(k: str) -> dict:
        a, c = q16.get(k, {}), coh.get(k, {})
        att = a.get("attacks", {})
        return {
            "asr_violence": a.get("asr_violence_mean"),
            "asr_i2p": (att.get("I2P") or {}).get("asr_violence"),
            "asr_ring": (att.get("Ring-A-Bell") or {}).get("asr_violence"),
            "ring_person": (c.get("ring_a_bell") or {}).get("person_prob"),
            "i2p_person": (c.get("i2p") or {}).get("person_prob"),
            "params": man.get(k, {}).get("params_changed_vs_sd"),
        }

    keys = [RAW, ODACE] + [v[0] for v in BANDS.values()] + EXTRA
    rows = {k: row(k) for k in keys}
    missing = [k for k in keys if rows[k]["asr_violence"] is None]
    if missing:
        print(f"WARNING: no violence Q16 yet for {missing}")

    print(f"{'model':<26}{'params':>12}{'asr_v':>7}{'i2p':>7}{'ring':>7}{'ring_p':>9}")
    for k in keys:
        r = rows[k]
        p = f"{r['params']:,}" if r["params"] else "-"
        print(f"{k:<26}{p:>12}{fmt(r['asr_violence'],1):>7}{fmt(r['asr_i2p'],1):>7}"
              f"{fmt(r['asr_ring'],1):>7}{fmt(r['ring_person'],3):>9}")

    o, raw = rows[ODACE], rows[RAW]
    res16 = rows["xplantv_sd_p_res16"]
    restore = rows["xplantv_odace_m_res16"]
    randel = rows["xplantv_sd_p_randel"]

    def recovered(r: dict) -> float | None:
        """fraction of the violence ODACE's erasure (from raw SD) that this transplant reproduces."""
        if r["asr_violence"] is None or raw["asr_violence"] is None or o["asr_violence"] is None:
            return None
        span = raw["asr_violence"] - o["asr_violence"]
        return round((raw["asr_violence"] - r["asr_violence"]) / span, 4) if span else None

    verdict = {}
    if res16["asr_violence"] is not None:
        verdict["sufficiency"] = {
            "recovered_frac_of_odace_erasure": recovered(res16),
            "asr_violence": res16["asr_violence"], "odace_asr_violence": o["asr_violence"],
            "ring_person": res16["ring_person"], "odace_ring_person": o["ring_person"],
            "note": "SD + violence dW(res16) only. High + coherent => the band carries the "
                    "violence erasure too.",
        }
    if restore["asr_violence"] is not None:
        verdict["necessity"] = {
            "asr_violence": restore["asr_violence"], "odace_asr_violence": o["asr_violence"],
            "raw_asr_violence": raw["asr_violence"],
            "erasure_lost": (round(restore["asr_violence"] - o["asr_violence"], 2)
                             if o["asr_violence"] is not None else None),
            "ring_person": restore["ring_person"],
            "note": "violence ODACE with res16 reset to raw SD. ASR back toward raw => required.",
        }
    if randel["asr_violence"] is not None and res16["asr_violence"] is not None:
        verdict["parameter_control"] = {
            "randel_asr_violence": randel["asr_violence"],
            "res16_asr_violence": res16["asr_violence"],
            "randel_params": randel["params"], "res16_params": res16["params"],
            "structure_beats_budget": bool(res16["asr_violence"] < randel["asr_violence"]),
            "note": "same parameter budget, no structure. res16 lower => location, not budget.",
        }

    bands, xs, ys = {}, [], []
    for b, (key, nud_suff) in BANDS.items():
        r = rows[key]
        bands[b] = {"transplant": key, "params": r["params"],
                    "asr_violence": r["asr_violence"], "ring_person": r["ring_person"],
                    "weight_recovered": recovered(r), "nudity_patch_sufficiency": nud_suff}
        if r["asr_violence"] is not None:
            xs.append(nud_suff)
            ys.append(recovered(r))
    rho = spearman(xs, ys) if len(xs) >= 3 else None

    print("\nband sweep -- NUDITY oracle order vs VIOLENCE weight transplant (cross-concept):")
    print(f"{'band':<8}{'params':>12}{'nud_suff':>10}{'weight_rec':>12}{'asr_v':>7}{'ring_p':>9}")
    for b, v in bands.items():
        p = f"{v['params']:,}" if v["params"] else "-"
        print(f"{b:<8}{p:>12}{fmt(v['nudity_patch_sufficiency']):>10}"
              f"{fmt(v['weight_recovered']):>12}{fmt(v['asr_violence'],1):>7}"
              f"{fmt(v['ring_person'],3):>9}")
    if rho is not None:
        print(f"\nspearman(nudity oracle, violence weight-transplant) = {rho}  "
              f"(high => the SAME band mediates a DIFFERENT concept: structure, not concept)")

    for name, v in verdict.items():
        print(f"\n{name}: {json.dumps(v, indent=1)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_doc": "E4 violence weight transplant. weight_recovered = (asr_raw - asr_model) / "
                "(asr_raw - asr_odace) on asr_violence_mean (NB: UDA aliases I2P in "
                "violence_q16.json, consistently across models). nudity_patch_sufficiency = the "
                "Stage-2 NUDITY oracle; no violence oracle exists, so band-order agreement here "
                "means the mediation is concept-general.",
        "odace_key": ODACE, "models": rows, "bands": bands,
        "spearman_vs_nudity_oracle": rho, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
