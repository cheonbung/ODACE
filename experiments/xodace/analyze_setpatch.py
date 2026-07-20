"""X-ODACE Stage 2A -- joint component sets on the HELD-OUT split (no GPU).

Stage 1 found the circuit on the discovery pairs. This scores it on 20 pairs it has never
seen, which is the only way Q2/Q3 can be honest:

  Q2 sufficiency : does patching ONLY the top-k cells reproduce the full ALL-patch erasure?
                   reported as the fraction of the achievable unsafe4 drop that top-k recovers
                   (ALL = every cross-attn signal replaced = the benign image = the ceiling).
  Q3 random      : is top-k better than the same NUMBER of random cells? 10 random draws per
                   size give a permutation-style p = fraction of draws that match or beat it.
  necessity      : complement16 / all_minus_topK -- patch everything EXCEPT the circuit. If the
                   erasure survives without the circuit, the circuit was not necessary.
  insertion curve: top4 -> top8 -> top16 -> top24 -> top32, unsafe4 remaining at each size.

Every number is paired with the person/scene axes, because an unsafe4 drop bought by breaking
the image is collapse, not erasure (CLAUDE.md).

Reads : outputs/stage2a.jsonl (run_pilot.py --mode setpatch --sanity, so clean + ALL are in it)
        outputs/sets_stage2.json (set name -> [[layer, bin], ...], written by analyze_pilot.py)
Writes: outputs/stage2a_summary.json

Caller: eval/run_xodace_stage2a.sh (analyze step). Not imported by anything.

Run (no GPU):
  python models/odace/experiments/xodace/analyze_setpatch.py --scan outputs/stage2a.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scorer import deltas                      # noqa: E402

DEFAULT_SCAN = HERE / "outputs" / "stage2a.jsonl"
DEFAULT_SETS = HERE / "outputs" / "sets_stage2.json"
DEFAULT_SUMMARY = HERE / "outputs" / "stage2a_summary.json"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def per_set(rows: list[dict]) -> tuple[dict, dict]:
    """set name -> {(pair, seed): raw deltas + absolute scores}, plus the clean baselines."""
    base = {(r["pair"], r["seed"]): r for r in rows if r["comp"] == "clean"}
    out: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r["comp"] in ("clean", "benign"):
            continue
        key = (r["pair"], r["seed"])
        if key in base:
            out[r["comp"]][key] = {**deltas(base[key], r), "unsafe4": r["unsafe4"],
                                   "person": r["person"], "scene": r["scene"]}
    return out, base


def agg(d: dict, field: str) -> float:
    return round(statistics.mean(v[field] for v in d.values()), 4) if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    ap.add_argument("--sets", type=Path, default=DEFAULT_SETS)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()

    rows = load(args.scan)
    sets_def = json.loads(args.sets.read_text(encoding="utf-8"))
    data, base = per_set(rows)
    if "ALL" not in data:
        raise SystemExit("no ALL rows -- rerun the setpatch with --sanity (ALL is the ceiling)")

    n_pairs = len({k[0] for k in base})
    clean_u4 = round(statistics.mean(r["unsafe4"] for r in base.values()), 4)
    ceiling = agg(data["ALL"], "d_unsafe4")          # the full-benign drop = 100% erasure
    print(f"held-out pairs={n_pairs} | clean unsafe4={clean_u4} | "
          f"ALL-patch drop (ceiling)={ceiling}")

    table = {}
    for name, d in data.items():
        n_comp = 80 if name == "ALL" else len(sets_def.get(name, []))
        table[name] = {
            "n_components": n_comp,
            "d_unsafe4": agg(d, "d_unsafe4"),
            "unsafe4_after": agg(d, "unsafe4"),
            "sufficiency": round(agg(d, "d_unsafe4") / ceiling, 4) if ceiling else 0.0,
            "d_person": agg(d, "d_person"),
            "d_scene": agg(d, "d_scene"),
        }

    print("\nset             n   d_unsafe4  u4_after  suff.   d_person  d_scene")
    for name in sorted(table, key=lambda n: -table[n]["d_unsafe4"]):
        if name.startswith("rand") and not name.endswith("_0"):
            continue                                  # one exemplar; the stats come below
        t = table[name]
        print(f"{name:<14}{t['n_components']:>3}   {t['d_unsafe4']:>8.3f}  "
              f"{t['unsafe4_after']:>7.3f}  {t['sufficiency']:>5.2f}   "
              f"{t['d_person']:>7.3f}  {t['d_scene']:>7.3f}")

    # ---- Q3: top-k vs the random control at matched size ----
    q3 = {}
    for k in (8, 16):
        rands = [table[n]["d_unsafe4"] for n in table if n.startswith(f"rand{k}_")]
        top = table.get(f"top{k}", {}).get("d_unsafe4")
        if not rands or top is None:
            continue
        mu, sd = statistics.mean(rands), statistics.pstdev(rands)
        beat = sum(1 for r in rands if r >= top)
        q3[f"top{k}"] = {
            "top_d_unsafe4": top, "random_mean": round(mu, 4), "random_std": round(sd, 4),
            "random_max": round(max(rands), 4), "n_random": len(rands),
            "z": round((top - mu) / sd, 2) if sd else None,
            "p_perm": round((beat + 1) / (len(rands) + 1), 4),   # add-one permutation p
        }
        print(f"\nQ3 random control top{k}: top={top:.3f} vs random {mu:.3f}+-{sd:.3f} "
              f"(max {max(rands):.3f}, n={len(rands)}) -> z={q3[f'top{k}']['z']} "
              f"p={q3[f'top{k}']['p_perm']}")

    # ---- Q2 sufficiency + necessity ----
    q2 = {n: table[n] for n in ("top4", "top8", "top16", "top24", "top32", "ALL") if n in table}
    nec = {n: table[n] for n in ("complement16", "all_minus_top8", "all_minus_top16", "bottom16")
           if n in table}
    print("\nQ2 insertion curve (sufficiency vs the ALL ceiling):")
    for n in ("top4", "top8", "top16", "top24", "top32", "ALL"):
        if n in table:
            print(f"  {n:<8} n={table[n]['n_components']:>2}  suff={table[n]['sufficiency']:.2f}  "
                  f"u4_after={table[n]['unsafe4_after']:.3f}  "
                  f"d_person={table[n]['d_person']:.3f}")
    print("necessity (patch everything EXCEPT the circuit):")
    for n, t in nec.items():
        print(f"  {n:<16} n={t['n_components']:>2}  suff={t['sufficiency']:.2f}  "
              f"u4_after={t['unsafe4_after']:.3f}")
    if "constrained16" in table:
        t = table["constrained16"]
        print(f"constrained16 (damage-budget ranking): suff={t['sufficiency']:.2f} "
              f"u4_after={t['unsafe4_after']:.3f} d_person={t['d_person']:.3f}")

    summary = {
        "_doc": "X-ODACE Stage 2A: joint component sets on the HELD-OUT split. sufficiency = "
                "fraction of the ALL-patch (full benign) unsafe4 drop recovered by the set. "
                "p_perm = add-one permutation p over the matched-size random draws.",
        "n_heldout_pairs": n_pairs, "clean_unsafe4": clean_u4, "ceiling_d_unsafe4": ceiling,
        "sets": table, "q2_insertion": q2, "q3_random_control": q3, "necessity": nec,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
