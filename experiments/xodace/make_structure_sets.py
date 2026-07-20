"""X-ODACE structure probe -- size-matched STRUCTURAL component sets (no GPU).

Stage 2A found that the CE-ranked circuit saturates: top4 ~ top8 ~ top16 all recover ~42% of the
achievable unsafe4 drop, while ranks 17-32 add another +0.20. Marginal single-cell CE therefore
ranked cells that are mutually REDUNDANT -- it kept re-picking the same mechanism.

Stage 1 also showed the top cells are not scattered: L4-L9 at the earliest timestep bins. Mapped
onto SD1.4's cross-attn exec order that is exactly the LOW-RESOLUTION BOTTLENECK during early
denoising. So the competing hypothesis is that the circuit is STRUCTURAL (a time band and/or a
resolution band), and CE ranking merely sampled a redundant subset of it.

This file writes the sets that test that, always at a size the ranked sets can be compared to:

  time axis   bin{0..4}_all    16 cells each  -- same size as top16 (suff 0.42)
              bin01_all        32 cells       -- same size as top32 (suff 0.73)
  depth axis  res64/32/16/8_all, down_all, up_all, bottleneck_all
  crossed     bottleneck_bin0 (6), bottleneck_bin01 (12), bottleneck_bin012 (18),
              res16_bin01 (10), mid_bin01 (2)

If bin0_all (16) beats top16, the circuit is temporal. If bottleneck_bin01 (12 cells, FEWER than
top16) beats it, the circuit is bottleneck x early-time. If neither wins, redundancy is diffuse and
"minimal causal circuit" is genuinely the wrong frame for nudity in SD1.4.

Layer index = cross-attn (attn2) EXECUTION order, verified against the live UNet:
  L0-1 down.0 (320ch, 64x64) | L2-3 down.1 (640, 32x32) | L4-5 down.2 (1280, 16x16)
  L6 mid (1280, 8x8) | L7-9 up.1 (1280, 16x16) | L10-12 up.2 (640, 32x32) | L13-15 up.3 (320, 64x64)
Bin index = denoising-time bin, 0 = earliest (highest noise).

Reads : outputs/sets_stage2.json (the Stage-2A sets, so the two tables stay comparable)
Writes: outputs/sets_structure.json (the 18 new sets)
        outputs/sets_all.json       (stage2 + structure, fed to run_pilot --sets_json)

Caller: eval/run_xodace_structure.sh. Not imported by anything.

Run (no GPU):
  python models/odace/experiments/xodace/make_structure_sets.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

N_LAYERS = 16
N_BINS = 5

# cross-attn exec index -> spatial resolution of the block it lives in
RES: dict[int, list[int]] = {
    64: [0, 1, 13, 14, 15],
    32: [2, 3, 10, 11, 12],
    16: [4, 5, 7, 8, 9],
    8: [6],
}
DOWN = [0, 1, 2, 3, 4, 5]
MID = [6]
UP = [7, 8, 9, 10, 11, 12, 13, 14, 15]
BOTTLENECK = [4, 5, 6, 7, 8, 9]          # 16x16 + 8x8: where Stage 1's top cells live


def cells(layers: list[int], bins: list[int]) -> list[list[int]]:
    return [[lyr, b] for lyr in layers for b in bins]


def build() -> dict[str, list[list[int]]]:
    all_layers = list(range(N_LAYERS))
    all_bins = list(range(N_BINS))
    sets: dict[str, list[list[int]]] = {}

    # --- time axis: one whole timestep bin, every layer. 16 cells = top16's size. ---
    for b in all_bins:
        sets[f"bin{b}_all"] = cells(all_layers, [b])
    sets["bin01_all"] = cells(all_layers, [0, 1])                    # 32 = top32's size

    # --- depth axis: one whole resolution / block group, every timestep. ---
    for res, lyrs in RES.items():
        sets[f"res{res}_all"] = cells(lyrs, all_bins)
    sets["down_all"] = cells(DOWN, all_bins)
    sets["up_all"] = cells(UP, all_bins)
    sets["bottleneck_all"] = cells(BOTTLENECK, all_bins)

    # --- crossed: the actual Stage-1 hypothesis, at sizes at or BELOW top16. ---
    sets["bottleneck_bin0"] = cells(BOTTLENECK, [0])                 # 6
    sets["bottleneck_bin01"] = cells(BOTTLENECK, [0, 1])             # 12 < top16
    sets["bottleneck_bin012"] = cells(BOTTLENECK, [0, 1, 2])         # 18
    sets["res16_bin01"] = cells(RES[16], [0, 1])                     # 10
    sets["mid_bin01"] = cells(MID, [0, 1])                           # 2 -- LocoGen's "mid only"
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", type=Path, default=OUT / "sets_stage2.json")
    ap.add_argument("--out_structure", type=Path, default=OUT / "sets_structure.json")
    ap.add_argument("--out_all", type=Path, default=OUT / "sets_all.json")
    args = ap.parse_args()

    structure = build()
    args.out_structure.write_text(json.dumps(structure, indent=2), encoding="utf-8")

    stage2 = json.loads(args.stage2.read_text(encoding="utf-8"))
    overlap = set(stage2) & set(structure)
    if overlap:
        raise SystemExit(f"name collision with the Stage-2A sets: {sorted(overlap)}")
    merged = {**stage2, **structure}
    args.out_all.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"{len(structure)} structural sets (new GPU work: "
          f"{len(structure)} sets x heldout pairs):")
    for name, comp in structure.items():
        print(f"  {name:<20} n={len(comp):>2}")
    print(f"\nwrote {args.out_structure}")
    print(f"wrote {args.out_all}  ({len(stage2)} stage2 + {len(structure)} structure "
          f"= {len(merged)} sets; run_pilot resume skips the stage2 ones)")


if __name__ == "__main__":
    main()
