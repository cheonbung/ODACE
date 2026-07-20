"""X-ODACE -- screen the matched pairs, then split them into DISCOVERY and HELD-OUT.

Two things this guards against, both of which would silently invalidate the pilot:

1. A pair whose CLEAN unsafe image contains no EXPOSED nudity has nothing to erase: CE is a
   drop from that baseline, so every component would score ~0 and the scan would "disprove"
   the minimal-circuit hypothesis for a data reason. Eligibility therefore requires
   clean.unsafe4 > THRESH -- the FCF exposed-only score, NOT unsafe8, whose covered labels
   fire on a clothed body (a pair can read unsafe8 0.63 / unsafe4 0.00 and is useless here).
   The benign counterfactual must ALSO be valid: benign.unsafe4 <= THRESH, i.e. the "fully
   clothed" prompt really did produce a non-exposed donor.

2. Ranking pairs by unsafe score and scanning the top-N is selection bias: the top-k circuit
   would then be discovered AND evaluated on the same, most-extreme pairs. Instead every
   eligible pair is equally valid, so we take a STRATIFIED RANDOM split -- balanced over the
   template slots (subject/pose/setting/lighting/style) -- into a discovery set (Stage 1, where
   the circuit is found) and a disjoint held-out set (Stage 2A, where top-k vs random vs
   complement is tested). Quotas count BASE pairs only; paraphrase variants ride along with
   their base pair's split and never consume the quota.

Reads : outputs/<screen>.jsonl (comp == "clean" and comp == "benign" rows)
        + data/matched_pairs_nudity.jsonl
Writes: data/matched_pairs_nudity_screened.jsonl (same schema + "split"). The source pair file
        is never overwritten.

Run (no GPU):
  python models/odace/experiments/xodace/select_pairs.py --screen outputs/screen.jsonl \
      --n_discovery 20 --n_heldout 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_pairs import STRATA, read_pairs, write_pairs      # noqa: E402
from scorer import THRESH                                   # noqa: E402

DEFAULT_PAIRS = HERE / "data" / "matched_pairs_nudity.jsonl"
DEFAULT_SCREEN = HERE / "outputs" / "screen.jsonl"
DEFAULT_OUT = HERE / "data" / "matched_pairs_nudity_screened.jsonl"


def load_screen(screen: Path) -> tuple[dict, dict]:
    """pair id -> clean row, pair id -> benign row (a later re-run of the same pair wins)."""
    clean: dict[str, dict] = {}
    benign: dict[str, dict] = {}
    for line in screen.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("comp") == "clean":
            clean[r["pair"]] = r
        elif r.get("comp") == "benign":
            benign[r["pair"]] = r
    return clean, benign


def is_eligible(clean: dict | None, benign: dict | None, min_unsafe4: float = THRESH,
                max_benign_unsafe4: float = THRESH) -> tuple[bool, str]:
    """(eligible, reason). Reason is the failing gate, for the screening report."""
    if clean is None:
        return False, "no_clean_row"
    if clean["unsafe4"] <= min_unsafe4:
        return False, "clean_not_exposed"          # nothing to erase -> CE would be noise
    if benign is None:
        return False, "no_benign_row"
    if benign["unsafe4"] > max_benign_unsafe4:
        return False, "benign_still_exposed"       # the counterfactual is not actually benign
    return True, "ok"


def stratified_split(rows: list[dict], n_discovery: int, n_heldout: int,
                     seed: int = 1234) -> tuple[list[dict], list[dict]]:
    """Round-robin over shuffled strata groups -> both splits stay balanced over the template.

    Stratifies on `setting` (the slot that most changes what the image looks like) and breaks
    ties by `subject`, so neither split can end up all-beach or all-man by chance.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("setting", "?"), r.get("subject", "?"))].append(r)
    buckets = list(groups.values())
    for b in buckets:
        rng.shuffle(b)
    rng.shuffle(buckets)

    order: list[dict] = []
    while any(buckets):
        for b in buckets:
            if b:
                order.append(b.pop())
    return order[:n_discovery], order[n_discovery:n_discovery + n_heldout]


def _strata_report(rows: list[dict]) -> str:
    parts = []
    for k in ("subject", "setting"):
        c = Counter(r.get(k, "?") for r in rows)
        parts.append(f"{k}={dict(c.most_common(3))}")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min_unsafe4", type=float, default=THRESH,
                    help="keep a pair when its CLEAN unsafe4 (FCF exposed-only) exceeds this")
    ap.add_argument("--max_benign_unsafe4", type=float, default=THRESH,
                    help="reject a pair whose BENIGN counterfactual is itself exposed")
    ap.add_argument("--n_discovery", type=int, default=20, help="base pairs for Stage 1")
    ap.add_argument("--n_heldout", type=int, default=20, help="disjoint base pairs for Stage 2A")
    ap.add_argument("--split_seed", type=int, default=1234)
    ap.add_argument("--report_only", action="store_true", help="print stats, write nothing")
    args = ap.parse_args()

    pairs = read_pairs(args.pairs)
    clean, benign = load_screen(args.screen)
    if not clean:
        raise SystemExit(f"no clean rows in {args.screen}")

    base = [p for p in pairs if p["paraphrase_of"] is None]
    reasons: Counter = Counter()
    eligible: list[dict] = []
    for p in base:
        ok, why = is_eligible(clean.get(p["id"]), benign.get(p["id"]),
                              args.min_unsafe4, args.max_benign_unsafe4)
        reasons[why] += 1
        if ok:
            eligible.append(p)

    scanned = sum(1 for p in base if p["id"] in clean)
    print(f"screened {scanned}/{len(base)} base pairs | eligible {len(eligible)} "
          f"({100 * len(eligible) / max(1, scanned):.0f}% of scanned)")
    print("  reject reasons:", dict(reasons))
    if len(eligible) < args.n_discovery + args.n_heldout:
        print(f"  WARNING: only {len(eligible)} eligible, wanted "
              f"{args.n_discovery + args.n_heldout} -- splits will be short")

    disc, held = stratified_split(eligible, args.n_discovery, args.n_heldout, args.split_seed)
    print(f"discovery {len(disc)}: {_strata_report(disc)}")
    print(f"heldout   {len(held)}: {_strata_report(held)}")
    for p in disc[:5]:
        b = clean[p["id"]]
        print(f"  [disc] {p['id']} u4={b['unsafe4']:.2f} u8={b['unsafe8']:.2f} "
              f"person={b['person']:.2f} | {p['unsafe'][:58]}")
    if args.report_only:
        return

    split_of = {p["id"]: "discovery" for p in disc}
    split_of.update({p["id"]: "heldout" for p in held})
    out_rows: list[dict] = []
    for p in pairs:
        base_id = p["paraphrase_of"] or p["id"]        # variants inherit their base's split
        if base_id in split_of:
            out_rows.append({**p, "split": split_of[base_id]})

    write_pairs(out_rows, args.out)
    n_var = sum(1 for r in out_rows if r["paraphrase_of"] is not None)
    print(f"wrote {len(out_rows)} rows ({len(out_rows) - n_var} base + {n_var} paraphrase) "
          f"-> {args.out}   [strata keys: {', '.join(STRATA)}]")


if __name__ == "__main__":
    main()
