"""X-ODACE -- turn the patch scan into the five go/no-go answers (no GPU).

The pilot only earns a paper if the explanation is CONCENTRATED, EFFECTIVE, BETTER THAN RANDOM
and STABLE. This computes exactly that from outputs/pilot_scan.jsonl:

  Q1 concentration : is causal mass carried by a few (layer, timestep) cells, or spread evenly?
                     -> top-20% mass share + Gini, reported on the MEAN matrix and as the
                        per-pair distribution (median/IQR), plus a bootstrap CI over pairs, so
                        "concentrated" cannot be an artifact of averaging.
  Q2 top-20% effect: how much EXPOSED nudity (unsafe4) does the strongest quintile remove, and
                     at what cost on the person/scene axes? Single-component upper bound here;
                     the joint answer comes from the setpatch stage, whose sets this emits.
  Q3 random control: emitted as rand_k sets (10 draws per size) for the setpatch stage.
  Q4 stability     : Spearman + top-k Jaccard of the CE map across seeds and paraphrases.
  Q5 ODACE overlap : does trained ODACE actually move the weights of the causal layers?
                     per-layer relative ||dW|| of cross-attn K/V/Q/Out vs the layer causal
                     score -- run for EVERY --odace_unet given (odace_v3 and odace_benign_n1
                     erase the same concept by different routes, so both must be checked).

CE is defined on the FCF exposed-only axis (scorer.PRIMARY = unsafe4). gamma/beta are swept so
the reported circuit cannot be an artifact of one arbitrary penalty weighting, and a separate
CONSTRAINED ranking (max unsafe4 drop subject to person drop < 0.05 and scene drop < 0.02)
gives a weighting-free answer to the same question.

Writes outputs/pilot_summary.json and outputs/sets_stage2.json (component sets for
`run_pilot.py --mode setpatch`: insertion curve, deletion curve, random control, complement).

Run (no GPU; --odace_unet loads UNets on CPU):
  python models/odace/experiments/xodace/analyze_pilot.py --scan outputs/pilot_scan.jsonl \
      --odace_unet models/odace/outputs/odace_v3/final models/odace/outputs/odace_benign_n1/final
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

from scorer import PRIMARY, ce_from_deltas, deltas          # noqa: E402

DEFAULT_SCAN = HERE / "outputs" / "pilot_scan.jsonl"
DEFAULT_SUMMARY = HERE / "outputs" / "pilot_summary.json"
DEFAULT_SETS = HERE / "outputs" / "sets_stage2.json"
SD_ID = "CompVis/stable-diffusion-v1-4"

# "Erase without breaking the image": the constrained ranking's damage budget.
MAX_PERSON_DROP = 0.05
MAX_SCENE_DROP = 0.02
GAMMA_BETA_GRID = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (0.0, 1.0), (1.0, 0.0)]
N_BOOTSTRAP = 500
N_RANDOM_SETS = 10


# ---------- small stats helpers (no scipy dependency) ----------

def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else None


def jaccard(a: set, b: set) -> float:
    return round(len(a & b) / len(a | b), 4) if (a or b) else 0.0


def gini(xs: list[float]) -> float:
    """0 = causal mass spread evenly over all components, 1 = all of it in one component."""
    vals = sorted(max(0.0, x) for x in xs)
    n, total = len(vals), sum(vals)
    if not n or total <= 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return round((2 * cum) / (n * total) - (n + 1) / n, 4)


def quartiles(xs: list[float]) -> dict | None:
    if not xs:
        return None
    s = sorted(xs)

    def q(p: float) -> float:
        i = p * (len(s) - 1)
        lo = int(i)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {"median": round(q(0.5), 4), "q1": round(q(0.25), 4), "q3": round(q(0.75), 4),
            "min": round(s[0], 4), "max": round(s[-1], 4), "n": len(s)}


# ---------- scan -> per-(pair, seed) deltas ----------

def load_scan(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def per_key_deltas(rows: list[dict]) -> tuple[dict, dict]:
    """(pair, seed) -> {(layer, bin): raw delta dict}, plus the clean baselines used."""
    base = {(r["pair"], r["seed"]): r for r in rows if r["comp"] == "clean"}
    out: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r.get("layer") is None or r["comp"] in ("clean", "benign", "ALL"):
            continue
        key = (r["pair"], r["seed"])
        if key in base:
            out[key][(r["layer"], r["bin"])] = deltas(base[key], r)
    return out, base


def mean_matrix(dmap: dict, keys: list, n_layers: int, n_bins: int, fn) -> list[list[float]]:
    mat = [[0.0] * n_bins for _ in range(n_layers)]
    for lyr in range(n_layers):
        for b in range(n_bins):
            vals = [fn(dmap[k][(lyr, b)]) for k in keys if (lyr, b) in dmap[k]]
            mat[lyr][b] = round(statistics.mean(vals), 4) if vals else 0.0
    return mat


def ce_fn(gamma: float, beta: float, primary: str = PRIMARY):
    return lambda d: ce_from_deltas(d, gamma, beta, primary)


def flat(mat: list[list[float]]) -> list[float]:
    return [v for row in mat for v in row]


def top_components(mat: list[list[float]], k: int) -> list[tuple[int, int]]:
    cells = [((lyr, b), v) for lyr, row in enumerate(mat) for b, v in enumerate(row)]
    cells.sort(key=lambda c: -c[1])
    return [c[0] for c in cells[:k]]


def mass_share(mat: list[list[float]], k: int) -> float:
    vals = flat(mat)
    tot = sum(max(0.0, v) for v in vals)
    if tot <= 0:
        return 0.0
    top = top_components(mat, k)
    return round(sum(max(0.0, mat[lyr][b]) for lyr, b in top) / tot, 4)


def print_heatmap(mat: list[list[float]]) -> None:
    n_bins = len(mat[0])
    print("layer".ljust(6) + "".join(f"bin{b}".rjust(9) for b in range(n_bins)))
    for lyr, row in enumerate(mat):
        print(f"L{lyr:<5}" + "".join(f"{v:9.3f}" for v in row))


# ---------- Q5: does trained ODACE move the causal layers? ----------

def odace_weight_delta(unet_dir: Path) -> list[float] | None:
    """Per cross-attn layer relative ||dW||_F over to_k/to_v (+to_q/to_out when trained)."""
    try:
        from diffusers import UNet2DConditionModel
    except Exception as exc:  # noqa: BLE001
        print(f"[Q5] skipped ({exc})")
        return None
    from patcher import list_cross_attn

    raw = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    trained = UNet2DConditionModel.from_pretrained(unet_dir)
    out = []
    for (_nr, m_raw), (_nt, m_tr) in zip(list_cross_attn(raw), list_cross_attn(trained)):
        num = den = 0.0
        for attr in ("to_k", "to_v", "to_q", "to_out"):
            a, b = getattr(m_raw, attr, None), getattr(m_tr, attr, None)
            if a is None or b is None:
                continue
            for pa, pb in zip(a.parameters(), b.parameters()):
                num += float((pb.detach() - pa.detach()).norm() ** 2)
                den += float(pa.detach().norm() ** 2)
        out.append(round((num ** 0.5) / (den ** 0.5), 6) if den else 0.0)
    return out


# ---------- stage-2 component sets ----------

def build_sets(mat: list[list[float]], constrained: list[tuple[int, int]], n_layers: int,
               n_bins: int, top_k: int, seed: int = 7) -> dict:
    """Insertion (topK), deletion (all-minus-topK), random control, complement, constrained."""
    rng = random.Random(seed)
    all_comps = [(lyr, b) for lyr in range(n_layers) for b in range(n_bins)]
    sets: dict[str, list] = {}
    for k in (4, 8, 16, 24, 32):                            # insertion curve (strongest first)
        sets[f"top{k}"] = [list(c) for c in top_components(mat, k)]
    for k in (8, 16):                                       # random control at matched size
        for d in range(N_RANDOM_SETS):
            sets[f"rand{k}_{d}"] = [list(c) for c in rng.sample(all_comps, k)]
    for k in (8, 16):                                       # deletion curve: everything BUT topK
        topk = set(top_components(mat, k))
        sets[f"all_minus_top{k}"] = [list(c) for c in all_comps if c not in topk]
    top_kk = set(top_components(mat, top_k))
    sets["complement16"] = [list(c) for c in all_comps if c not in top_kk]
    sets["bottom16"] = [list(c) for c in top_components([[-v for v in row] for row in mat], 16)]
    if constrained:
        sets[f"constrained{len(constrained[:top_k])}"] = [list(c) for c in constrained[:top_k]]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--sets", type=Path, default=DEFAULT_SETS)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0, help="scene-damage penalty in CE")
    ap.add_argument("--beta", type=float, default=1.0, help="coherence-damage penalty in CE")
    ap.add_argument("--primary", default=PRIMARY, choices=["unsafe4", "unsafe8"])
    ap.add_argument("--top_k", type=int, default=16, help="top-k = the 20% quintile of 80 cells")
    ap.add_argument("--odace_unet", type=Path, nargs="*", default=[],
                    help="trained ODACE UNet dirs for the Q5 dW overlap (CPU, slow)")
    args = ap.parse_args()

    rows = load_scan(args.scan)
    dmap, base = per_key_deltas(rows)
    if not dmap:
        raise SystemExit(f"no intervention rows in {args.scan}")
    keys = sorted(dmap)
    pairs = sorted({k[0] for k in keys})
    seeds = sorted({k[1] for k in keys})
    base_pairs = [p for p in pairs if not p.endswith("v")]
    n_cells = sum(len(v) for v in dmap.values())

    CE = ce_fn(args.gamma, args.beta, args.primary)
    mat = mean_matrix(dmap, keys, args.n_layers, args.n_bins, CE)
    raw = {f"mean_{f}": mean_matrix(dmap, keys, args.n_layers, args.n_bins,
                                    lambda d, f=f: d[f])
           for f in ("d_unsafe4", "d_unsafe8", "d_person", "d_scene")}
    top = top_components(mat, args.top_k)

    print(f"pairs={len(pairs)} ({len(base_pairs)} base) seeds={seeds} interventions={n_cells} "
          f"| primary={args.primary} gamma={args.gamma} beta={args.beta}")
    print(f"\n=== mean CE ({args.primary} drop - {args.gamma}*scene damage "
          f"- {args.beta}*coherence damage) ===")
    print_heatmap(mat)

    # ---- Q1 concentration: mean matrix, per-pair distribution, bootstrap CI ----
    per_pair_share, per_pair_gini = [], []
    for k in keys:
        cells = [CE(dmap[k][(lyr, b)]) for lyr in range(args.n_layers)
                 for b in range(args.n_bins) if (lyr, b) in dmap[k]]
        if len(cells) < args.top_k:
            continue
        km = [[CE(dmap[k][(lyr, b)]) if (lyr, b) in dmap[k] else 0.0
               for b in range(args.n_bins)] for lyr in range(args.n_layers)]
        per_pair_share.append(mass_share(km, args.top_k))
        per_pair_gini.append(gini(cells))

    rng = random.Random(11)
    boot = []
    for _ in range(N_BOOTSTRAP):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        boot.append(mass_share(mean_matrix(dmap, sample, args.n_layers, args.n_bins, CE),
                               args.top_k))
    boot.sort()
    ci = [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot)) - 1], 4)]

    q1 = {
        "top20_mass_share": mass_share(mat, args.top_k),
        "gini": gini(flat(mat)),
        "bootstrap_ci95_top20_mass_share": ci,
        "per_pair_top20_mass_share": quartiles(per_pair_share),
        "per_pair_gini": quartiles(per_pair_gini),
    }

    # ---- component prevalence: how often is a cell in a SINGLE pair's own top-k? ----
    prev: dict[tuple, int] = defaultdict(int)
    for k in keys:
        km = [[CE(dmap[k][(lyr, b)]) if (lyr, b) in dmap[k] else 0.0
               for b in range(args.n_bins)] for lyr in range(args.n_layers)]
        for c in top_components(km, args.top_k):
            prev[c] += 1
    prevalence = [{"component": f"L{lyr}_B{b}", "frac_pairs_in_top_k": round(prev[(lyr, b)] / len(keys), 3)}
                  for lyr, b in top]

    # ---- Q2 effect: raw 3 axes for the strongest cells ----
    q2 = {"top_single_components": [
        {"component": f"L{lyr}_B{b}",
         "ce": mat[lyr][b],
         "d_unsafe4": raw["mean_d_unsafe4"][lyr][b],
         "d_unsafe8": raw["mean_d_unsafe8"][lyr][b],
         "d_person": raw["mean_d_person"][lyr][b],
         "d_scene": raw["mean_d_scene"][lyr][b]}
        for lyr, b in top[:8]]}

    # ---- constrained ranking: max unsafe4 drop s.t. the image is NOT broken ----
    cand = [((lyr, b), raw["mean_d_unsafe4"][lyr][b])
            for lyr in range(args.n_layers) for b in range(args.n_bins)
            if raw["mean_d_person"][lyr][b] < MAX_PERSON_DROP
            and raw["mean_d_scene"][lyr][b] < MAX_SCENE_DROP]
    cand.sort(key=lambda c: -c[1])
    constrained_comps = [c[0] for c in cand]
    constrained = {
        "budget": {"max_person_drop": MAX_PERSON_DROP, "max_scene_drop": MAX_SCENE_DROP},
        "n_eligible": len(cand),
        "ranked": [f"L{lyr}_B{b}" for lyr, b in constrained_comps[:args.top_k]],
        "jaccard_vs_ce_top_k": jaccard(set(constrained_comps[:args.top_k]), set(top)),
    }

    # ---- gamma/beta sensitivity: is the circuit an artifact of the penalty weights? ----
    sensitivity = []
    for g, b_ in GAMMA_BETA_GRID:
        m_gb = mean_matrix(dmap, keys, args.n_layers, args.n_bins, ce_fn(g, b_, args.primary))
        sensitivity.append({
            "gamma": g, "beta": b_,
            "jaccard_vs_default": jaccard(set(top_components(m_gb, args.top_k)), set(top)),
            "spearman_vs_default": spearman(flat(m_gb), flat(mat)),
            "top20_mass_share": mass_share(m_gb, args.top_k),
        })

    # ---- Q4 stability across seeds and paraphrases ----
    stab: dict = {"seed_spearman": None, "seed_jaccard": None, "paraphrase_jaccard": None}
    if len(seeds) > 1:
        per_seed = {s: mean_matrix(dmap, [k for k in keys if k[1] == s], args.n_layers,
                                   args.n_bins, CE) for s in seeds}
        sp, jc = [], []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b_ = per_seed[seeds[i]], per_seed[seeds[j]]
                sp.append(spearman(flat(a), flat(b_)))
                jc.append(jaccard(set(top_components(a, args.top_k)),
                                  set(top_components(b_, args.top_k))))
        sp = [v for v in sp if v is not None]
        stab["seed_spearman"] = round(statistics.mean(sp), 4) if sp else None
        stab["seed_jaccard"] = round(statistics.mean(jc), 4) if jc else None
    if any(p.endswith("v") for p in pairs):
        m_base = mean_matrix(dmap, [k for k in keys if not k[0].endswith("v")],
                             args.n_layers, args.n_bins, CE)
        m_var = mean_matrix(dmap, [k for k in keys if k[0].endswith("v")],
                            args.n_layers, args.n_bins, CE)
        stab["paraphrase_jaccard"] = jaccard(set(top_components(m_base, args.top_k)),
                                             set(top_components(m_var, args.top_k)))
        stab["paraphrase_spearman"] = spearman(flat(m_base), flat(m_var))

    # ---- Q5 ODACE dW overlap, per trained model ----
    layer_ce = [round(sum(max(0.0, v) for v in row), 4) for row in mat]
    q5: dict = {}
    for d in args.odace_unet:
        path = d if d.is_absolute() else REPO / d
        name = path.parent.name if path.name == "final" else path.name
        dw = odace_weight_delta(path)
        if dw:
            q5[name] = {"per_layer_dW": dw, "spearman_dW_vs_layer_ce": spearman(dw, layer_ce)}

    summary = {
        "_doc": "X-ODACE pilot. CE_c = unsafe drop (FCF exposed-only 4-label, primary) "
                "- gamma*scene damage - beta*coherence damage, measured by patching the matched "
                "benign counterfactual's cross-attn signal into the unsafe run at "
                "(layer, timestep bin) on frozen SD1.4. ALL-patch rows are a plumbing check "
                "(they must reproduce the benign image) and are excluded from CE.",
        "config": {"primary": args.primary, "gamma": args.gamma, "beta": args.beta,
                   "top_k": args.top_k, "n_layers": args.n_layers, "n_bins": args.n_bins,
                   "patch_modes": sorted({r.get("patch_mode", "output") for r in rows}),
                   "splits": sorted({str(r.get("split")) for r in rows})},
        "n_pairs": len(pairs), "n_base_pairs": len(base_pairs), "seeds": seeds,
        "n_interventions": n_cells,
        "ce_matrix": mat,
        "raw_axes": raw,
        "layer_ce": layer_ce,
        "bin_ce": [round(sum(max(0.0, mat[lyr][b]) for lyr in range(args.n_layers)), 4)
                   for b in range(args.n_bins)],
        "top_components": [f"L{lyr}_B{b}" for lyr, b in top],
        "prevalence": prevalence,
        "concentration": q1, "effect": q2, "constrained": constrained,
        "sensitivity": sensitivity, "stability": stab, "odace_overlap": q5,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sets = build_sets(mat, constrained_comps, args.n_layers, args.n_bins, args.top_k)
    args.sets.write_text(json.dumps(sets, indent=2), encoding="utf-8")

    print(f"\nQ1 concentration : top-{args.top_k}/80 carries {q1['top20_mass_share']*100:.0f}% "
          f"of positive CE mass (95% CI {ci}), gini={q1['gini']}; "
          f"per-pair median share={q1['per_pair_top20_mass_share']['median'] if per_pair_share else 'n/a'}")
    for e in q2["top_single_components"][:3]:
        print(f"Q2 strongest     : {e['component']} CE={e['ce']} "
              f"d_unsafe4={e['d_unsafe4']} d_person={e['d_person']} d_scene={e['d_scene']}")
    print(f"   constrained    : {constrained['n_eligible']}/80 cells inside the damage budget, "
          f"top={constrained['ranked'][:4]} (Jaccard vs CE top-k {constrained['jaccard_vs_ce_top_k']})")
    print(f"   sensitivity    : jaccard vs default over gamma/beta grid = "
          f"{[s['jaccard_vs_default'] for s in sensitivity]}")
    print(f"Q4 stability     : {stab}")
    for name, v in q5.items():
        print(f"Q5 ODACE overlap : {name} spearman(dW, layer CE) = {v['spearman_dW_vs_layer_ce']}")
    print(f"\nwrote {args.summary}\nwrote {args.sets} (feed to run_pilot.py --mode setpatch)")


if __name__ == "__main__":
    main()
