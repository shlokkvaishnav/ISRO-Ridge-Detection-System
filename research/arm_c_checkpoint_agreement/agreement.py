"""Do the two Arm C checkpoints catch the same benchmark vertices, beyond within-tile chance?

Pre-registered in issue #10 (research/arm_c_checkpoint_agreement/SPEC.md). Reads only
committed JSON: no tiles, masks or checkpoints are loaded. Two steps:

1. ``python research/arm_c_checkpoint_agreement/agreement.py``
   Validation gate and power check. Uses each checkpoint's per-tile hit counts
   (the marginals) only; it never looks at which vertices both checkpoints hit.
   - Gate: per-tile sums of C and C2 in results/three_way_vertex_map/vertices.json
     must equal the per-tile hits in results/arm_c/eval.json and
     results/arm_c_early_stopping/eval_best.json (and per-tile vertex counts must
     equal the eval files' truth counts).
   - Informative tiles: 0 < a_t < n_t and 0 < b_t < n_t for both checkpoints.
   - Power: smallest achievable one-sided p = P(S >= S_max) under the null, with
     S_max = sum over informative tiles of min(a_t, b_t).
   Writes results/arm_c_checkpoint_agreement/gate.json.

2. ``python research/arm_c_checkpoint_agreement/agreement.py --analyze``
   Refuses to run unless gate.json says the gate passed and the smallest
   achievable p is <= 0.05. Computes the pre-registered deciding metrics
   (overlap S vs the exact within-tile hypergeometric null, and the normalized
   agreement index), the pre-registered secondary adjacency (circular-shift)
   null by exact enumeration, Monte Carlo cross-checks of both, and the
   descriptive extras (2x2 tables, disagreeing vertices, leave-one-informative-
   tile-out, A/B reference contrast). Writes
   results/arm_c_checkpoint_agreement/analysis.json.

Null (pre-registered): within each informative tile t with n_t vertices, the two
hit sets are independent uniformly random subsets of sizes a_t and b_t. Their
overlap is Hypergeometric(N=n_t, K=a_t, draws=b_t); the total S is the
convolution of the per-tile pmfs.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import product
from math import comb

import numpy as np

VERTICES = "results/three_way_vertex_map/vertices.json"
EVAL = {
    "C": "results/arm_c/eval.json",
    "C2": "results/arm_c_early_stopping/eval_best.json",
}
OUT_DIR = "results/arm_c_checkpoint_agreement"
GATE_JSON = os.path.join(OUT_DIR, "gate.json")
ANALYSIS_JSON = os.path.join(OUT_DIR, "analysis.json")
BENCHMARK = [3674, 1851, 748, 5017, 4098, 3461]
ALPHA = 0.05
MC_DRAWS = 200_000
SEED = 20260927


# --------------------------------------------------------------------------
# exact null machinery
# --------------------------------------------------------------------------

def hypergeom_pmf(n: int, a: int, b: int) -> np.ndarray:
    """P(overlap = k), k = 0..min(a, b), for independent uniform subsets of sizes a, b of n."""
    kmax = min(a, b)
    denom = comb(n, b)
    return np.array([comb(a, k) * comb(n - a, b - k) / denom for k in range(kmax + 1)])


def convolve_all(pmfs) -> np.ndarray:
    out = np.array([1.0])
    for p in pmfs:
        out = np.convolve(out, p)
    return out


def null_summary(marg: dict, tiles: list[int]) -> dict:
    """Exact null distribution of S over ``tiles`` from marginals {tile: (n, a, b)}."""
    pmf = convolve_all([hypergeom_pmf(*marg[t]) for t in tiles])
    s_max = sum(min(marg[t][1], marg[t][2]) for t in tiles)
    expected = sum(marg[t][1] * marg[t][2] / marg[t][0] for t in tiles)
    return {"pmf": pmf, "S_max": s_max, "E_S": expected}


def upper_tail(pmf: np.ndarray, s: int) -> float:
    return float(pmf[s:].sum()) if s < len(pmf) else 0.0


def critical(nul: dict) -> dict:
    """Smallest S with P(S >= s) <= ALPHA, and the agreement index that S implies."""
    pmf = nul["pmf"]
    crit = next((s for s in range(len(pmf)) if upper_tail(pmf, s) <= ALPHA), None)
    return {
        "critical_S": crit,
        "p_at_critical_S": None if crit is None else upper_tail(pmf, crit),
        "p_at_critical_S_minus_1": None if crit is None else upper_tail(pmf, crit - 1),
        "index_at_critical_S": None if crit is None else index_value(crit, nul["E_S"], nul["S_max"]),
    }


def index_value(s_obs: int, e: float, s_max: int) -> float | None:
    return None if s_max == e else (s_obs - e) / (s_max - e)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_vertices() -> list[dict]:
    with open(VERTICES) as f:
        return json.load(f)["vertices"]


def tile_rows(V, t) -> list[dict]:
    """A tile's vertices in polyline order (vertex index)."""
    return sorted((v for v in V if v["tile"] == t), key=lambda v: v["vertex"])


def marginals(V, x: str, y: str) -> dict:
    """{tile: (n, hits_x, hits_y)}. Reads each column separately, never jointly."""
    out = {}
    for t in BENCHMARK:
        rows = tile_rows(V, t)
        out[t] = (len(rows), sum(bool(v[x]) for v in rows), sum(bool(v[y]) for v in rows))
    return out


def informative(marg: dict) -> list[int]:
    return [t for t in BENCHMARK if 0 < marg[t][1] < marg[t][0] and 0 < marg[t][2] < marg[t][0]]


# --------------------------------------------------------------------------
# step 1: gate + power (marginals only)
# --------------------------------------------------------------------------

def loto_power(marg: dict, inf: list[int], drop: int) -> dict:
    nul = null_summary(marg, [u for u in inf if u != drop])
    return {"E_S": nul["E_S"], "S_max": nul["S_max"],
            "min_achievable_p": upper_tail(nul["pmf"], nul["S_max"]), **critical(nul)}


def run_gate() -> dict:
    V = load_vertices()
    checks = {}
    for arm, path in EVAL.items():
        with open(path) as f:
            ref = json.load(f)
        ref_tiles = {t["id"]: {"hits": t["hits"], "truth": t["truth"]} for t in ref["per_tile"]}
        obs_tiles = {}
        for t in BENCHMARK:
            rows = tile_rows(V, t)
            obs_tiles[t] = {"hits": sum(bool(v[arm]) for v in rows), "truth": len(rows)}
        checks[arm] = {
            "reference_file": path,
            "reference_per_tile": {str(k): ref_tiles[k] for k in BENCHMARK},
            "observed_per_tile": {str(k): obs_tiles[k] for k in BENCHMARK},
            "passed": all(ref_tiles[k] == obs_tiles[k] for k in BENCHMARK),
        }

    # structural check for the adjacency null: each tile's vertices are 0..n-1 in polyline order
    order_ok = all([v["vertex"] for v in tile_rows(V, t)] == list(range(len(tile_rows(V, t))))
                   for t in BENCHMARK)
    checks["vertex_order_contiguous"] = {"passed": bool(order_ok)}
    gate_passed = all(c["passed"] for c in checks.values())

    marg = marginals(V, "C", "C2")
    inf = informative(marg)
    uninf = [t for t in BENCHMARK if t not in inf]
    nul = null_summary(marg, inf)
    min_p = upper_tail(nul["pmf"], nul["S_max"])

    def reason(t):
        n, a, b = marg[t]
        parts = []
        for name, k in (("C", a), ("C2", b)):
            if k == 0:
                parts.append(f"{name} 0/{n}")
            elif k == n:
                parts.append(f"{name} {k}/{n}")
        return ", ".join(parts)

    return {
        "note": "Computed from each checkpoint's per-tile hit counts only. No joint C x C2 "
                "quantity has been read at this step.",
        "gate": checks,
        "gate_passed": gate_passed,
        "marginals": {str(t): {"n": marg[t][0], "C": marg[t][1], "C2": marg[t][2]} for t in BENCHMARK},
        "informative_tiles": inf,
        "uninformative_tiles": {str(t): reason(t) for t in uninf},
        "n_informative_vertices": sum(marg[t][0] for t in inf),
        "power": {
            "E_S": nul["E_S"],
            "S_max": nul["S_max"],
            "null_pmf_S": [float(x) for x in nul["pmf"]],
            "min_achievable_p": min_p,
            "alpha": ALPHA,
            **critical(nul),
            "test_can_reject": min_p <= ALPHA,
        },
        "power_leave_one_informative_tile_out": {str(t): loto_power(marg, inf, t) for t in inf},
        "proceed_to_analysis": bool(gate_passed and min_p <= ALPHA),
    }


# --------------------------------------------------------------------------
# step 2: analysis
# --------------------------------------------------------------------------

def overlap_by_tile(V, x: str, y: str, tiles) -> dict:
    return {t: sum(bool(v[x]) and bool(v[y]) for v in tile_rows(V, t)) for t in tiles}


def hypergeom_test(V, x: str, y: str, tiles) -> dict:
    marg = marginals(V, x, y)
    nul = null_summary(marg, tiles)
    s_t = overlap_by_tile(V, x, y, tiles)
    s = sum(s_t.values())
    return {
        "tiles": list(tiles),
        "S_obs": s,
        "E_S": nul["E_S"],
        "S_max": nul["S_max"],
        "p_one_sided": upper_tail(nul["pmf"], s),
        "min_achievable_p": upper_tail(nul["pmf"], nul["S_max"]),
        "index": index_value(s, nul["E_S"], nul["S_max"]),
        "per_tile": {
            str(t): {
                "n": marg[t][0], x: marg[t][1], y: marg[t][2], "overlap": s_t[t],
                "E_overlap": marg[t][1] * marg[t][2] / marg[t][0],
                "max_overlap": min(marg[t][1], marg[t][2]),
            } for t in tiles
        },
    }


def hypergeom_mc(V, x: str, y: str, tiles, s_obs: int, rng) -> dict:
    """Monte Carlo cross-check: y's hit set replaced by a uniformly random subset of equal size."""
    total = np.zeros(MC_DRAWS, dtype=np.int64)
    for t in tiles:
        rows = tile_rows(V, t)
        xs = np.array([bool(v[x]) for v in rows])
        b = sum(bool(v[y]) for v in rows)
        keys = rng.random((MC_DRAWS, len(rows)))
        chosen = np.argsort(keys, axis=1)[:, :b]
        total += xs[chosen].sum(axis=1)
    return {"draws": MC_DRAWS, "p_one_sided": float((total >= s_obs).mean()), "mean_S": float(total.mean())}


def adjacency_test(V, x: str, y: str, tiles, s_obs: int, rng) -> dict:
    """Circularly shift y's hit sequence along polyline order, independently per tile.

    Exact: all prod(n_t) shift combinations enumerated (identity included).
    """
    per_tile_overlap = []  # for each tile, overlap as a function of shift k
    sizes = []
    for t in tiles:
        rows = tile_rows(V, t)
        xs = np.array([bool(v[x]) for v in rows], dtype=int)
        ys = np.array([bool(v[y]) for v in rows], dtype=int)
        per_tile_overlap.append(np.array([int((xs * np.roll(ys, k)).sum()) for k in range(len(rows))]))
        sizes.append(len(rows))
    # exact distribution by convolving each tile's (uniform over shifts) overlap distribution
    pmf = np.array([1.0])
    for ov in per_tile_overlap:
        p = np.bincount(ov, minlength=ov.max() + 1) / len(ov)
        pmf = np.convolve(pmf, p)
    n_combos = int(np.prod(sizes))
    # brute-force enumeration as an independent check of the convolution
    count_ge = sum(1 for ks in product(*[range(n) for n in sizes])
                   if sum(per_tile_overlap[i][k] for i, k in enumerate(ks)) >= s_obs)
    # Monte Carlo cross-check
    draws = 20_000
    tot = np.zeros(draws, dtype=np.int64)
    for ov, n in zip(per_tile_overlap, sizes):
        tot += ov[rng.integers(0, n, size=draws)]
    return {
        "tiles": list(tiles),
        "n_shift_combinations": n_combos,
        "p_one_sided_exact": count_ge / n_combos,
        "p_one_sided_convolution": upper_tail(pmf, s_obs),
        "E_S_under_shift_null": float(sum(ov.mean() for ov in per_tile_overlap)),
        "overlap_by_shift": {str(t): ov.tolist() for t, ov in zip(tiles, per_tile_overlap)},
        "mc_draws": draws,
        "p_one_sided_mc": float((tot >= s_obs).mean()),
    }


def two_by_two(V, x: str, y: str, tiles) -> dict:
    def cell(rows):
        return {
            "both": sum(bool(v[x]) and bool(v[y]) for v in rows),
            f"{x}_only": sum(bool(v[x]) and not bool(v[y]) for v in rows),
            f"{y}_only": sum(bool(v[y]) and not bool(v[x]) for v in rows),
            "neither": sum(not bool(v[x]) and not bool(v[y]) for v in rows),
        }
    out = {str(t): cell(tile_rows(V, t)) for t in tiles}
    out["pooled"] = cell([v for t in tiles for v in tile_rows(V, t)])
    return out


def per_tile_nulls(V, x: str, y: str, tiles) -> dict:
    """Descriptive, not pre-registered (added in review round 1): each informative tile's own
    overlap against its within-tile hypergeometric null and its circular-shift null, both
    tails, so a per-tile pattern ("nested", "minimal overlap") can be read against chance."""
    out = {}
    for t in tiles:
        rows = tile_rows(V, t)
        n = len(rows)
        xs = np.array([bool(v[x]) for v in rows], dtype=int)
        ys = np.array([bool(v[y]) for v in rows], dtype=int)
        a, b, s = int(xs.sum()), int(ys.sum()), int((xs * ys).sum())
        pmf = hypergeom_pmf(n, a, b)
        kmin = max(0, a + b - n)  # pmf is zero below kmin
        shifts = np.array([int((xs * np.roll(ys, k)).sum()) for k in range(n)])
        out[str(t)] = {
            "n": n, x: a, y: b, "overlap": s, "min_overlap": kmin, "max_overlap": min(a, b),
            "E_overlap": a * b / n,
            "hypergeom_p_upper": float(pmf[s:].sum()),
            "hypergeom_p_lower": float(pmf[: s + 1].sum()),
            "hypergeom_p_equal": float(pmf[s]),
            "adjacency_p_upper": float((shifts >= s).mean()),
            "adjacency_p_lower": float((shifts <= s).mean()),
            "adjacency_n_shifts": n,
        }
    return out


def classify(p_hyp: float, idx: float, p_adj: float, loto: dict) -> dict:
    if p_hyp < ALPHA:
        base = "a" if idx >= 0.5 else "b"
    else:
        base = "c"
    full_sig = p_hyp < ALPHA
    loto_flips = [t for t, r in loto.items() if (r["p_one_sided"] < ALPHA) != full_sig]
    nulls_disagree = (p_hyp < ALPHA) != (p_adj < ALPHA)
    return {
        "base_outcome": base,
        "d_triggered": bool(loto_flips or nulls_disagree),
        "d_reasons": {
            "loto_tiles_whose_drop_flips_p_across_alpha": loto_flips,
            "hypergeometric_and_adjacency_on_opposite_sides_of_alpha": bool(nulls_disagree),
        },
    }


def run_analysis() -> dict:
    with open(GATE_JSON) as f:
        gate = json.load(f)
    if not gate["proceed_to_analysis"]:
        raise SystemExit("gate.json: gate failed or test has no power -> outcome (e); not analysing.")

    V = load_vertices()
    rng = np.random.default_rng(SEED)
    inf = gate["informative_tiles"]

    primary = hypergeom_test(V, "C", "C2", inf)
    primary["mc_cross_check"] = hypergeom_mc(V, "C", "C2", inf, primary["S_obs"], rng)
    adjacency = adjacency_test(V, "C", "C2", inf, primary["S_obs"], rng)

    loto = {}
    for t in inf:
        rest = [u for u in inf if u != t]
        r = hypergeom_test(V, "C", "C2", rest)
        r.pop("per_tile")
        adj = adjacency_test(V, "C", "C2", rest, r["S_obs"], rng)
        r["adjacency_p_exact_descriptive"] = adj["p_one_sided_exact"]
        loto[str(t)] = r

    outcome = classify(primary["p_one_sided"], primary["index"], adjacency["p_one_sided_exact"], loto)

    disagreements = {
        str(t): {
            "C_only": [v["vertex"] for v in tile_rows(V, t) if v["C"] and not v["C2"]],
            "C2_only": [v["vertex"] for v in tile_rows(V, t) if v["C2"] and not v["C"]],
        } for t in BENCHMARK
    }

    # A/B reference contrast (descriptive, same statistic and index, tiles informative for A/B)
    ab_marg = marginals(V, "A", "B")
    ab_inf = informative(ab_marg)
    ab = hypergeom_test(V, "A", "B", ab_inf)
    ab["mc_cross_check"] = hypergeom_mc(V, "A", "B", ab_inf, ab["S_obs"], rng)
    ab_adj = adjacency_test(V, "A", "B", ab_inf, ab["S_obs"], rng)
    ab["adjacency_p_exact_descriptive"] = ab_adj["p_one_sided_exact"]
    ab["uninformative_tiles"] = [t for t in BENCHMARK if t not in ab_inf]

    per_tile_p = per_tile_nulls(V, "C", "C2", inf)

    return {
        "note": "Deciding: primary (hypergeometric S, p, index). Pre-registered secondary: "
                "adjacency (circular-shift) null. Outcome (d) uses LOTO and the adjacency p per "
                "the issue's outcome definitions. Everything else is descriptive.",
        "informative_tiles": inf,
        "primary_hypergeometric": primary,
        "secondary_adjacency": adjacency,
        "outcome": outcome,
        "descriptive": {
            "two_by_two_all_tiles": two_by_two(V, "C", "C2", BENCHMARK),
            "disagreeing_vertices_by_tile": disagreements,
            "leave_one_informative_tile_out": loto,
            "AB_reference_contrast": ab,
            "AB_two_by_two_all_tiles": two_by_two(V, "A", "B", BENCHMARK),
            "per_tile_nulls_not_preregistered": per_tile_p,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.analyze:
        res = run_analysis()
        path = ANALYSIS_JSON
    else:
        res = run_gate()
        path = GATE_JSON
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "descriptive"}, indent=1, default=str)[:6000])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
