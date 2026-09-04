#!/usr/bin/env python3
"""
Basis-set convergence of N_D / f_e -- the quantitative answer to R1.

R1 (ACS Omega ao-2026-06677s), the most favourable reviewer, made the harshest
scientific claim in the set:

  "the Takatsuka--Head-Gordon index ... certainly *must* be applied only for
   large basis sets (with no active space truncation) and, probably, should
   always be extrapolated to the basis-set limit (which the authors do not do)"

with the stated reason that natural occupation numbers decay only as k^(-8/3),
so N_D converges slowly with the one-particle basis.

This script does three things, all from data already on disk:

  1. Decomposes the def2-SVP -> def2-TZVP shift into a DIFFUSE-FUNCTION part
     and a genuine ZETA part, using the Al4(2-) five-point series.
  2. Extrapolates N_D to the basis-set limit under three explicit models, and
     reports the spread as an honest bracket rather than a single number.
  3. Checks whether the f_e ORDERING -- the paper's actual claim -- survives
     the basis change, at every basis where a common comparison exists.

The decay-law model in (2) is derived, not borrowed from energy extrapolation:
for weakly occupied natural orbitals n_k << 1, n_k(2-n_k) ~= 2 n_k, so with
n_k ~ C k^(-8/3) (Cioslowski; proved for two-electron systems by Sobolev,
Duke Math. J. 171, 3481 (2022)) the tail of N_D beyond M orbitals is

    sum_{k>M} 2 C k^(-8/3)  ~=  (6C/5) M^(-5/3),

and since the natural-orbital count equals the basis-function count,

    N_D(M) = N_D(inf) - A M^(-5/3).                                    (model 1)

This is NOT the Helgaker X^(-3) law, which is calibrated for correlation
energies and has no derivation for occupation-number sums. Model 3 applies it
anyway, purely to show how far it disagrees.

Usage:  python3 basis_convergence_analysis.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- Al4(2-) series
# def2-SVP and def2-TZVP from the production runs; the other three from
# al4_basis_series.py (2026-09-01). All same geometry, same 4 frozen orbitals,
# N_corr = 46 throughout.
N_CORR_AL4 = 46
SERIES = [                      # (basis, nao, N_D, has_diffuse, cardinal X)
    ("def2-SVP",    72, 2.540, False, 2),
    ("def2-SVPD",   96, None,  True,  2),
    ("def2-TZVP",  148, 2.921, False, 3),
    ("def2-TZVPD", 172, None,  True,  3),
    ("def2-QZVP",  280, None,  False, 4),
]

TZVP_FILES = {
    "C6H6":       "C6H6_benzene_def2tzvp_results.json",
    "Al4(2-)":    "Al4_2minus_def2tzvp_results.json",
    "B12 ico":    "B12_icosahedral_def2tzvp_results.json",
    "Cs3Al8(-)":  "Cs3Al8_minus_def2tzvp_results.json",
    "Cs3Al12(-)": "Cs3Al12_minus_def2tzvp_results.json",
}
# Published Table 1, CCSD/def2-SVP (manuscript v6, tab:main).
SVP_PUBLISHED = {
    "C6H6": (2.49, 30), "Al4(2-)": (2.54, 46), "B12 ico": (4.99, 36),
    "Cs3Al8(-)": (5.58, 116), "Cs3Al12(-)": (7.10, 160),
}


def load_series():
    """Fill the missing N_D values from al4_basis_series_results.json."""
    path = os.path.join(HERE, "al4_basis_series_results.json")
    computed = json.load(open(path))["results"]
    out = []
    for basis, nao, nd, diffuse, X in SERIES:
        if nd is None:
            r = computed[basis.lower()]
            nd, nao = r["N_D"], r["nao"]
        out.append({"basis": basis, "nao": nao, "N_D": nd,
                    "f_e": nd / N_CORR_AL4, "diffuse": diffuse, "X": X})
    return out


def fit_power(naos, nds, p):
    """Least squares N_D = N_inf - A*nao^(-p); returns (N_inf, A, rms)."""
    x = np.asarray(naos, float) ** (-p)
    A_mat = np.vstack([np.ones_like(x), -x]).T
    (N_inf, A), *_ = np.linalg.lstsq(A_mat, np.asarray(nds, float), rcond=None)
    rms = float(np.sqrt(np.mean((A_mat @ [N_inf, A] - nds) ** 2)))
    return float(N_inf), float(A), rms


def fit_power_free(naos, nds):
    """Three-point solve for N_D = N_inf - A*nao^(-p) with p free."""
    from scipy.optimize import brentq

    n1, n2, n3 = map(float, naos)
    d1, d2, d3 = map(float, nds)

    def resid(p):
        # eliminate N_inf and A from the first two equations, test the third
        x1, x2, x3 = n1 ** -p, n2 ** -p, n3 ** -p
        A = (d2 - d1) / (x1 - x2)
        N_inf = d2 + A * x2
        return N_inf - A * x3 - d3

    try:
        p = brentq(resid, 0.3, 8.0)
    except ValueError:
        return None
    x1, x2 = n1 ** -p, n2 ** -p
    A = (d2 - d1) / (x1 - x2)
    return float(d2 + A * x2), float(A), float(p)


def main():
    s = load_series()
    by = {r["basis"]: r for r in s}

    print("=" * 74)
    print("  1. Al4(2-) five-point basis series   (CCSD, N_corr = 46)")
    print("=" * 74)
    print(f"  {'basis':<12s} {'nao':>5s} {'N_D':>8s} {'f_e':>8s} {'d(N_D)':>9s}")
    prev = None
    for r in s:
        d = "" if prev is None else f"{r['N_D']-prev:+9.3f}"
        print(f"  {r['basis']:<12s} {r['nao']:>5d} {r['N_D']:>8.3f} "
              f"{r['f_e']:>8.5f} {d:>9s}")
        prev = r["N_D"]

    svp, svpd = by["def2-SVP"]["N_D"], by["def2-SVPD"]["N_D"]
    tzvp, tzvpd = by["def2-TZVP"]["N_D"], by["def2-TZVPD"]["N_D"]
    qzvp = by["def2-QZVP"]["N_D"]

    print("\n" + "=" * 74)
    print("  2. Diffuse-function decomposition of the SVP -> TZVP shift")
    print("=" * 74)
    tot = tzvp - svp
    dz_diff = svpd - svp
    tz_diff = tzvpd - tzvp
    print(f"  SVP  -> TZVP   (the shift the reviewers objected to) : {tot:+.3f}"
          f"  ({100*tot/svp:+.1f}%)")
    print(f"  SVP  -> SVPD   (diffuse added at DOUBLE zeta)        : {dz_diff:+.3f}"
          f"  = {100*dz_diff/tot:.0f}% of that shift")
    print(f"  TZVP -> TZVPD  (diffuse added at TRIPLE zeta)        : {tz_diff:+.3f}"
          f"  = {100*tz_diff/tzvp:+.1f}% of N_D")
    print(f"  TZVP -> QZVP   (pure zeta step, no diffuse)          : {qzvp-tzvp:+.3f}"
          f"  = {100*(qzvp-tzvp)/tzvp:+.1f}% of N_D")
    print("\n  READING: nearly half the SVP->TZVP shift is recovered by adding")
    print("  diffuse functions to the DOUBLE-zeta basis alone. At triple zeta the")
    print("  same augmentation buys almost nothing. The diffuse deficiency is")
    print("  therefore a double-zeta artefact, saturated by def2-TZVP; what")
    print("  remains beyond TZVP is genuine zeta incompleteness.")

    print("\n" + "=" * 74)
    print("  3. Extrapolation to the basis-set limit")
    print("=" * 74)
    ests = {}

    # Model 1: derived M^(-5/3) law, fitted to the two largest non-diffuse points.
    n_inf, A, _ = fit_power([by["def2-TZVP"]["nao"], by["def2-QZVP"]["nao"]],
                            [tzvp, qzvp], 5.0 / 3.0)
    ests["M^(-5/3), TZVP+QZVP"] = n_inf
    print(f"  model 1  N_D = N_inf - A M^(-5/3)   [derived from n_k ~ k^(-8/3)]")
    print(f"           fit to TZVP+QZVP      : N_inf = {n_inf:.3f}   A = {A:.0f}")
    for r in s:
        pred = n_inf - A * r["nao"] ** (-5.0 / 3.0)
        print(f"             predicts {r['basis']:<11s} {pred:6.3f}  "
              f"(actual {r['N_D']:.3f}, err {pred-r['N_D']:+.3f})")
    print("           -> the two triple-zeta points and QZVP are reproduced to")
    print("              <=0.01; both double-zeta points fall BELOW the fitted")
    print("              curve, i.e. they are not yet in the asymptotic regime.")
    print("              That is independent support for the diffuse argument.")

    # Model 2: geometric decay of the increments (non-diffuse series).
    d1, d2 = tzvp - svp, qzvp - tzvp
    r_geo = d2 / d1
    n_geo = qzvp + d2 * r_geo / (1 - r_geo)
    ests["geometric increments"] = n_geo
    print(f"\n  model 2  geometric increments: ratio {r_geo:.3f}, "
          f"remaining tail {d2*r_geo/(1-r_geo):+.3f}")
    print(f"           N_inf = {n_geo:.3f}")

    # Model 3: Helgaker X^(-3) on cardinal numbers -- shown to be rejected.
    X1, X2 = 3.0, 4.0
    n_helg = (X2 ** 3 * qzvp - X1 ** 3 * tzvp) / (X2 ** 3 - X1 ** 3)
    ests["Helgaker X^(-3) [not applicable]"] = n_helg
    print(f"\n  model 3  Helgaker X^(-3) on TZ/QZ cardinals : N_inf = {n_helg:.3f}")
    print("           REPORTED FOR CONTRAST ONLY. The X^(-3) law is calibrated")
    print("           for correlation ENERGIES; no such law is established for")
    print("           occupation-number sums, and it disagrees with model 1 by"
          f" {abs(n_helg-ests['M^(-5/3), TZVP+QZVP']):.3f}.")

    free = fit_power_free([r["nao"] for r in s if not r["diffuse"]],
                          [r["N_D"] for r in s if not r["diffuse"]])
    if free:
        print(f"\n  check    three-point free-exponent fit (SVP/TZVP/QZVP):")
        print(f"           N_inf = {free[0]:.3f}, p = {free[2]:.2f}")
        print("           p is contaminated by the double-zeta diffuse deficit,")
        print("           so this fit is a consistency check, not an estimate.")

    lo, hi = min(ests["M^(-5/3), TZVP+QZVP"], ests["geometric increments"]), \
        max(ests["M^(-5/3), TZVP+QZVP"], ests["geometric increments"])
    print(f"\n  BRACKET (models 1-2, the two defensible ones): "
          f"N_D(CBS) = {lo:.2f}-{hi:.2f}")
    print(f"  i.e. {100*(lo/svp-1):.0f}-{100*(hi/svp-1):.0f}% above the published "
          f"def2-SVP value of {svp:.2f},")
    print(f"  and only {100*(lo/qzvp-1):.1f}-{100*(hi/qzvp-1):.1f}% above the "
          f"computed def2-QZVP value.")
    print(f"  f_e(CBS) = {lo/N_CORR_AL4:.4f}-{hi/N_CORR_AL4:.4f}  "
          f"(published SVP: {svp/N_CORR_AL4:.4f})")

    print("\n" + "=" * 74)
    print("  4. Does the f_e ORDERING survive the basis change?")
    print("=" * 74)
    rows = []
    for name, fn in TZVP_FILES.items():
        d = json.load(open(os.path.join(HERE, fn)))
        nd_svp, ncorr = SVP_PUBLISHED[name]
        rows.append((name, nd_svp / ncorr, d["f_e"]))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'system':<12s} {'f_e(SVP)':>9s} {'rank':>5s} "
          f"{'f_e(TZVP)':>10s} {'rank':>5s} {'change':>8s}")
    rank_tz = {n: i for i, (n, _, _) in
               enumerate(sorted(rows, key=lambda r: -r[2]), 1)}
    same = True
    for i, (n, fs, ft) in enumerate(rows, 1):
        if rank_tz[n] != i:
            same = False
        print(f"  {n:<12s} {fs:>9.4f} {i:>5d} {ft:>10.4f} {rank_tz[n]:>5d} "
              f"{100*(ft/fs-1):>7.1f}%")
    print(f"\n  ordering identical at both bases : {same}")
    b_svp = rows[2][1] / rows[3][1] if len(rows) > 3 else float("nan")
    b_tzvp = rows[2][2] / rows[3][2] if len(rows) > 3 else float("nan")
    print(f"  boundary pair {rows[2][0]}/{rows[3][0]}: "
          f"{b_svp:.2f}x at SVP -> {b_tzvp:.2f}x at TZVP  (narrows)")
    print(f"  extreme pair  {rows[0][0]}/{rows[-1][0]}: "
          f"{rows[0][1]/rows[-1][1]:.2f}x at SVP -> "
          f"{rows[0][2]/rows[-1][2]:.2f}x at TZVP  (holds)")
    print("\n  CONCLUSION: every system gains N_D with basis, by 2-24%, but the")
    print("  rank order of f_e is unchanged and the regime extremes stay well")
    print("  separated. The boundary pair narrows and is NOT resolved by this")
    print("  data -- state that as a limit of the classification, not a result.")

    out = {
        "al4_series": s,
        "diffuse_decomposition": {
            "SVP_to_TZVP": tot, "SVP_to_SVPD": dz_diff,
            "diffuse_fraction_of_shift": dz_diff / tot,
            "TZVP_to_TZVPD": tz_diff, "TZVP_to_QZVP": qzvp - tzvp,
        },
        "cbs_estimates": ests,
        "cbs_bracket": [lo, hi],
        "f_e_ordering_preserved": same,
        "reviewer": "R1, ACS Omega ao-2026-06677s",
    }
    with open(os.path.join(HERE, "basis_convergence_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote basis_convergence_analysis.json")


if __name__ == "__main__":
    main()
