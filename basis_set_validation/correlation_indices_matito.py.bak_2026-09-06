#!/usr/bin/env python3
"""
f_e and N_D placed against the Ramos-Cordoba/Salvador/Matito correlation indices.

Reviewer complaints this addresses (ACS Omega ao-2026-06677s):
  R2: "what is f_e measuring? how does it compare to other diagnostics that were
       not even attempted?"  + named Matito's group as uncited literature.
  R4(4): "full-space N_D includes dynamic-correlation tails, so avoid presenting
       it as a direct measure of bonding character."

DEFINITIONS -- verified 2026-09-02 against the journal text of
Xu, Soriano-Agueda, Lopez, Ramos-Cordoba & Matito, JCTC 20, 721 (2024), eqs 1-8,
20 and 38 (page 2-3, 5). They restate the PCCP 18, 24015 (2016) indices. Sums run
over natural SPIN-orbital occupancies n_i^sigma in [0,1]; N = total number of
electrons (the paper says nothing about frozen cores).

    I_ND     = 1/2 sum_{i,s} n(1-n)                    extensive, range [0, N/2]   (2)
             = 1/4 sum_i n_i(2-n_i)                    closed shell                (3)
    I_ND-bar = (1/N) sum_{i,s} n(1-n) = (1/2N) sum_i n_i(2-n_i)   size-intensive  (4)
    I_D      = 1/4 sum_{i,s} sqrt(n(1-n)) - 1/2 sum_{i,s} n(1-n)                   (1)
    I_D-bar  = (2/N) I_D                               their suggested intensive form
    I_ND^max = max_i n_i^s(1-n_i^s) = 1/4 max_i n_i(2-n_i)                         (8)
    I_ND-bar ~ (4/N)(1 - c0^2)                        CISD, non-highly correlated (20)
    I_ND^max ~ D2^2 (2 - 4 D2^2)                      small-D2 limit               (38)
    Table 1 (CCSD): I_ND^max >= 0.024 <-> D2 >= 0.15, the usual MR threshold.

Matito's 2017 RDMFT-school slides (slide 22) carry the same intensive forms:
I_ND = (1/N) sum n(1-n), I_D = (1/2N) sum [sqrt(n(1-n)) - 2 n(1-n)]. Consistent
with the journal version once the bar is understood.

RESULTS (algebraic, checked numerically below):

    N_D  = 4 * I_ND                    exactly, no convention involved.
    f_e  = N_D / N_corr
         = 2 * I_ND-bar * (N / N_corr)

so f_e coincides with Matito's intensive index ONLY under the convention
N == N_corr. With N = all electrons, as written in the paper, the two differ by
N/N_corr, which is fixed by the frozen-core choice and is 1.14-1.67 across the
five systems here (B12 is the outlier: 24 of 60 electrons frozen). The
PROPORTIONALITY is exact; the constant is a normalisation choice and must be
stated as such in the manuscript. The earlier note in this file ("f_e = 2 I_ND
exactly") was true of this script's N_corr normalisation, not of the paper.

WHAT THIS DOES AND DOES NOT SHOW
  * f_e is not an ad hoc ratio. It is Takatsuka's N_D per correlated electron,
    and N_D is four times Matito's nondynamic index, which the authors themselves
    identify with "the deviation from idempotency of the first-order reduced
    density matrix". The dynamic part is a different functional (I_D, with the
    square root). R4's objection targets a quantity the paper never used.
  * I_ND is NOT independent validation of f_e: same occupations, same
    functional. It is an identification, not a second measurement. Only I_D,
    I_ND^max and the c0 / D2 proxies carry new content.
  * I_D is the less basis-converged index. Cs3Al12(-), N_corr = 160 at both bases:
        def2-SVP  (nao 288):  N_D 7.101   I_ND-bar 0.02219   I_D-bar 0.07539
        def2-TZVP (nao 531):  N_D 8.786   I_ND-bar 0.02746   I_D-bar 0.11168
        shift:                   +23.7%           +23.7%           +48.1%
    (I_ND-bar and I_D-bar here with N = N_corr, as in the earlier version.)
    Note Xu et al. report I_ND-bar as having "small sensitivity to the basis set"
    on small molecules; Cs3Al12(-) is a counterexample for anionic superatoms.
  * The c0 proxy (eq 20) gives c0^2 ~ 1 - N_D/4, which goes NEGATIVE for every
    cluster here with N_D > 4. That is not a bug: the relation holds for CISD on
    small, non-highly-correlated molecules, and c0 is not size-intensive (the
    paper says so). It is a one-line demonstration of why the clusters need an
    intensive measure at all.

Usage:  python3 correlation_indices_matito.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "C6H6":       "C6H6_benzene_def2tzvp_results.json",
    "Al4(2-)":    "Al4_2minus_def2tzvp_results.json",
    "B12 ico":    "B12_icosahedral_def2tzvp_results.json",
    "Cs3Al8(-)":  "Cs3Al8_minus_def2tzvp_results.json",
    "Cs3Al12(-)": "Cs3Al12_minus_def2tzvp_results.json",
}

OCC_KEYS = ("natural_occupations_FULL", "natural_occupations", "noons")

# Xu et al. 2024, Table 1, CCSD rows: I_ND^max threshold equivalent to D2 = 0.15.
I_ND_MAX_MR_THRESHOLD = 0.024


def indices(occ_spatial, n_total, n_corr):
    """All indices from spatial occupations in [0,2].

    Returns a dict. Extensive quantities carry no suffix; '_bar' is Matito's
    size-intensive form with N = n_total; '_bar_corr' the same with N = n_corr
    (the convention under which f_e = 2 * I_ND_bar_corr exactly).
    """
    n = np.clip(np.asarray(occ_spatial, dtype=float), 0.0, 2.0)
    nu = n / 2.0
    w = nu * (1.0 - nu)                     # spin-orbital n(1-n), one per spin
    s = np.sqrt(np.clip(w, 0.0, None))
    sum_w = 2.0 * np.sum(w)                 # two spin-orbitals per spatial NO
    sum_s = 2.0 * np.sum(s)

    N_D = float(np.sum(n * (2.0 - n)))
    I_ND = 0.5 * sum_w                                       # eq 2  (= N_D/4)
    I_D = 0.25 * sum_s - 0.5 * sum_w                         # eq 1
    I_ND_max = float(np.max(w))                              # eq 8

    out: dict = {
        "N_D": N_D,
        "I_ND": float(I_ND),
        "I_D": float(I_D),
        "I_ND_max": I_ND_max,
        "I_ND_bar": float(I_ND / n_total * 2.0),             # eq 4: (1/N) sum n(1-n)
        "I_D_bar": float(2.0 * I_D / n_total),               # (2/N) I_D
        "I_ND_bar_corr": float(I_ND / n_corr * 2.0),
        "I_D_bar_corr": float(2.0 * I_D / n_corr),
    }
    out["I_T_bar"] = out["I_ND_bar"] + out["I_D_bar"]
    # eq 20 inverted: c0^2 ~ 1 - N * I_ND_bar / 4 = 1 - I_ND
    out["c0_sq_est"] = float(1.0 - I_ND)
    # eq 38 inverted (small root of 4x^2 - 2x + I = 0, x = D2^2), if real.
    # CAUTION: eq 38 is the D2 -> 0 asymptotic limit and does NOT reproduce the
    # authors' own empirical calibration in their Table 1 (I_ND^max = 0.024 <->
    # D2 = 0.15 at CCSD; eq 38 maps 0.024 to D2 = 0.111). Use the tabulated
    # THRESHOLD for MR classification; treat this column as indicative only and
    # do not quote it as a computed D2.
    disc = 1.0 - 4.0 * I_ND_max
    out["D2_asymptotic_est"] = (float(np.sqrt((1.0 - np.sqrt(disc)) / 4.0))
                                if disc >= 0 else None)
    return out


def main():
    out = {}
    hdr = (f"{'system':<11s} {'N':>4s} {'Ncorr':>5s} {'N_D':>7s} {'f_e':>7s} "
           f"{'4I_ND':>7s} {'I_NDbar':>8s} {'I_Dbar':>8s} {'f_e/I_NDbar':>11s} "
           f"{'2N/Ncorr':>8s} {'I_NDmax':>8s} {'D2~eq38':>8s} {'c0^2est':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, fn in FILES.items():
        d = json.load(open(os.path.join(HERE, fn)))
        key = next((k for k in OCC_KEYS if k in d), None)
        if key is None:
            print(f"{name:<11s}  no occupation array found -- skipped")
            continue
        n_corr = d.get("n_corr") or d.get("N_corr")
        n_total = d.get("n_electrons") or d.get("nelectron")
        if not n_total:
            print(f"{name:<11s}  no total electron count in file -- skipped")
            continue
        r = indices(d[key], n_total, n_corr)
        f_e = d["f_e"]
        r.update({"f_e": f_e, "n_electrons": n_total, "n_corr": n_corr,
                  "basis": d.get("basis"),
                  "f_e_over_I_ND_bar": f_e / r["I_ND_bar"],
                  "f_e_over_I_ND_bar_corr": f_e / r["I_ND_bar_corr"],
                  "MR_by_I_ND_max": r["I_ND_max"] >= I_ND_MAX_MR_THRESHOLD})
        d2 = f"{r['D2_asymptotic_est']:.3f}" if r["D2_asymptotic_est"] is not None else "  n/a"
        print(f"{name:<11s} {n_total:>4d} {n_corr:>5d} {r['N_D']:>7.3f} {f_e:>7.4f} "
              f"{4*r['I_ND']:>7.3f} {r['I_ND_bar']:>8.5f} {r['I_D_bar']:>8.5f} "
              f"{r['f_e_over_I_ND_bar']:>11.4f} {2*n_total/n_corr:>8.4f} "
              f"{r['I_ND_max']:>8.4f} {d2:>8s} {r['c0_sq_est']:>8.3f}")
        out[name] = r

    nd_eq = all(abs(v["N_D"] - 4 * v["I_ND"]) < 1e-9 for v in out.values())
    # 1e-4 not 1e-9: f_e is read from the stored JSON, so any rounding or manual
    # edit in that file shows up here. Cs3Al8(-) reads 1.999985 (7.5e-6 relative),
    # a stored-value artifact (its n_fractional_orbitals was hand-corrected
    # 132->131), not a physics discrepancy.
    corr_eq = all(abs(v["f_e_over_I_ND_bar_corr"] - 2.0) < 1e-4 for v in out.values())
    ratio_eq = all(abs(v["f_e_over_I_ND_bar"] - 2.0 * v["n_electrons"] / v["n_corr"])
                   < 1e-4 for v in out.values())
    ratios = [v["f_e_over_I_ND_bar"] for v in out.values()]
    print(f"\n  N_D == 4 * I_ND (extensive, exact)               : {nd_eq}")
    print(f"  f_e == 2 * I_ND-bar   with N = N_corr             : {corr_eq}")
    print(f"  f_e == 2 * I_ND-bar * N/N_corr  with N = N_total  : {ratio_eq}")
    print(f"  f_e / I_ND-bar spans {min(ratios):.2f} .. {max(ratios):.2f} "
          f"(paper's N = all electrons) -- NOT a constant 2.")
    print(f"  I_ND^max >= {I_ND_MAX_MR_THRESHOLD} (Xu 2024 Table 1, D2 = 0.15 CCSD) "
          f"flags MR: "
          + ", ".join(f"{k}={'yes' if v['MR_by_I_ND_max'] else 'no'}" for k, v in out.items()))
    print("  c0^2 estimate (eq 20) is negative wherever N_D > 4: the CISD c0 proxy")
    print("  is not size-intensive and cannot be used for clusters of this size.")

    with open(os.path.join(HERE, "correlation_indices_matito.json"), "w") as fh:
        json.dump({"indices": out,
                   "N_D_equals_4_I_ND": nd_eq,
                   "f_e_equals_2_I_ND_bar_with_N_corr": corr_eq,
                   "f_e_equals_2_I_ND_bar_times_N_over_N_corr": ratio_eq,
                   "I_ND_max_MR_threshold": I_ND_MAX_MR_THRESHOLD,
                   "reference": ["Ramos-Cordoba, Salvador & Matito, PCCP 18, 24015 (2016)",
                                 "Xu, Soriano-Agueda, Lopez, Ramos-Cordoba & Matito, "
                                 "JCTC 20, 721 (2024), eqs 1-8, 20, 38"]},
                  fh, indent=2)
    print("\n  wrote correlation_indices_matito.json")


if __name__ == "__main__":
    main()
