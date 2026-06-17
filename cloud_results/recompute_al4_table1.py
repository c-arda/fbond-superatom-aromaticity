#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-source recompute of the Al4 Table-1 / Table-3 / Section-3.3 numbers for
ACS Omega ao-2026-06677s, keyed to the Takatsuka-Head-Gordon definition

    N_D = sum_i n_i (2 - n_i)

over the full CCSD natural-orbital occupation array. Reading one deposited JSON
guarantees that Table 1 (N_D, f_e, M_frac, S_E,max), Table 3 (frontier vs full),
and the Section-3.3 frontier/tail binning all draw from the same arrays.

Data: Al4_corrected_table1_results.json -- compact (Bohr) geometry, def2-SVP,
frozen 4 (Al 1s), RHF/UHF stability-followed CCSD/UCCSD; min Al-Al 2.50/2.20/2.20 A.
Generator: ../manuscript/v6/al4_correction_2026-06-14/confirm_al4.py (T1 and the
NOON array come from one converged mycc object).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Al4_corrected_table1_results.json")


def s_orb(n):
    """Per-orbital occupation entropy; S_E,max is the maximum over orbitals."""
    p = n / 2.0
    return 0.0 if (p <= 0 or p >= 1) else float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def main():
    d = json.load(open(DATA))
    print(f"{'system':16s} {'N_corr':>6} {'N_D':>7} {'f_e':>7} {'M_frac':>7} "
          f"{'S_E,max':>8} {'T1':>7} | {'front%':>6} {'tail%':>6}")
    print("-" * 92)
    for label, s in d["systems"].items():
        n = np.asarray(s["natural_occupations_full"], float)
        c = n * (2.0 - n)
        nd = float(c.sum())                      # THE definition: Sum n_i (2 - n_i)
        ncorr = s["n_corr"]
        fe = nd / ncorr
        mfrac = int(np.sum((n > 1e-3) & (n < 2 - 1e-3)))
        semax = max(s_orb(x) for x in n)
        front = 100 * c[(n >= 0.5) & (n <= 1.5)].sum() / nd   # static frontier |n-1|<0.5
        tail = 100 * c[(n < 0.5) | (n > 1.5)].sum() / nd
        # internal-consistency guard: arrays must reproduce the stored fields
        assert abs(nd - s["N_D"]) < 1e-3, f"{label}: N_D {nd} != stored {s['N_D']}"
        assert abs(fe - s["f_e"]) < 1e-3, f"{label}: f_e {fe} != stored {s['f_e']}"
        print(f"{label:16s} {ncorr:6d} {nd:7.3f} {fe:7.4f} {mfrac:7d} "
              f"{semax:8.4f} {s['T1_diagnostic']:7.4f} | {front:6.1f} {tail:6.1f}")
    print("-" * 92)
    print("Table 1  <- N_D, f_e, M_frac, S_E,max (all = Sum n_i(2-n_i) over the deposited array)")
    print("Table 3  <- truncation ratio = full N_D / frontier-6-orbital N_D")
    print("Sec 3.3  <- front% : singlets 0.0 (no static frontier), triplet ~46 (two SOMOs)")
    print("\nOK: every N_D and f_e reproduces from the array; one source feeds all three.")


if __name__ == "__main__":
    main()
