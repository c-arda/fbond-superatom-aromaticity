#!/usr/bin/env python3
"""
Al4(2-) basis-set series -- answers two reviewer-adjacent questions at once.

Raised in the 2026-09-01 council (Kimi, echoed by DeepSeek):

  (a) DIFFUSE FUNCTIONS. Cs3Al12(-), Al4(2-) and Au13(-) are anions, and
      def2-SVP/def2-TZVP carry no diffuse functions. N_D is built from weakly
      occupied virtual natural orbitals, which for an anion are exactly where
      diffuse functions matter. So part of the reported SVP->TZVP shift may be
      a diffuse-function artifact rather than pure zeta incompleteness. A
      referee is likely to raise this independently.

  (b) THIRD ZETA POINT. Both labs called two-point CBS extrapolation a
      distraction. Three points (SVP/TZVP/QZVP) make a trend statement
      defensible, and Al4(2-) is the cheapest system that already has two.

Design: same geometry and same frozen core as the existing def2-TZVP run
(al4_tzvp_comparison.py), so every number here is directly comparable to the
published Table 1 SVP value (N_D = 2.54) and the stored TZVP value (2.921).

  def2-SVP   nao  72   (already have: N_D 2.540)
  def2-SVPD  nao  96   <- diffuse effect at double zeta
  def2-TZVP  nao 148   (already have: N_D 2.921)
  def2-TZVPD nao 172   <- diffuse effect at triple zeta
  def2-QZVP  nao 280   <- third zeta point

Reading the result: if N_D(SVPD) lands much closer to N_D(TZVP) than N_D(SVP)
does, then a large share of the "basis-set shift" the reviewers objected to is
really a missing-diffuse-function effect, and the narrative in the revision has
to change accordingly. If SVPD ~ SVP, the shift is genuine zeta incompleteness
and the ordinal-stability argument stands as written.

Usage:  python3 al4_basis_series.py [--bases def2-svpd,def2-tzvpd,def2-qzvp]
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from pyscf import gto, scf, cc

os.environ.setdefault("PYSCF_TMPDIR", "/data/pyscf_scratch")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "al4_basis_series_results.json")

# Square-planar Al4(2-), side 2.5 A -- identical to al4_tzvp_comparison.py.
GEOM = [("Al", (0.0, 0.0, 0.0)),
        ("Al", (2.5, 0.0, 0.0)),
        ("Al", (2.5, 2.5, 0.0)),
        ("Al", (0.0, 2.5, 0.0))]
N_FROZEN = 4       # Al 1s, one per atom
CHARGE = -2

# Reference values already on disk, for the comparison table.
KNOWN = {"def2-svp": 2.540, "def2-tzvp": 2.921}


def indices(occ, n_corr):
    """Ramos-Cordoba/Salvador/Matito global indices, PCCP 18, 24015 (2016)."""
    n = np.clip(np.asarray(occ, float), 0.0, 2.0)
    nu = n / 2.0
    w = nu * (1.0 - nu)
    s = np.sqrt(np.clip(w, 0.0, None))
    I_ND = 2.0 * np.sum(w) / n_corr
    I_D = 2.0 * np.sum(s - 2.0 * w) / (2.0 * n_corr)
    N_D = float(np.sum(n * (2.0 - n)))
    return N_D, float(I_ND), float(I_D)


def run_basis(basis, max_memory):
    t0 = time.time()
    print("=" * 68, flush=True)
    print(f"  Al4(2-)  CCSD/{basis}   {datetime.now().isoformat()}")
    print("=" * 68, flush=True)

    mol = gto.M(atom=GEOM, basis=basis, charge=CHARGE, spin=0,
                unit="Angstrom", verbose=4)
    mol.max_memory = max_memory
    n_corr = mol.nelectron - 2 * N_FROZEN
    assert mol.nelectron == 54, f"nelectron {mol.nelectron} != 54"
    assert n_corr == 46, f"n_corr {n_corr} != 46 (must match stored runs)"
    print(f"  nao={mol.nao_nr()}  nelec={mol.nelectron}  N_corr={n_corr}", flush=True)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        print("  DIIS failed, retrying with Newton...", flush=True)
        mf = scf.RHF(mol).newton()
        mf.kernel()
    assert mf.converged, f"SCF failed for {basis}"
    print(f"  E_HF = {mf.e_tot:.10f}", flush=True)

    mycc = cc.CCSD(mf)
    mycc.frozen = N_FROZEN
    mycc.conv_tol = 1e-7
    mycc.conv_tol_normt = 1e-5
    mycc.max_cycle = 200
    mycc.verbose = 4
    e_corr, t1, t2 = mycc.kernel()
    t1_diag = float(np.sqrt(np.sum(t1 ** 2) / mol.nelectron))
    print(f"  E_corr = {e_corr:.8f}  T1 = {t1_diag:.6f}", flush=True)

    mycc.solve_lambda()
    dm1 = mycc.make_rdm1()
    noons = np.sort(np.linalg.eigh(dm1)[0])[::-1]
    n_out = int(np.sum((noons < -1e-6) | (noons > 2 + 1e-6)))
    N_D, I_ND, I_D = indices(noons, n_corr)
    f_e = N_D / n_corr

    print(f"\n  N_D = {N_D:.4f}   f_e = {f_e:.5f}   I_ND = {I_ND:.5f}   "
          f"I_D = {I_D:.5f}")
    print(f"  occ outside [0,2] = {n_out}   runtime = {(time.time()-t0)/60:.1f} min")
    print("=" * 68, flush=True)

    return {"basis": basis, "nao": int(mol.nao_nr()), "n_corr": n_corr,
            "E_HF": float(mf.e_tot), "E_corr": float(e_corr),
            "T1_diagnostic": t1_diag, "ccsd_converged": bool(mycc.converged),
            "rdm_type": "relaxed (Lambda-CCSD)",
            "n_unphysical_occupations": n_out,
            "N_D": N_D, "f_e": f_e, "I_ND": I_ND, "I_D": I_D,
            "natural_occupations_FULL": [float(x) for x in noons],
            "runtime_minutes": (time.time() - t0) / 60.0,
            "calculation_date": datetime.now().isoformat()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", default="def2-svpd,def2-tzvpd,def2-qzvp")
    p.add_argument("--max-memory", type=int, default=12000)
    args = p.parse_args()

    results = {}
    if os.path.exists(OUT):                     # resume-friendly
        results = json.load(open(OUT)).get("results", {})

    for basis in [b.strip() for b in args.bases.split(",") if b.strip()]:
        if basis in results:
            print(f"  {basis} already done, skipping", flush=True)
            continue
        try:
            results[basis] = run_basis(basis, args.max_memory)
        except Exception as exc:                # keep partial results on failure
            print(f"  !! {basis} FAILED: {exc}", flush=True)
            results[basis] = {"basis": basis, "error": str(exc)}
        with open(OUT, "w") as fh:
            json.dump({"system": "Al4(2-)", "geometry": "square planar, 2.5 A",
                       "n_frozen": N_FROZEN, "results": results}, fh, indent=2)

    print("\n" + "=" * 68)
    print("  SUMMARY  (known values from disk shown for comparison)")
    print("=" * 68)
    print(f"  {'basis':<12s} {'nao':>5s} {'N_D':>8s} {'f_e':>8s} {'I_ND':>8s} {'I_D':>8s}")
    for b, v in KNOWN.items():
        print(f"  {b:<12s} {'':>5s} {v:>8.3f}  (from disk)")
    for b, r in results.items():
        if "error" in r:
            print(f"  {b:<12s}  FAILED: {r['error'][:40]}")
        else:
            print(f"  {b:<12s} {r['nao']:>5d} {r['N_D']:>8.3f} {r['f_e']:>8.5f} "
                  f"{r['I_ND']:>8.5f} {r['I_D']:>8.5f}")

    svpd = results.get("def2-svpd", {}).get("N_D")
    if svpd:
        d_zeta = KNOWN["def2-tzvp"] - KNOWN["def2-svp"]
        d_diff = svpd - KNOWN["def2-svp"]
        print(f"\n  SVP->TZVP total shift : {d_zeta:+.3f}")
        print(f"  SVP->SVPD (diffuse)   : {d_diff:+.3f}"
              f"   = {100*d_diff/d_zeta:.0f}% of the total shift")
        print("  -> a large percentage here means the reviewers' 'basis-set' "
              "objection\n     is substantially a MISSING-DIFFUSE-FUNCTION "
              "objection, and the\n     revision narrative must change.")


if __name__ == "__main__":
    main()
