#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis-set comparison: Benzene CCSD/def2-TZVP vs def2-SVP
=========================================================
Runs benzene at def2-TZVP to validate that f_e is basis-set stable.
This addresses a key reviewer concern about whether the regime
classification (f_e ~ 0.08-0.14 for small clusters) is a basis-set artifact.

Expected runtime: ~2-4 hours on multi-core CPU.
"""

import numpy as np
from pyscf import gto, scf, cc
import json
import os
from datetime import datetime
import time

# ============================================================================
# BENZENE GEOMETRY (D6h, experimental, NIST) — same as SVP calculation
# ============================================================================

BENZENE_GEOMETRY = """
C   1.3940   0.0000   0.0000
C   0.6970   1.2073   0.0000
C  -0.6970   1.2073   0.0000
C  -1.3940   0.0000   0.0000
C  -0.6970  -1.2073   0.0000
C   0.6970  -1.2073   0.0000
H   2.4810   0.0000   0.0000
H   1.2405   2.1483   0.0000
H  -1.2405   2.1483   0.0000
H  -2.4810   0.0000   0.0000
H  -1.2405  -2.1483   0.0000
H   1.2405  -2.1483   0.0000
"""


def run_benzene_ccsd(basis='def2-tzvp'):
    """Run CCSD N_D analysis for benzene at given basis set."""

    t_start = time.time()

    print("=" * 70)
    print(f"  N_D Basis-Set Comparison: Benzene (C6H6)")
    print(f"  Method: CCSD/{basis}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    mol = gto.M(
        atom=BENZENE_GEOMETRY,
        basis=basis,
        charge=0,
        spin=0,
        symmetry=True,
        unit='Angstrom',
        verbose=4,
    )

    n_atoms = mol.natm
    n_elec = mol.nelectron
    n_ao = mol.nao
    # Freeze C 1s core orbitals (6 carbons)
    n_frozen = 6
    n_corr = n_elec - 2 * n_frozen  # correlated electrons

    print(f"\nMolecule: {n_atoms} atoms, {n_elec} electrons, {n_ao} AOs")
    print(f"Frozen core: {n_frozen} orbitals ({2*n_frozen} electrons)")
    print(f"Correlated electrons: {n_corr}")
    print(f"Correlated orbitals (M): {n_ao - n_frozen}")

    # SCF
    print(f"\n{'─' * 50}")
    print("  Running RHF...")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()

    if not mf.converged:
        print("SCF failed!")
        return None

    print(f"  E(HF) = {mf.e_tot:.10f} Ha")

    # HOMO-LUMO gap
    eps = mf.mo_energy
    homo_idx = n_elec // 2 - 1
    gap = eps[homo_idx + 1] - eps[homo_idx]
    print(f"  HOMO-LUMO gap = {gap:.6f} Ha = {gap*27.2114:.2f} eV")

    # CCSD
    print(f"\n{'─' * 50}")
    print("  Running CCSD (this may take a while at TZVP)...")
    mycc = cc.CCSD(mf)
    mycc.frozen = n_frozen
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-6
    mycc.max_cycle = 200

    ecc, t1, t2 = mycc.kernel()

    if not mycc.converged:
        print("CCSD failed!")
        return None

    print(f"  E(CCSD) = {mycc.e_tot:.10f} Ha")
    print(f"  E_corr  = {mycc.e_corr:.6f} Ha = {mycc.e_corr * 1000:.3f} mHa")

    # T1 diagnostic
    t1_diag = np.sqrt(np.sum(t1**2) / n_corr)
    print(f"  T1 diagnostic = {t1_diag:.6f}")

    # Lambda equations for proper 1-RDM
    print(f"\n{'─' * 50}")
    print("  Solving Lambda-CCSD equations...")
    mycc.solve_lambda()

    # Natural orbitals from CCSD 1-RDM
    print(f"\n{'─' * 50}")
    print("  Computing natural orbitals from CCSD 1-RDM...")
    dm1 = mycc.make_rdm1()
    noons, natorbs = np.linalg.eigh(dm1)
    noons = np.sort(noons.flatten())[::-1]

    M = len(noons)  # total correlated orbitals

    print(f"\n  Top 15 natural orbital occupations:")
    for i, n in enumerate(noons[:15]):
        dev = abs(n - 2.0) if n > 1.0 else abs(n)
        marker = " <-- fractional" if dev > 0.001 else ""
        print(f"    NO {i+1:3d}: n = {n:.6f}{marker}")

    # N_D = sum_i n_i(2 - n_i) — Takatsuka-Head-Gordon index
    nd_contributions = np.array([n * (2 - n) for n in noons])
    N_D = float(np.sum(nd_contributions))

    # f_e = N_D / N_corr
    f_e = N_D / n_corr

    # Count fractional orbitals (0.001 < n < 1.999)
    n_fractional = int(np.sum((noons > 0.001) & (noons < 1.999)))
    frac_pct = n_fractional / M * 100

    # Maximum single-orbital entanglement entropy
    S_list = []
    for n in noons:
        if 0.0 < n < 2.0:
            p = n / 2.0
            if 0.0 < p < 1.0:
                S_i = -p * np.log(p) - (1 - p) * np.log(1 - p)
            else:
                S_i = 0.0
        else:
            S_i = 0.0
        S_list.append(S_i)
    S_E_max = max(S_list)

    t_elapsed = time.time() - t_start

    print(f"\n  {'═' * 50}")
    print(f"  RESULTS for benzene CCSD/{basis}:")
    print(f"  {'═' * 50}")
    print(f"  N_e             = {n_elec}")
    print(f"  N_corr          = {n_corr}")
    print(f"  M (corr. NOs)   = {M}")
    print(f"  M_frac          = {n_fractional} ({frac_pct:.0f}%)")
    print(f"  |E_corr|        = {abs(mycc.e_corr):.6f} Ha")
    print(f"  N_D             = {N_D:.4f}")
    print(f"  f_e             = {f_e:.4f}")
    print(f"  S_E,max         = {S_E_max:.6f} nats")
    print(f"  T1 diagnostic   = {t1_diag:.6f}")
    print(f"  Wall time       = {t_elapsed:.0f} s ({t_elapsed/60:.1f} min)")
    print(f"  {'═' * 50}")

    # Save results
    os.makedirs('data', exist_ok=True)
    results = {
        'system': 'C6H6_benzene',
        'method': 'CCSD',
        'basis': basis,
        'calculation_date': datetime.now().isoformat(),
        'wall_time_seconds': t_elapsed,
        'n_atoms': n_atoms,
        'n_electrons': n_elec,
        'n_frozen': n_frozen,
        'n_corr': n_corr,
        'n_ao': n_ao,
        'M_corr_orbitals': M,
        'M_fractional': n_fractional,
        'M_fractional_pct': frac_pct,
        'E_HF': float(mf.e_tot),
        'E_CCSD': float(mycc.e_tot),
        'E_corr': float(mycc.e_corr),
        'HOMO_LUMO_gap': float(gap),
        'T1_diagnostic': float(t1_diag),
        'N_D': N_D,
        'f_e': f_e,
        'S_E_max': float(S_E_max),
        'natural_occupations': [float(n) for n in noons],
        'nd_contributions': [float(c) for c in nd_contributions],
    }

    outfile = f'data/C6H6_benzene_{basis.replace("-","")}_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {outfile}")

    # ----- Comparison with def2-SVP if data exists -----
    svp_file = 'data/C6H6_benzene_results.json'
    if os.path.exists(svp_file):
        with open(svp_file) as f:
            svp = json.load(f)
        svp_nd = svp.get('fbond_B_total', svp.get('N_D', None))
        svp_ne = svp.get('n_electrons', 42)
        svp_nfrozen = svp.get('n_frozen', 6)
        svp_ncorr = svp_ne - 2 * svp_nfrozen
        svp_fe = svp_nd / svp_ncorr if svp_nd else None

        if svp_nd:
            print(f"\n  ╔══════════════════════════════════════════════════╗")
            print(f"  ║  BASIS-SET COMPARISON: def2-SVP vs {basis}      ║")
            print(f"  ╠══════════════════════════════════════════════════╣")
            print(f"  ║  Property       │ def2-SVP  │ {basis:10s}      ║")
            print(f"  ╠══════════════════════════════════════════════════╣")
            print(f"  ║  N_D            │ {svp_nd:9.4f} │ {N_D:9.4f}        ║")
            print(f"  ║  f_e            │ {svp_fe:9.4f} │ {f_e:9.4f}        ║")
            print(f"  ║  Δ(N_D)         │           │ {(N_D-svp_nd)/svp_nd*100:+.1f}%          ║")
            print(f"  ║  Δ(f_e)         │           │ {(f_e-svp_fe)/svp_fe*100:+.1f}%          ║")
            print(f"  ╚══════════════════════════════════════════════════╝")
    else:
        print(f"\n  [No SVP data at {svp_file} for comparison]")

    return results


if __name__ == '__main__':
    run_benzene_ccsd(basis='def2-tzvp')
