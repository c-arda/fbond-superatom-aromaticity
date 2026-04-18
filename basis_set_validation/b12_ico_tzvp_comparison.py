#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis-set comparison: B12 icosahedral CCSD/def2-TZVP vs def2-SVP
=================================================================
Validates whether the highest-f_e system (0.139 at SVP) remains in
the small cluster regime at TZVP.

Expected runtime: ~2-8 hours on AMD Ryzen 9 9950X (16c/32t).
"""

import numpy as np
import os

# Redirect PySCF scratch to /data (avoids /tmp tmpfs overflow)
os.environ['PYSCF_TMPDIR'] = '/data/pyscf_scratch'
os.environ['TMPDIR'] = '/data/pyscf_scratch'

from pyscf import gto, scf, cc, lib
import json
from datetime import datetime
import time

# Set PySCF max memory (MB) — Ryzen 9 9950X has 60 GB
lib.num_threads(16)  # Use 16 physical cores

# ============================================================================
# B12 ICOSAHEDRAL GEOMETRY (Ih symmetry) — from structures/
# ============================================================================

B12_ICO_GEOMETRY = """
B  0.000000 -0.850000 -1.375329
B -1.375329  0.000000 -0.850000
B -0.850000 -1.375329  0.000000
B  0.000000 -0.850000  1.375329
B  1.375329  0.000000 -0.850000
B -0.850000  1.375329  0.000000
B  0.000000  0.850000 -1.375329
B -1.375329  0.000000  0.850000
B  0.850000 -1.375329  0.000000
B  0.000000  0.850000  1.375329
B  1.375329  0.000000  0.850000
B  0.850000  1.375329  0.000000
"""

# SVP reference values from Table 1 of the manuscript
SVP_REFERENCE = {
    'N_D': 4.99,
    'f_e': 0.139,
    'n_electrons': 60,
    'n_corr': 36,
    'M': 168,
    'M_frac': 106,
    'E_corr': -1.173,
}


def run_b12_ico_ccsd(basis='def2-tzvp'):
    """Run CCSD N_D analysis for B12 icosahedral at given basis set."""

    t_start = time.time()

    print("=" * 70)
    print(f"  N_D Basis-Set Validation: B12 Icosahedral (Ih)")
    print(f"  Method: CCSD/{basis}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    mol = gto.M(
        atom=B12_ICO_GEOMETRY,
        basis=basis,
        charge=0,
        spin=0,
        symmetry=True,
        unit='Angstrom',
        verbose=4,
        max_memory=24000,  # 24 GB — force outcore mode for CCSD integrals
    )

    n_atoms = mol.natm
    n_elec = mol.nelectron
    n_ao = mol.nao
    # Freeze B 1s core orbitals (12 borons)
    n_frozen = 12
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

    # Free SCF intermediates before starting CCSD
    import gc
    gc.collect()

    # CCSD — force outcore integral handling for large basis
    print(f"\n{'─' * 50}")
    print("  Running CCSD (outcore mode, this may take several hours at TZVP)...")
    print(f"  Scratch directory: {os.environ.get('PYSCF_TMPDIR', '/tmp')}")
    mycc = cc.CCSD(mf)
    mycc.frozen = n_frozen
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-6
    mycc.max_cycle = 200
    mycc.max_memory = 24000  # Explicitly set CCSD memory limit
    mycc.direct = True  # Force AO-direct integrals (much less RAM)

    ecc, t1, t2 = mycc.kernel()

    if not mycc.converged:
        print("CCSD failed to converge!")
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

    print(f"\n  Top 20 natural orbital occupations:")
    for i, n in enumerate(noons[:20]):
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
    print(f"  RESULTS for B12(ico) CCSD/{basis}:")
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
    results = {
        'system': 'B12_icosahedral',
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

    outfile = f'B12_icosahedral_def2tzvp_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {outfile}")

    # ----- Comparison with def2-SVP reference -----
    svp = SVP_REFERENCE
    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  BASIS-SET COMPARISON: def2-SVP vs {basis}      ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Property       │ def2-SVP  │ {basis:10s}      ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  N_D            │ {svp['N_D']:9.4f} │ {N_D:9.4f}        ║")
    print(f"  ║  f_e            │ {svp['f_e']:9.4f} │ {f_e:9.4f}        ║")
    delta_nd = (N_D - svp['N_D']) / svp['N_D'] * 100
    delta_fe = (f_e - svp['f_e']) / svp['f_e'] * 100
    print(f"  ║  Δ(N_D)         │           │ {delta_nd:+.1f}%          ║")
    print(f"  ║  Δ(f_e)         │           │ {delta_fe:+.1f}%          ║")
    print(f"  ║  Regime?        │ small clust│ {'small clust' if f_e > 0.06 else 'SUPERATOM!':11s}║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    return results


if __name__ == '__main__':
    run_b12_ico_ccsd(basis='def2-tzvp')
