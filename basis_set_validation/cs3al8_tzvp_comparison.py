#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis-set comparison: Cs3Al8- CCSD/def2-TZVP vs def2-SVP
==========================================================
CRITICAL: This is the first superatom tested at TZVP.
Validates whether the metallic superatom regime (f_e ~ 0.03-0.05)
is robust to basis set choice.

Expected runtime: ~8-20 hours on AMD Ryzen 9 9950X (16c/32t).
Cs requires def2-ECP (46 core electrons replaced per atom).
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
# Cs3Al8- GEOMETRY — from structures/ (B3LYP/def2-SVP optimized)
# ============================================================================

CS3AL8_GEOMETRY = """
Al   -0.411813   -0.144325   -1.074838
Al    1.765804   -0.621967    0.370974
Al   -1.848368    0.914399    0.842624
Al    0.567026    1.804993    0.442228
Al   -0.399734   -2.017759    0.757018
Al    3.152135    1.592862   -0.269693
Al   -2.862224   -1.376900   -0.372964
Al   -0.020634   -0.055130    2.470913
Cs    0.126746   -0.054060    6.696133
Cs    3.503523    1.657175    3.433021
Cs   -3.450254   -1.780681    3.316235
"""

# SVP reference values from Table 1 of the manuscript
SVP_REFERENCE = {
    'N_D': 5.58,
    'f_e': 0.048,
    'n_electrons': 132,
    'n_corr': 116,
    'M': 216,
    'M_frac': 208,
    'E_corr': -0.836,
}


def run_cs3al8_ccsd(basis='def2-tzvp'):
    """Run CCSD N_D analysis for Cs3Al8- at given basis set."""

    t_start = time.time()

    print("=" * 70)
    print(f"  N_D Basis-Set Validation: Cs3Al8- (Metallic Superatom)")
    print(f"  Method: CCSD/{basis}")
    print(f"  NOTE: def2-ECP used for Cs (replaces 46 core e- per atom)")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    mol = gto.M(
        atom=CS3AL8_GEOMETRY,
        basis=basis,
        charge=-1,
        spin=0,
        symmetry=False,  # Low symmetry, don't force
        unit='Angstrom',
        verbose=4,
        max_memory=24000,  # 24 GB — force outcore mode
        ecp=basis,  # def2-ECP automatically applied for Cs
    )

    n_atoms = mol.natm
    n_elec = mol.nelectron
    n_ao = mol.nao

    # Frozen core: Al 1s (8 orbitals for 8 Al atoms)
    # Cs core is already handled by ECP (46 electrons replaced per Cs)
    n_frozen_al = 8  # 8 x Al 1s
    n_frozen = n_frozen_al
    n_corr = n_elec - 2 * n_frozen  # correlated electrons

    print(f"\nMolecule: {n_atoms} atoms, {n_elec} electrons (post-ECP), {n_ao} AOs")
    print(f"Cs ECP: 46 core electrons replaced per atom (3 atoms = 138 e-)")
    print(f"Frozen core: {n_frozen} orbitals ({2*n_frozen} electrons) [Al 1s]")
    print(f"Correlated electrons: {n_corr}")
    print(f"Correlated orbitals (M): {n_ao - n_frozen}")

    # Memory estimate
    m_corr = n_ao - n_frozen
    mem_est_gb = (m_corr**4 * 8) / (1024**3) * 0.1  # rough CCSD memory
    print(f"\n  Estimated CCSD memory: ~{mem_est_gb:.1f} GB")
    print(f"  Available RAM: ~60 GB — {'OK' if mem_est_gb < 50 else 'TIGHT!'}")

    # SCF
    print(f"\n{'─' * 50}")
    print("  Running RHF...")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()

    if not mf.converged:
        print("SCF failed! Trying with DIIS damping...")
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.max_cycle = 500
        mf.diis_space = 12
        mf.level_shift = 0.1
        mf.kernel()
        if not mf.converged:
            print("SCF still failed!")
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

    # CCSD
    print(f"\n{'─' * 50}")
    print("  Running CCSD (outcore/direct mode, THIS WILL TAKE MANY HOURS)...")
    print(f"  Scratch directory: {os.environ.get('PYSCF_TMPDIR', '/tmp')}")
    print(f"  Start time: {datetime.now().isoformat()}")
    mycc = cc.CCSD(mf)
    mycc.frozen = n_frozen
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-6
    mycc.max_cycle = 300
    mycc.max_memory = 24000
    mycc.direct = True  # Force AO-direct integrals

    ecc, t1, t2 = mycc.kernel()

    if not mycc.converged:
        print("CCSD failed to converge!")
        # Still save partial results
        results = {
            'system': 'Cs3Al8_minus',
            'method': 'CCSD',
            'basis': basis,
            'status': 'CCSD_NOT_CONVERGED',
            'calculation_date': datetime.now().isoformat(),
            'wall_time_seconds': time.time() - t_start,
        }
        with open('Cs3Al8_minus_def2tzvp_FAILED.json', 'w') as f:
            json.dump(results, f, indent=2)
        return None

    print(f"  E(CCSD) = {mycc.e_tot:.10f} Ha")
    print(f"  E_corr  = {mycc.e_corr:.6f} Ha = {mycc.e_corr * 1000:.3f} mHa")

    # T1 diagnostic
    t1_diag = np.sqrt(np.sum(t1**2) / n_corr)
    print(f"  T1 diagnostic = {t1_diag:.6f}")

    # Lambda equations for proper 1-RDM
    print(f"\n{'─' * 50}")
    print("  Solving Lambda-CCSD equations...")
    print(f"  Start time: {datetime.now().isoformat()}")
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
    print(f"  RESULTS for Cs3Al8- CCSD/{basis}:")
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
    print(f"  Wall time       = {t_elapsed:.0f} s ({t_elapsed/3600:.1f} hours)")
    print(f"  {'═' * 50}")

    # Save results
    results = {
        'system': 'Cs3Al8_minus',
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

    outfile = f'Cs3Al8_minus_def2tzvp_results.json'
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
    print(f"  ║  Regime?        │ superatom │ {'superatom' if f_e < 0.06 else 'SMALL CLU!':10s}║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    # Key question: is there still clear separation?
    print(f"\n  ┌────────────────────────────────────────────────┐")
    print(f"  │  REGIME SEPARATION TEST                        │")
    print(f"  │  Small cluster boundary: f_e > 0.06            │")
    print(f"  │  Superatom boundary:     f_e < 0.06            │")
    print(f"  │  This system f_e = {f_e:.4f}                      │")
    if f_e < 0.06:
        print(f"  │  ✓ CONFIRMED: Remains in superatom regime      │")
    else:
        print(f"  │  ✗ WARNING: Shifted to small-cluster regime!    │")
    print(f"  └────────────────────────────────────────────────┘")

    return results


if __name__ == '__main__':
    run_cs3al8_ccsd(basis='def2-tzvp')
