#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis-set comparison: Cs3Al12- CCSD/def2-TZVP vs def2-SVP
==========================================================
Validates whether the metallic superatom regime (f_e ~ 0.03-0.05)
is robust to basis set choice for the LARGER superatom.

Together with Cs3Al8- TZVP, this establishes whether f_e convergence
within the Cs3Al_n^- family is maintained at the triple-zeta level.

Expected runtime: ~50-80 hours on AMD Ryzen 9 9950X (16c/32t).
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

# Set PySCF max memory (MB) — use 16 physical cores
lib.num_threads(16)

# ============================================================================
# Cs3Al12- GEOMETRY — from structures/ (B3LYP/def2-SVP optimized)
# ============================================================================

CS3AL12_GEOMETRY = """
Al    0.240775   -1.015621    0.787064
Al    2.791948    0.020814    0.757961
Al   -1.578346    1.030335    1.379282
Al    0.748146    1.751398   -0.006054
Al   -0.589180   -1.967493   -1.631605
Al   -0.889506   -0.961686    3.260760
Al    1.322308   -0.158891   -1.898615
Al    3.265819    1.509080   -1.447930
Al   -2.636775   -1.412811    0.047917
Al    1.866522   -1.101066    2.948536
Al   -1.255320    0.628876   -1.172729
Al    0.766838    1.406232    2.559120
Cs    0.660649    0.243682    6.273551
Cs    3.973228    3.473232    1.588375
Cs   -4.629039   -0.879977    3.093120
"""

# SVP reference values from Table 1 of the manuscript
SVP_REFERENCE = {
    'N_D': 7.10,
    'f_e': 0.044,
    'n_electrons': 184,
    'n_corr': 160,
    'M': 288,
    'M_frac': 276,
    'E_corr': -1.184,
}


def run_cs3al12_ccsd(basis='def2-tzvp'):
    """Run CCSD N_D analysis for Cs3Al12- at given basis set."""

    t_start = time.time()

    print("=" * 70)
    print(f"  N_D Basis-Set Validation: Cs3Al12- (Metallic Superatom)")
    print(f"  Method: CCSD/{basis}")
    print(f"  NOTE: def2-ECP used for Cs (replaces 46 core e- per atom)")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    mol = gto.M(
        atom=CS3AL12_GEOMETRY,
        basis=basis,
        charge=-1,
        spin=0,
        symmetry=False,
        unit='Angstrom',
        verbose=4,
        max_memory=28000,  # 28 GB — larger system needs more
        ecp=basis,  # def2-ECP automatically applied for Cs
    )

    n_atoms = mol.natm
    n_elec = mol.nelectron
    n_ao = mol.nao

    # Frozen core: Al 1s (12 orbitals for 12 Al atoms)
    # Cs core handled by ECP (46 electrons replaced per Cs)
    n_frozen_al = 12  # 12 x Al 1s
    n_frozen = n_frozen_al
    n_corr = n_elec - 2 * n_frozen  # correlated electrons

    print(f"\nMolecule: {n_atoms} atoms, {n_elec} electrons (post-ECP), {n_ao} AOs")
    print(f"Cs ECP: 46 core electrons replaced per atom (3 atoms = 138 e-)")
    print(f"Frozen core: {n_frozen} orbitals ({2*n_frozen} electrons) [Al 1s]")
    print(f"Correlated electrons: {n_corr}")
    print(f"Correlated orbitals (M): {n_ao - n_frozen}")

    # Memory estimate
    m_corr = n_ao - n_frozen
    mem_est_gb = (m_corr**4 * 8) / (1024**3) * 0.1
    print(f"\n  Estimated CCSD memory: ~{mem_est_gb:.1f} GB")
    print(f"  Available RAM: ~60 GB — {'OK' if mem_est_gb < 50 else 'TIGHT!'}")

    # SCF — try to restore from checkpoint saved during prior interrupted run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chkfile = os.path.join(script_dir, 'cs3al12_tzvp_scf.chk')

    print(f"\n{'─' * 50}")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.chkfile = chkfile  # persist future SCF to stable location

    if os.path.isfile(chkfile):
        print("  ► RESTART: Loading converged RHF from checkpoint...")
        print(f"    Checkpoint: {chkfile}")
        from pyscf import scf as _scf
        mf.__dict__.update(_scf.chkfile.load(chkfile, 'scf'))
        mf.converged = True
        print(f"  E(HF) = {mf.e_tot:.10f} Ha  [restored from checkpoint]")
    else:
        print("  Running RHF (no checkpoint found)...")
        mf.kernel()
        if not mf.converged:
            print("SCF failed! Trying with DIIS damping...")
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-10
            mf.max_cycle = 500
            mf.diis_space = 12
            mf.level_shift = 0.1
            mf.chkfile = chkfile
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
    mycc.max_memory = 28000
    mycc.direct = True  # Force AO-direct integrals

    ecc, t1, t2 = mycc.kernel()

    if not mycc.converged:
        print("CCSD failed to converge!")
        results = {
            'system': 'Cs3Al12_minus',
            'method': 'CCSD',
            'basis': basis,
            'status': 'CCSD_NOT_CONVERGED',
            'calculation_date': datetime.now().isoformat(),
            'wall_time_seconds': time.time() - t_start,
        }
        with open('Cs3Al12_minus_def2tzvp_FAILED.json', 'w') as f:
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

    M = len(noons)

    print(f"\n  Top 20 natural orbital occupations:")
    for i, n in enumerate(noons[:20]):
        dev = abs(n - 2.0) if n > 1.0 else abs(n)
        marker = " <-- fractional" if dev > 0.001 else ""
        print(f"    NO {i+1:3d}: n = {n:.6f}{marker}")

    # N_D = sum_i n_i(2 - n_i)
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
    print(f"  RESULTS for Cs3Al12- CCSD/{basis}:")
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
        'system': 'Cs3Al12_minus',
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

    outfile = f'Cs3Al12_minus_def2tzvp_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {outfile}")

    # Comparison with SVP reference
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

    # Comparison with Cs3Al8- TZVP
    cs3al8_tzvp_fe = 0.059
    print(f"\n  ┌────────────────────────────────────────────────┐")
    print(f"  │  SUPERATOM FAMILY CONVERGENCE TEST             │")
    print(f"  │  Cs3Al8-  TZVP f_e = {cs3al8_tzvp_fe:.4f}                │")
    print(f"  │  Cs3Al12- TZVP f_e = {f_e:.4f}                │")
    delta_family = abs(f_e - cs3al8_tzvp_fe) / cs3al8_tzvp_fe * 100
    print(f"  │  Family Δf_e       = {delta_family:.1f}%                  │")
    if f_e < 0.06:
        print(f"  │  ✓ Both in superatom regime at TZVP            │")
    else:
        print(f"  │  ✗ WARNING: Cs3Al12- exits superatom regime!    │")
    print(f"  └────────────────────────────────────────────────┘")

    return results


if __name__ == '__main__':
    run_cs3al12_ccsd(basis='def2-tzvp')
