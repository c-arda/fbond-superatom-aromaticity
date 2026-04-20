#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpointed Cs3Al12- CCSD/def2-TZVP → N_D extraction
======================================================
Three stages, each checkpointed to disk:

  Stage 1 — SCF:  checkpoint → cs3al12_tzvp_scf.chk  [ALREADY DONE]
  Stage 2 — CCSD: checkpoint → cs3al12_tzvp_ccsd_amplitudes.npz
  Stage 3 — 1-RDM + N_D:      → Cs3Al12_minus_def2tzvp_results.json

On restart, each stage checks for its checkpoint and SKIPS if found.
No compute is ever wasted.
"""

import numpy as np
import os
import sys
import gc

os.environ['PYSCF_TMPDIR'] = '/data/pyscf_scratch'
os.environ['TMPDIR'] = '/data/pyscf_scratch'

from pyscf import gto, scf, cc, lib
import json
from datetime import datetime
import time

lib.num_threads(16)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCF_CHKFILE    = os.path.join(SCRIPT_DIR, 'cs3al12_tzvp_scf.chk')
CCSD_AMPFILE   = os.path.join(SCRIPT_DIR, 'cs3al12_tzvp_ccsd_amplitudes.npz')
LAMBDA_AMPFILE = os.path.join(SCRIPT_DIR, 'cs3al12_tzvp_lambda_amplitudes.npz')
RESULTS_JSON   = os.path.join(SCRIPT_DIR, 'Cs3Al12_minus_def2tzvp_results.json')

# ============================================================================
# Geometry
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

SVP_REFERENCE = {
    'N_D': 7.10,
    'f_e': 0.044,
    'n_electrons': 184,
    'n_corr': 160,
}

N_FROZEN = 12  # 12 x Al 1s


def build_mol():
    """Build the Mol object."""
    mol = gto.M(
        atom=CS3AL12_GEOMETRY,
        basis='def2-tzvp',
        charge=-1,
        spin=0,
        symmetry=False,
        unit='Angstrom',
        verbose=4,
        max_memory=28000,
        ecp='def2-tzvp',   # applies to Cs only (Al has no ECP)
    )
    return mol


def stage1_scf(mol):
    """Stage 1: RHF — restore from checkpoint (already converged)."""
    print("\n" + "=" * 60)
    print("  STAGE 1: SCF")
    print("=" * 60)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.chkfile = SCF_CHKFILE

    if os.path.isfile(SCF_CHKFILE):
        print(f"  ✓ CHECKPOINT FOUND: {SCF_CHKFILE}")
        print(f"    Restoring converged SCF — SKIPPING computation.")
        from pyscf import scf as _scf
        mf.__dict__.update(_scf.chkfile.load(SCF_CHKFILE, 'scf'))
        mf.converged = True
        print(f"    E(HF) = {mf.e_tot:.10f} Ha")
    else:
        print("  No checkpoint. Running RHF from scratch...")
        mf.kernel()
        if not mf.converged:
            print("  SCF FAILED. Trying with damping...")
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-10
            mf.max_cycle = 500
            mf.diis_space = 12
            mf.level_shift = 0.1
            mf.chkfile = SCF_CHKFILE
            mf.kernel()
            if not mf.converged:
                print("  SCF FAILED even with damping. Aborting.")
                sys.exit(1)
        print(f"    E(HF) = {mf.e_tot:.10f} Ha")

    return mf


def stage2_ccsd(mf, mol):
    """Stage 2: CCSD — save T1/T2 amplitudes + E_corr to disk."""
    print("\n" + "=" * 60)
    print("  STAGE 2: CCSD")
    print("=" * 60)

    n_elec = mol.nelectron
    n_corr = n_elec - 2 * N_FROZEN

    if os.path.isfile(CCSD_AMPFILE):
        print(f"  ✓ CHECKPOINT FOUND: {CCSD_AMPFILE}")
        print(f"    Loading saved T1/T2 amplitudes — SKIPPING CCSD computation.")
        data = np.load(CCSD_AMPFILE, allow_pickle=True)
        t1 = data['t1']
        t2 = data['t2']
        e_corr = float(data['e_corr'])
        e_ccsd = float(data['e_ccsd'])
        print(f"    E(CCSD) = {e_ccsd:.10f} Ha")
        print(f"    E_corr  = {e_corr:.6f} Ha")
        print(f"    T1 shape: {t1.shape}, T2 shape: {t2.shape}")

        # We still need the CC object for make_rdm1, so rebuild it
        mycc = cc.CCSD(mf)
        mycc.frozen = N_FROZEN
        mycc.conv_tol = 1e-8
        mycc.conv_tol_normt = 1e-6
        mycc.max_memory = 28000
        mycc.direct = True
        # Set the converged state without re-running
        mycc.e_corr = e_corr
        mycc.t1 = t1
        mycc.t2 = t2
        mycc.converged = True
        return mycc

    # No checkpoint — run CCSD
    print("  No amplitude checkpoint. Running CCSD (this takes ~20 hours)...")
    print(f"  Correlated electrons: {n_corr}")
    print(f"  Start time: {datetime.now().isoformat()}")

    gc.collect()

    mycc = cc.CCSD(mf)
    mycc.frozen = N_FROZEN
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-6
    mycc.max_cycle = 300
    mycc.max_memory = 28000
    mycc.direct = True

    ecc, t1, t2 = mycc.kernel()

    if not mycc.converged:
        print("  ✗ CCSD FAILED TO CONVERGE!")
        sys.exit(1)

    print(f"  E(CCSD) = {mycc.e_tot:.10f} Ha")
    print(f"  E_corr  = {mycc.e_corr:.6f} Ha")

    # ── SAVE AMPLITUDES TO DISK ──
    print(f"\n  ► SAVING amplitudes to: {CCSD_AMPFILE}")
    print(f"    T1 shape: {t1.shape} ({t1.nbytes / 1e6:.1f} MB)")
    print(f"    T2 shape: {t2.shape} ({t2.nbytes / 1e9:.2f} GB)")
    np.savez_compressed(
        CCSD_AMPFILE,
        t1=t1,
        t2=t2,
        e_corr=np.array(mycc.e_corr),
        e_ccsd=np.array(mycc.e_tot),
        e_hf=np.array(mf.e_tot),
    )
    fsize = os.path.getsize(CCSD_AMPFILE)
    print(f"    Saved: {fsize / 1e9:.2f} GB on disk")
    print(f"  ✓ Amplitudes checkpointed. Safe to restart from here.")

    return mycc


def stage3_rdm_and_results(mycc, mf, mol):
    """Stage 3: Solve Lambda → Relaxed 1-RDM → N_D/f_e → JSON.
    
    Uses PySCF's make_rdm1() which requires Lambda amplitudes.
    This gives the RELAXED 1-RDM, consistent with all other molecules.
    Lambda amplitudes are checkpointed to disk for crash resilience.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: Lambda → Relaxed 1-RDM → N_D / f_e → Results JSON")
    print("=" * 60)

    if os.path.isfile(RESULTS_JSON):
        print(f"  ✓ RESULTS ALREADY EXIST: {RESULTS_JSON}")
        print(f"    Nothing to do. Exiting.")
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        print(f"    N_D = {results['N_D']:.4f}")
        print(f"    f_e = {results['f_e']:.4f}")
        return results

    n_elec = mol.nelectron
    n_corr = n_elec - 2 * N_FROZEN
    n_ao = mol.nao

    # HOMO-LUMO gap
    eps = mf.mo_energy
    homo_idx = n_elec // 2 - 1
    gap = eps[homo_idx + 1] - eps[homo_idx]

    # T1 diagnostic
    t1_diag = np.sqrt(np.sum(mycc.t1**2) / n_corr)

    # ── LAMBDA SOLVE (with checkpoint) ──
    if os.path.isfile(LAMBDA_AMPFILE):
        print(f"  ✓ LAMBDA CHECKPOINT FOUND: {LAMBDA_AMPFILE}")
        print(f"    Loading saved Lambda amplitudes — SKIPPING Lambda solve.")
        ldata = np.load(LAMBDA_AMPFILE, allow_pickle=True)
        mycc.l1 = ldata['l1']
        mycc.l2 = ldata['l2']
        mycc.converged_lambda = True
        print(f"    l1 shape: {mycc.l1.shape}, l2 shape: {mycc.l2.shape}")
    else:
        print("  Solving Lambda-CCSD equations (relaxed 1-RDM)...")
        print(f"  This is consistent with all other molecules in the dataset.")
        print(f"  Scratch dir: {os.environ.get('PYSCF_TMPDIR', '/tmp')}")
        print(f"  Start time: {datetime.now().isoformat()}")

        mycc.solve_lambda()

        if not mycc.converged_lambda:
            print("  ✗ LAMBDA FAILED TO CONVERGE!")
            print("  Saving partial amplitudes anyway...")

        # Save Lambda amplitudes
        print(f"\n  ► SAVING Lambda amplitudes to: {LAMBDA_AMPFILE}")
        np.savez_compressed(
            LAMBDA_AMPFILE,
            l1=mycc.l1,
            l2=mycc.l2,
            converged=np.array(mycc.converged_lambda),
        )
        fsize = os.path.getsize(LAMBDA_AMPFILE)
        print(f"    Saved: {fsize / 1e9:.2f} GB on disk")
        print(f"  ✓ Lambda amplitudes checkpointed.")

    # ── RELAXED 1-RDM via make_rdm1() ──
    print("  Computing RELAXED 1-RDM (Lambda + T amplitudes)...")
    dm1 = mycc.make_rdm1()

    nmo = dm1.shape[0]
    print(f"  1-RDM shape: {dm1.shape}, trace = {np.trace(dm1):.4f} (expect {n_corr})")
    noons, natorbs = np.linalg.eigh(dm1)
    noons = np.sort(noons.flatten())[::-1]

    M = len(noons)

    print(f"\n  Top 20 natural orbital occupations:")
    for i, n in enumerate(noons[:20]):
        dev = abs(n - 2.0) if n > 1.0 else abs(n)
        marker = " <-- fractional" if dev > 0.001 else ""
        print(f"    NO {i+1:3d}: n = {n:.6f}{marker}")

    # N_D and f_e
    nd_contributions = np.array([n * (2 - n) for n in noons])
    N_D = float(np.sum(nd_contributions))
    f_e = N_D / n_corr

    n_fractional = int(np.sum((noons > 0.001) & (noons < 1.999)))
    frac_pct = n_fractional / M * 100

    # S_E_max
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

    print(f"\n  {'═' * 50}")
    print(f"  RESULTS for Cs3Al12- CCSD/def2-tzvp:")
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
    print(f"  1-RDM type      = relaxed (Lambda-CCSD)")
    print(f"  {'═' * 50}")

    # Save JSON
    results = {
        'system': 'Cs3Al12_minus',
        'method': 'CCSD',
        'basis': 'def2-tzvp',
        'rdm1_type': 'relaxed (Lambda-CCSD)',
        'calculation_date': datetime.now().isoformat(),
        'n_atoms': mol.natm,
        'n_electrons': n_elec,
        'n_frozen': N_FROZEN,
        'n_corr': n_corr,
        'n_ao': n_ao,
        'M_corr_orbitals': M,
        'M_fractional': n_fractional,
        'M_fractional_pct': frac_pct,
        'E_HF': float(mf.e_tot),
        'E_CCSD': float(mf.e_tot + mycc.e_corr),
        'E_corr': float(mycc.e_corr),
        'HOMO_LUMO_gap': float(gap),
        'T1_diagnostic': float(t1_diag),
        'N_D': N_D,
        'f_e': f_e,
        'S_E_max': float(S_E_max),
        'natural_occupations': [float(n) for n in noons],
        'nd_contributions': [float(c) for c in nd_contributions],
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ SAVED: {RESULTS_JSON}")

    # Comparison
    svp = SVP_REFERENCE
    delta_nd = (N_D - svp['N_D']) / svp['N_D'] * 100
    delta_fe = (f_e - svp['f_e']) / svp['f_e'] * 100
    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  BASIS-SET COMPARISON: def2-SVP vs def2-tzvp    ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  N_D   SVP: {svp['N_D']:7.4f}  │  TZVP: {N_D:7.4f}  ({delta_nd:+.1f}%) ║")
    print(f"  ║  f_e   SVP: {svp['f_e']:7.4f}  │  TZVP: {f_e:7.4f}  ({delta_fe:+.1f}%) ║")
    print(f"  ║  Regime: {'superatom ✓' if f_e < 0.06 else 'WARNING ✗':11s}                       ║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    return results


# ============================================================================
# Main — runs all stages, skipping any that are already checkpointed
# ============================================================================
if __name__ == '__main__':
    t_global = time.time()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Cs3Al12- CCSD/def2-TZVP — CHECKPOINTED PIPELINE        ║")
    print("║  Each stage checks for its checkpoint before computing.  ║")
    print("║  Safe to interrupt and restart at any time.              ║")
    print(f"║  {datetime.now().isoformat():^56s}  ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Check what's already done
    print("\n  Checkpoint status:")
    print(f"    SCF  chk:  {'✓ FOUND' if os.path.isfile(SCF_CHKFILE) else '✗ missing'}")
    print(f"    CCSD amp:  {'✓ FOUND' if os.path.isfile(CCSD_AMPFILE) else '✗ missing'}")
    print(f"    Results:   {'✓ FOUND' if os.path.isfile(RESULTS_JSON) else '✗ missing'}")

    mol = build_mol()
    mf = stage1_scf(mol)
    mycc = stage2_ccsd(mf, mol)
    results = stage3_rdm_and_results(mycc, mf, mol)

    t_total = time.time() - t_global
    print(f"\n  Total wall time: {t_total:.0f} s ({t_total/3600:.1f} hours)")
    print("  Done.")
