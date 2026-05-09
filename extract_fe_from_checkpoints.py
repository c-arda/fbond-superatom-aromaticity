#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute f_e / N_D from existing CCSD checkpoints (def2-SVP)
=============================================================

The original automated_fbond_workflow.py saved CCSD checkpoints with the
full 1-RDM (dm1) but only computed the model-based F_bond metric. This
script re-extracts the PROPER quantum chemical metrics:

  N_D   = Σ_i n_i(2 − n_i)          Takatsuka–Head-Gordon delocalization index
  f_e   = N_D / N_corr              fractional electron per correlated electron
  S_E_max  = max_i entropy(n_i)     maximum single-orbital entanglement entropy

These are the metrics that should go in the paper — NOT the model-based
F_bond = 0.5 × O_MOS × entropy(sigmoid(−O_MOS)) shortcut.

Usage:
  python extract_fe_from_checkpoints.py

Reads from:  cloud_checkpoints/*_checkpoint_ccsd.pkl
             cloud_results/*_results.json      (for existing HF data)
Writes to:   cloud_results/*_results_corrected.json
"""

import json
import os
import pickle
import sys
import glob
import numpy as np
from datetime import datetime


# ============================================================================
# FROZEN CORE LOOKUP
# ============================================================================
# Same convention as the original workflow: freeze 1s orbitals for
# elements up to Ne, 1s2s2p for elements up to Ar, etc.
# For ECP-treated elements (Cs, Rb), the ECP replaces core electrons,
# so we only count the valence-treated atoms.

SYSTEM_FROZEN_CORES = {
    # system_name: (n_frozen_orbitals, n_correlated_electrons)
    # These were set in automated_fbond_workflow.py at runtime.
    # n_frozen = count of Al atoms for the Al-cluster systems,
    # n_frozen = 6 for benzene (C 1s), etc.
    # We'll auto-detect from the results JSON where possible.
}


def get_frozen_core_count(system_name, existing_results):
    """Get frozen core orbital count from existing results or heuristics."""
    if existing_results and 'n_frozen' in existing_results:
        return existing_results['n_frozen']

    # Fallback: count from system name
    name_upper = system_name.upper()
    if 'BENZENE' in name_upper or 'C6H6' in name_upper:
        return 6  # 6 carbon 1s
    elif 'B12' in name_upper:
        return 12  # 12 boron 1s
    elif 'B6N6' in name_upper:
        return 12  # 6 boron + 6 nitrogen 1s
    elif 'CS3AL12' in name_upper:
        return 12  # 12 Al 1s (Cs uses ECP)
    elif 'CS3AL8' in name_upper:
        return 8   # 8 Al 1s (Cs uses ECP)
    elif 'AL4' in name_upper:
        return 4   # 4 Al 1s
    else:
        raise ValueError(f"Cannot determine frozen core for {system_name}")


def compute_noon_metrics(dm1, n_electrons, n_frozen):
    """
    Compute f_e, N_D, and S_E_max from the CCSD 1-RDM.

    Parameters
    ----------
    dm1 : ndarray
        CCSD 1-RDM in the MO basis (correlated space only).
    n_electrons : int
        Total number of electrons.
    n_frozen : int
        Number of frozen core orbitals.

    Returns
    -------
    dict with N_D, f_e, S_E_max, noons, T1_diag (if available), etc.
    """
    # Diagonalize 1-RDM → Natural Orbital Occupation Numbers (NOONs)
    noons_raw, natorbs = np.linalg.eigh(dm1)
    # Sort descending (highest occupation first)
    noons = np.sort(noons_raw.flatten())[::-1]

    M = len(noons)  # number of correlated orbitals
    n_corr = n_electrons - 2 * n_frozen  # correlated electrons

    # ── N_D: Takatsuka–Head-Gordon delocalization index ──
    # N_D = Σ_i n_i(2 − n_i)
    # For doubly occupied (n=2): contributes 0
    # For empty (n=0): contributes 0
    # Maximum contribution at n=1 (fully fractional): contributes 1
    nd_contributions = noons * (2.0 - noons)
    N_D = float(np.sum(nd_contributions))

    # ── f_e: fractional electron per correlated electron ──
    f_e = N_D / n_corr if n_corr > 0 else 0.0

    # ── S_E_max: maximum single-orbital entanglement entropy ──
    S_list = []
    for n in noons:
        if 0.0 < n < 2.0:
            p = n / 2.0
            if 0.0 < p < 1.0:
                S_i = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
            else:
                S_i = 0.0
        else:
            S_i = 0.0
        S_list.append(S_i)
    S_E_max = float(max(S_list)) if S_list else 0.0

    # ── Fractional orbital count ──
    n_fractional = int(np.sum((noons > 0.001) & (noons < 1.999)))
    frac_pct = n_fractional / M * 100 if M > 0 else 0.0

    return {
        'N_D': float(N_D),
        'f_e': float(f_e),
        'S_E_max_noon': float(S_E_max),
        'n_corr': int(n_corr),
        'M_corr_orbitals': int(M),
        'M_fractional': int(n_fractional),
        'M_fractional_pct': float(frac_pct),
        'natural_occupations_full': [float(x) for x in noons],
        'nd_contributions': [float(x) for x in nd_contributions],
        'entropy_values_noon': [float(x) for x in S_list],
    }


def process_checkpoint(checkpoint_path, results_dir='cloud_results'):
    """Process a single CCSD checkpoint and compute f_e/N_D."""

    basename = os.path.basename(checkpoint_path)
    # Extract system name: "{system}_checkpoint_ccsd.pkl"
    system_name = basename.replace('_checkpoint_ccsd.pkl', '')

    print(f"\n{'─'*60}")
    print(f"  Processing: {system_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'─'*60}")

    # Load checkpoint
    try:
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        return None

    # Extract dm1
    if 'dm1' not in ckpt:
        print(f"  ✗ Checkpoint does not contain 'dm1' — cannot compute NOONs")
        print(f"    Available keys: {list(ckpt.keys())}")
        return None

    dm1_raw = ckpt['dm1']

    # Handle open-shell systems where dm1 is (alpha, beta) tuple
    if isinstance(dm1_raw, (tuple, list)):
        dm1_a, dm1_b = dm1_raw[0], dm1_raw[1]
        dm1 = dm1_a + dm1_b  # total 1-RDM
        print(f"  ✓ Open-shell dm1: alpha {dm1_a.shape} + beta {dm1_b.shape} → total {dm1.shape}")
    else:
        dm1 = dm1_raw
        print(f"  ✓ dm1 shape: {dm1.shape}")

    # Load existing results for metadata
    results_file = os.path.join(results_dir, f"{system_name}_results.json")
    existing = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = json.load(f)
        print(f"  ✓ Loaded existing results from {results_file}")
    else:
        print(f"  ⚠ No existing results file found at {results_file}")

    # Get frozen core count
    n_frozen = get_frozen_core_count(system_name, existing)
    n_electrons = existing['n_electrons'] if existing else None

    if n_electrons is None:
        print(f"  ✗ Cannot determine n_electrons — skipping")
        return None

    print(f"  n_electrons = {n_electrons}")
    print(f"  n_frozen    = {n_frozen}")
    print(f"  n_corr      = {n_electrons - 2 * n_frozen}")

    # Compute NOON-based metrics
    metrics = compute_noon_metrics(dm1, n_electrons, n_frozen)

    print(f"\n  ◉ RESULTS:")
    print(f"    N_D       = {metrics['N_D']:.4f}")
    print(f"    f_e       = {metrics['f_e']:.4f}")
    print(f"    S_E_max   = {metrics['S_E_max_noon']:.6f} nats")
    print(f"    M_frac    = {metrics['M_fractional']} / {metrics['M_corr_orbitals']} "
          f"({metrics['M_fractional_pct']:.0f}%)")

    # Top NOONs
    noons = metrics['natural_occupations_full']
    print(f"\n    Top 15 NOONs:")
    for i, n in enumerate(noons[:15]):
        dev = abs(n - 2.0) if n > 1.0 else abs(n)
        marker = " ◄ fractional" if dev > 0.001 else ""
        print(f"      NO {i+1:3d}: {n:.6f}{marker}")

    # If existing results: also report old F_bond for comparison
    if existing and 'F_bond' in existing:
        print(f"\n    ── Comparison with model-based metric ──")
        print(f"    F_bond (model)    = {existing['F_bond']:.6f}")
        print(f"    O_MOS  (HF gap)   = {existing.get('O_MOS', 'N/A')}")
        print(f"    S_E_max (model)   = {existing.get('S_E_max', 'N/A')}")
        print(f"    f_e    (from NOON) = {metrics['f_e']:.6f}")

    # Build corrected results
    corrected = {}
    if existing:
        corrected.update(existing)
    corrected.update(metrics)
    corrected['extraction_date'] = datetime.now().isoformat()
    corrected['extraction_source'] = 'checkpoint_reanalysis'

    # Replace the truncated natural_occupations with full set
    if 'natural_occupations' in corrected:
        corrected['natural_occupations_truncated_20'] = corrected.pop('natural_occupations')

    # Save corrected results
    out_file = os.path.join(results_dir, f"{system_name}_results_corrected.json")
    with open(out_file, 'w') as f:
        json.dump(corrected, f, indent=2)
    print(f"\n  ✓ Saved corrected results: {out_file}")

    return corrected


def main():
    """Process all available CCSD checkpoints."""

    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'cloud_checkpoints')
    results_dir = os.path.join(os.path.dirname(__file__), 'cloud_results')

    # Find all completed checkpoints (exclude .gstmp partial downloads)
    pattern = os.path.join(checkpoint_dir, '*_checkpoint_ccsd.pkl')
    checkpoints = sorted(glob.glob(pattern))
    # Filter out temp files
    checkpoints = [c for c in checkpoints if not c.endswith('.gstmp')]

    if not checkpoints:
        print("No completed CCSD checkpoints found.")
        print(f"Looking in: {checkpoint_dir}")
        print(f"Pattern:    *_checkpoint_ccsd.pkl")

        # Check for partial downloads
        partial = glob.glob(os.path.join(checkpoint_dir, '*.gstmp'))
        if partial:
            print(f"\nFound {len(partial)} partial downloads (.gstmp):")
            for p in partial:
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"  {os.path.basename(p)} ({size_mb:.0f} MB)")
            print("\nWait for downloads to complete and re-run.")
        sys.exit(1)

    print("=" * 60)
    print("  FBOND — NOON Metric Re-Extraction from CCSD Checkpoints")
    print("=" * 60)
    print(f"  Found {len(checkpoints)} completed checkpoints")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Results dir:    {results_dir}")
    print("=" * 60)

    all_results = []
    for ckpt_path in checkpoints:
        result = process_checkpoint(ckpt_path, results_dir)
        if result:
            all_results.append(result)

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print("  SUMMARY: f_e / N_D for all systems (def2-SVP)")
        print("=" * 80)
        print(f"  {'System':<25s} {'N_D':>8s} {'f_e':>8s} {'S_E_max':>10s} "
              f"{'M_frac':>8s} {'F_bond(old)':>12s}")
        print("  " + "─" * 78)

        # Sort by f_e descending
        all_results.sort(key=lambda x: x.get('f_e', 0), reverse=True)

        for r in all_results:
            f_bond_old = f"{r.get('F_bond', float('nan')):.4f}" if 'F_bond' in r else "N/A"
            print(f"  {r['system']:<25s} "
                  f"{r['N_D']:8.4f} "
                  f"{r['f_e']:8.4f} "
                  f"{r['S_E_max_noon']:10.6f} "
                  f"{r['M_fractional']:4d}/{r['M_corr_orbitals']:<3d} "
                  f"{f_bond_old:>12s}")

        print("  " + "─" * 78)
        print(f"\n  Systems processed: {len(all_results)}/{len(checkpoints)}")

        # Save combined corrected results
        combined_file = os.path.join(results_dir, 'fbond_results_corrected.json')
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Combined corrected results: {combined_file}")

    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
