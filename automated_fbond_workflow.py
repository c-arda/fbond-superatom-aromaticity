#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Automated F_bond Analysis & Figure Generation Workflow (BATCH-READY)

Modes:
  Single system:  python automated_fbond_workflow.py --system-file geometries/Cs3Al8.json
  All embedded:   python automated_fbond_workflow.py --all
  Batch (Docker): python automated_fbond_workflow.py --system-file /input/mol.json \
                      --output-dir /output --checkpoint-dir /checkpoints \
                      --scratch-dir /scratch --max-memory 120000

CHECKPOINT FEATURES:
  - Saves after HF convergence
  - Saves after CCSD convergence (The expensive part!)
  - Can resume from last checkpoint if interrupted

Generates:
  1. XYZ files for structure visualization
  2. Cube files for orbital plots
  3. F_bond, O_MOS, S_E,max values in JSON
  4. LaTeX-ready table rows
'''

import argparse
import json
import os
import sys
import numpy as np
import pickle
from pathlib import Path

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_MAX_MEMORY_MB = 4000   # PySCF default; override for cloud VMs!
DEFAULT_BASIS = 'def2-SVP'
DEFAULT_SCRATCH = os.path.expanduser('~/scratch/pyscf_temp')

# Embedded systems registry (backward compatibility with --all mode)
SYSTEMS_REGISTRY = {
    'Cs3Al8': {
        'geometry': '''
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
        ''',
        'charge': -1,
        'spin': 0,
    },
    'Cs3Al12': {
        'geometry': '''
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
        ''',
        'charge': -1,
        'spin': 0,
    }
}


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def configure_environment(args):
    '''Set PySCF environment variables before importing pyscf.'''
    # Scratch directory (local SSD on cloud, ~/scratch locally)
    scratch = args.scratch_dir or DEFAULT_SCRATCH
    os.makedirs(scratch, exist_ok=True)
    os.environ['PYSCF_TMPDIR'] = scratch

    # Memory limit — CRITICAL for cloud VMs
    # PySCF defaults to 4 GB if PYSCF_MAX_MEMORY is not set.
    # On a c2-highmem-16 (128 GB), this wastes 97% of available RAM.
    max_mem = args.max_memory or _auto_detect_memory()
    os.environ['PYSCF_MAX_MEMORY'] = str(max_mem)

    # Thread control for MKL / OpenBLAS
    n_threads = str(os.cpu_count() or 1)
    os.environ.setdefault('OMP_NUM_THREADS', n_threads)
    os.environ.setdefault('MKL_NUM_THREADS', n_threads)

    print(f"[ENV] PYSCF_TMPDIR     = {scratch}")
    print(f"[ENV] PYSCF_MAX_MEMORY = {max_mem} MB")
    print(f"[ENV] OMP_NUM_THREADS  = {os.environ['OMP_NUM_THREADS']}")
    print(f"[ENV] MKL_NUM_THREADS  = {os.environ['MKL_NUM_THREADS']}")


def _auto_detect_memory():
    '''Auto-detect available RAM and reserve ~8 GB for OS.'''
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    total_kb = int(line.split()[1])
                    total_mb = total_kb // 1024
                    usable = max(total_mb - 8000, DEFAULT_MAX_MEMORY_MB)
                    print(f"[MEM] Detected {total_mb} MB total, allocating {usable} MB to PySCF")
                    return usable
    except (FileNotFoundError, ValueError):
        pass
    return DEFAULT_MAX_MEMORY_MB


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

_checkpoint_dir = '.'  # Set by configure_paths()


def _checkpoint_path(system_name, stage):
    return os.path.join(_checkpoint_dir, f"{system_name}_checkpoint_{stage}.pkl")


def save_checkpoint(system_name, stage, data):
    '''Save checkpoint data to disk.'''
    path = _checkpoint_path(system_name, stage)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[✓] CHECKPOINT: Saved {stage} data to {path} ({size_mb:.1f} MB)")


def load_checkpoint(system_name, stage):
    '''Load checkpoint data if it exists.'''
    path = _checkpoint_path(system_name, stage)
    if Path(path).exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"[✓] RESUME: Loaded {stage} checkpoint from {path}")
        return data
    return None


def checkpoint_exists(system_name, stage):
    '''Check if checkpoint exists.'''
    return Path(_checkpoint_path(system_name, stage)).exists()


# ============================================================================
# STEP 1: FBOND CALCULATION (WITH CHECKPOINTS)
# ============================================================================

def calculate_fbond(system_name, geometry, charge, spin, basis=DEFAULT_BASIS):
    '''Run HF+CCSD and compute F_bond metrics with checkpoint support.'''
    from pyscf import gto, scf, cc

    print(f"\n{'='*70}")
    print(f"Processing: {system_name}")
    print(f"  Basis: {basis}  |  Charge: {charge}  |  Spin: {spin}")
    print(f"{'='*70}")

    # Build molecule
    # Determine unique elements for basis/ECP assignment
    elements = set()
    for line in geometry.strip().split('\n'):
        parts = line.strip().split()
        if parts:
            elements.add(parts[0])

    basis_dict = {el: basis for el in elements}
    ecp_dict = {}
    # Apply ECP for heavy elements (Cs, Rb, etc.)
    heavy_elements = {'Cs', 'Rb', 'Ba', 'Sr', 'K'}
    for el in elements & heavy_elements:
        ecp_dict[el] = basis

    mol = gto.M(
        atom=geometry,
        basis=basis_dict,
        ecp=ecp_dict if ecp_dict else None,
        charge=charge,
        spin=spin,
        verbose=4,
        max_memory=int(os.environ.get('PYSCF_MAX_MEMORY', DEFAULT_MAX_MEMORY_MB))
    )

    n_al = geometry.upper().count('\nAL') + (1 if geometry.strip().upper().startswith('AL') else 0)
    # Count Al atoms more robustly
    n_al = sum(1 for line in geometry.strip().split('\n')
               if line.strip().split()[0].upper() == 'AL') if geometry.strip() else 0
    n_frozen = n_al

    # ========================================================================
    # STAGE 1: HF CALCULATION (or resume from checkpoint)
    # ========================================================================

    hf_checkpoint = load_checkpoint(system_name, 'hf')

    if hf_checkpoint is not None:
        print("\n[RESUME] Using saved HF results from checkpoint")
        mf = scf.RHF(mol)
        mf.mo_coeff = hf_checkpoint['mo_coeff']
        mf.mo_energy = hf_checkpoint['mo_energy']
        mf.mo_occ = hf_checkpoint['mo_occ']
        mf.e_tot = hf_checkpoint['e_tot']
        mf.converged = True
    else:
        print("\n[1/3] Running Hartree-Fock...")
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.kernel()

        if not mf.converged:
            raise RuntimeError(f"HF did not converge for {system_name}")

        # Save HF checkpoint
        hf_data = {
            'mo_coeff': mf.mo_coeff,
            'mo_energy': mf.mo_energy,
            'mo_occ': mf.mo_occ,
            'e_tot': mf.e_tot
        }
        save_checkpoint(system_name, 'hf', hf_data)

    # ========================================================================
    # STAGE 2: CCSD CALCULATION (or resume from checkpoint)
    # ========================================================================

    ccsd_checkpoint = load_checkpoint(system_name, 'ccsd')

    if ccsd_checkpoint is not None:
        print("\n[RESUME] Using saved CCSD results from checkpoint")
        mycc = cc.CCSD(mf, frozen=n_frozen)
        mycc.e_corr = ccsd_checkpoint['e_corr']
        mycc.e_tot = ccsd_checkpoint['e_tot']
        mycc.t1 = ccsd_checkpoint['t1']
        mycc.t2 = ccsd_checkpoint['t2']
        mycc.converged = True
        dm1 = ccsd_checkpoint['dm1']
    else:
        print(f"\n[2/3] Running CCSD with {n_frozen} frozen core orbitals...")
        mycc = cc.CCSD(mf, frozen=n_frozen)
        mycc.conv_tol = 1e-8
        mycc.kernel()

        if not mycc.converged:
            raise RuntimeError(f"CCSD did not converge for {system_name}")

        # Compute density matrix
        dm1 = mycc.make_rdm1()

        # Save CCSD checkpoint (THE MOST IMPORTANT ONE - 20 hours of work!)
        ccsd_data = {
            'e_corr': mycc.e_corr,
            'e_tot': mycc.e_tot,
            't1': mycc.t1,
            't2': mycc.t2,
            'dm1': dm1
        }
        save_checkpoint(system_name, 'ccsd', ccsd_data)

    # ========================================================================
    # STAGE 3: NATURAL ORBITALS & F_BOND ANALYSIS
    # ========================================================================

    print("\n[3/3] Computing natural orbitals and F_bond...")

    # Diagonalize 1-RDM to get natural occupations
    natocc, natorb = np.linalg.eigh(dm1)
    natocc = natocc[::-1]
    natorb = natorb[:, ::-1]

    # Transform to AO basis
    natorb_ao = mf.mo_coeff @ natorb

    # HOMO-LUMO gap from HF
    nelec = mol.nelectron
    homo_idx_hf = nelec // 2 - 1
    lumo_idx_hf = nelec // 2
    eps_homo = mf.mo_energy[homo_idx_hf]
    eps_lumo = mf.mo_energy[lumo_idx_hf]
    O_MOS = eps_lumo - eps_homo

    # Entanglement entropy
    def entropy(n):
        if n <= 0 or n >= 2:
            return 0.0
        p = n / 2.0
        if 0 < p < 1:
            return -p * np.log(p) - (1 - p) * np.log(1 - p)
        return 0.0

    # CORRECTED F_BOND FORMULA (Manuscript Eq 1)
    S_E = np.array([entropy(n) for n in natocc])
    n_eff_model = 1.0 + np.exp(-O_MOS) / (1.0 + np.exp(-O_MOS))
    S_E_max = entropy(n_eff_model)
    F_bond = 0.5 * O_MOS * S_E_max

    # Find frontier natural orbitals
    threshold_upper = 1.98
    homo_no_idx = None
    lumo_no_idx = None

    for i, n in enumerate(natocc):
        if n < threshold_upper and homo_no_idx is None:
            homo_no_idx = i
        if n < 1.0 and lumo_no_idx is None:
            lumo_no_idx = i
            break

    if homo_no_idx is None:
        homo_no_idx = nelec // 2 - 1
    if lumo_no_idx is None:
        lumo_no_idx = homo_no_idx + 1

    # Save natural orbital data checkpoint
    no_data = {
        'natocc': natocc,
        'natorb_ao': natorb_ao,
        'homo_no_idx': homo_no_idx,
        'lumo_no_idx': lumo_no_idx
    }
    save_checkpoint(system_name, 'natural_orbitals', no_data)

    results = {
        'system': system_name,
        'n_electrons': mol.nelectron,
        'n_atoms': mol.natm,
        'n_frozen': n_frozen,
        'basis': basis,
        'E_HF': float(mf.e_tot),
        'E_CCSD': float(mycc.e_tot),
        'E_corr': float(mycc.e_corr),
        'eps_HOMO': float(eps_homo),
        'eps_LUMO': float(eps_lumo),
        'O_MOS': float(O_MOS),
        'S_E_max': float(S_E_max),
        'F_bond': float(F_bond),
        'natural_occupations': natocc.tolist()[:20],
        'entropy_values': S_E.tolist()[:20],
        'homo_no_idx': int(homo_no_idx),
        'lumo_no_idx': int(lumo_no_idx),
        'max_memory_mb': int(os.environ.get('PYSCF_MAX_MEMORY', DEFAULT_MAX_MEMORY_MB)),
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {system_name}")
    print(f"{'='*70}")
    print(f"  Basis         = {basis}")
    print(f"  E(HF)         = {mf.e_tot:.8f} Ha")
    print(f"  E(CCSD)       = {mycc.e_tot:.8f} Ha")
    print(f"  E_corr        = {mycc.e_corr*1000:.3f} mHa")
    print(f"  ε_HOMO        = {eps_homo:.6f} Ha")
    print(f"  ε_LUMO        = {eps_lumo:.6f} Ha")
    print(f"  O_MOS         = {O_MOS:.6f} Ha")
    print(f"  S_E,max       = {S_E_max:.6f} nats")
    print(f"  F_bond        = {F_bond:.6f}")
    print(f"  PYSCF_MAX_MEM = {os.environ.get('PYSCF_MAX_MEMORY', '?')} MB")
    print(f"{'='*70}\n")

    return results, mol, mf, natorb_ao, homo_no_idx, lumo_no_idx


# ============================================================================
# STEP 2: GENERATE XYZ FILES FOR VMD/AVOGADRO
# ============================================================================

def save_xyz(system_name, geometry, output_dir='.'):
    '''Save geometry as XYZ file for structure visualization.'''

    filename = os.path.join(output_dir, f"{system_name}_structure.xyz")

    if Path(filename).exists():
        print(f"[✓] XYZ file already exists: {filename}")
        return filename

    lines = [l.strip() for l in geometry.strip().split('\n') if l.strip()]
    n_atoms = len(lines)

    with open(filename, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{system_name} optimized geometry\n")
        for line in lines:
            f.write(line + '\n')

    print(f"[✓] Saved structure: {filename}")
    return filename


# ============================================================================
# STEP 3: GENERATE CUBE FILES FOR ORBITAL VISUALIZATION
# ============================================================================

def save_orbital_cubes(system_name, mol, mf, natorb_ao, homo_idx, lumo_idx, output_dir='.'):
    '''Generate cube files for frontier natural orbitals.'''
    from pyscf import tools

    print(f"\n[Generating orbital cube files for {system_name}...]")

    homo_file = os.path.join(output_dir, f"{system_name}_HOMO_NO.cube")
    if not Path(homo_file).exists():
        tools.cubegen.orbital(mol, homo_file, natorb_ao[:, homo_idx])
        print(f"[✓] Saved HOMO-like natural orbital: {homo_file}")
    else:
        print(f"[✓] HOMO cube already exists: {homo_file}")

    lumo_file = os.path.join(output_dir, f"{system_name}_LUMO_NO.cube")
    if not Path(lumo_file).exists():
        tools.cubegen.orbital(mol, lumo_file, natorb_ao[:, lumo_idx])
        print(f"[✓] Saved LUMO-like natural orbital: {lumo_file}")
    else:
        print(f"[✓] LUMO cube already exists: {lumo_file}")

    return homo_file, lumo_file


# ============================================================================
# STEP 4: GENERATE LATEX TABLE ROWS
# ============================================================================

def generate_latex_table(all_results):
    '''Create LaTeX table rows ready to paste.'''

    print("\n" + "="*70)
    print("LATEX TABLE (paste into Table 1 in your .tex file):")
    print("="*70)
    print("System & O_MOS (Ha) & S_E,max (nats) & F_bond \\\\")
    print("\\hline")

    for res in all_results:
        row = (f"{res['system']} & "
               f"{res['O_MOS']:.4f} & "
               f"{res['S_E_max']:.4f} & "
               f"{res['F_bond']:.4f} \\\\")
        print(row)

    print("="*70)


# ============================================================================
# SYSTEM LOADING
# ============================================================================

def load_system_from_file(filepath):
    '''Load a single system from a geometry JSON file.'''
    with open(filepath) as f:
        data = json.load(f)

    required = ['name', 'geometry', 'charge', 'spin']
    for key in required:
        if key not in data:
            raise ValueError(f"Geometry JSON missing required key: '{key}' in {filepath}")

    return data


# ============================================================================
# SINGLE-SYSTEM WORKFLOW
# ============================================================================

def run_single_system(system_data, args, output_dir):
    '''Run the full F_bond workflow for one system.'''

    name = system_data['name']
    geometry = system_data['geometry']
    charge = system_data['charge']
    spin = system_data['spin']
    basis = system_data.get('basis', args.basis or DEFAULT_BASIS)

    os.makedirs(output_dir, exist_ok=True)

    # Calculate F_bond
    results, mol, mf, natorb_ao, homo_idx, lumo_idx = calculate_fbond(
        name, geometry, charge, spin, basis=basis
    )

    # Save XYZ
    save_xyz(name, geometry, output_dir)

    # Save cube files
    save_orbital_cubes(name, mol, mf, natorb_ao, homo_idx, lumo_idx, output_dir)

    # Save individual results JSON
    results_file = os.path.join(output_dir, f"{name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Results saved to: {results_file}")

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='F_bond Quantum Chemical Bonding Analysis — Batch-Ready CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single system from JSON file
  python %(prog)s --system-file geometries/Cs3Al8.json

  # All embedded systems (backward compatible)
  python %(prog)s --all

  # Cloud batch mode (Docker entrypoint)
  python %(prog)s --system-file /input/mol.json \\
      --output-dir /output \\
      --checkpoint-dir /checkpoints \\
      --scratch-dir /scratch \\
      --max-memory 120000 \\
      --basis def2-TZVP
'''
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--system-file', type=str,
                      help='Path to a geometry JSON file (single-system mode)')
    mode.add_argument('--all', action='store_true',
                      help='Run all systems in the embedded SYSTEMS_REGISTRY')

    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files (default: cwd)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for checkpoint files (default: output-dir)')
    parser.add_argument('--scratch-dir', type=str, default=None,
                        help='Scratch directory for PySCF temp files '
                             '(default: ~/scratch/pyscf_temp). '
                             'Use local NVMe SSD on cloud for best performance.')
    parser.add_argument('--basis', type=str, default=None,
                        help=f'Override basis set (default: from JSON or {DEFAULT_BASIS})')
    parser.add_argument('--max-memory', type=int, default=None,
                        help='PySCF max memory in MB (default: auto-detect from system RAM). '
                             'Set to ~120000 for c2-highmem-16 (128GB), '
                             '~500000 for n2-highmem-64 (512GB).')

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Configure environment BEFORE importing PySCF
    configure_environment(args)

    # Set checkpoint directory
    global _checkpoint_dir
    _checkpoint_dir = args.checkpoint_dir or args.output_dir
    os.makedirs(_checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("F_BOND WORKFLOW — BATCH-READY")
    print("="*70)
    print("Checkpoint stages:")
    print("  [1] HF convergence (~5 min)")
    print("  [2] CCSD convergence (~4–20 hours depending on hardware)")
    print("  [3] Natural orbital analysis (~5 min)")
    print("\nIf interrupted, re-run with same args to resume from last checkpoint!")
    print("="*70 + "\n")

    all_results = []

    if args.system_file:
        # ── Single-system mode ──
        system_data = load_system_from_file(args.system_file)
        print(f"[MODE] Single system: {system_data['name']}")
        print(f"[FILE] {args.system_file}\n")

        try:
            results = run_single_system(system_data, args, args.output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {system_data['name']}: {e}\n")
            print("[INFO] Checkpoints have been saved. Re-run to resume.")
            sys.exit(1)

    elif args.all:
        # ── All-systems mode (backward compatible) ──
        print(f"[MODE] All systems ({len(SYSTEMS_REGISTRY)} registered)\n")

        for system_name, config in SYSTEMS_REGISTRY.items():
            if 'PASTE' in config['geometry']:
                print(f"\n[SKIP] {system_name}: Geometry not provided yet.\n")
                continue

            # Build system_data dict
            system_data = {
                'name': system_name,
                'geometry': config['geometry'],
                'charge': config['charge'],
                'spin': config['spin'],
            }

            # Check existing checkpoints
            has_hf = checkpoint_exists(system_name, 'hf')
            has_ccsd = checkpoint_exists(system_name, 'ccsd')
            has_no = checkpoint_exists(system_name, 'natural_orbitals')

            print(f"\n[CHECKPOINT STATUS] {system_name}:")
            print(f"  HF checkpoint:      {'✓ FOUND' if has_hf else '✗ Missing'}")
            print(f"  CCSD checkpoint:    {'✓ FOUND' if has_ccsd else '✗ Missing'}")
            print(f"  NatOrb checkpoint:  {'✓ FOUND' if has_no else '✗ Missing'}")
            print()

            try:
                results = run_single_system(system_data, args, args.output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"\n[ERROR] Failed to process {system_name}: {e}\n")
                print("[INFO] Checkpoints have been saved. Re-run to resume.")
                continue

    # Save combined JSON summary
    if all_results:
        combined_file = os.path.join(args.output_dir, 'fbond_results.json')
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[✓] Combined results saved to: {combined_file}")

        generate_latex_table(all_results)

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Checkpoint directory: {_checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
