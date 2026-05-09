#!/usr/bin/env python3
"""Minimal Au13- debug test — run inside the Docker container to find the crash."""
import sys, os, traceback

print("=== Au13- Debug Test ===")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except ImportError as e:
    print(f"FAIL numpy: {e}")
    sys.exit(1)

try:
    import pyscf
    print(f"pyscf: {pyscf.__version__}")
    print(f"pyscf path: {pyscf.__file__}")
except ImportError as e:
    print(f"FAIL pyscf: {e}")
    sys.exit(1)

# Check if def2-SVP ECP data exists for Au
try:
    from pyscf import gto
    mol = gto.M(
        atom='Au 0 0 0',
        basis='def2-SVP',
        ecp='def2-SVP',
        charge=0,
        spin=1,
        verbose=3
    )
    print(f"\n✓ Single Au atom with def2-SVP ECP: {mol.nelectron} electrons")
except Exception as e:
    print(f"\n✗ FAIL: Single Au atom ECP test")
    traceback.print_exc()

# Try building full Au13-
try:
    import json
    geom_path = '/app/geometries/Au13_minus.json'
    if not os.path.exists(geom_path):
        geom_path = 'geometries/Au13_minus.json'
    
    geom = json.load(open(geom_path))
    geometry = geom['geometry']
    
    elements = set()
    element_counts = {}
    for line in geometry.strip().split('\n'):
        parts = line.strip().split()
        if parts:
            el = parts[0]
            elements.add(el)
            element_counts[el] = element_counts.get(el, 0) + 1
    
    basis_dict = {el: 'def2-SVP' for el in elements}
    ecp_dict = {'Au': 'def2-SVP'}
    
    mol = gto.M(
        atom=geometry,
        basis=basis_dict,
        ecp=ecp_dict,
        charge=-1,
        spin=0,
        verbose=4,
        max_memory=200000
    )
    print(f"\n✓ Au13- molecule: {mol.nelectron} electrons, {mol.nao} AOs")
    
    # Try HF
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    # Don't actually run — just check setup
    print(f"✓ RHF object created")
    
    # Try CCSD setup with frozen core
    from pyscf import cc
    FROZEN_PER_ATOM = {'Au': 4}
    n_frozen = sum(FROZEN_PER_ATOM.get(el, 0) * count
                   for el, count in element_counts.items())
    print(f"  n_frozen = {n_frozen}")
    
    # Check if checkpoint exists
    ckpt_dir = '/mnt/disks/checkpoints'
    if not os.path.exists(ckpt_dir):
        ckpt_dir = '.'
    
    hf_ckpt_path = os.path.join(ckpt_dir, 'Au13_minus_checkpoint_hf.pkl')
    print(f"  HF checkpoint path: {hf_ckpt_path}")
    print(f"  Exists: {os.path.exists(hf_ckpt_path)}")
    
    if os.path.exists(hf_ckpt_path):
        import pickle
        ckpt = pickle.load(open(hf_ckpt_path, 'rb'))
        print(f"  HF checkpoint loaded: E={ckpt['e_tot']:.6f}")
        mf.mo_coeff = ckpt['mo_coeff']
        mf.mo_energy = ckpt['mo_energy']
        mf.mo_occ = ckpt['mo_occ']
        mf.e_tot = ckpt['e_tot']
        mf.converged = True
    
    # Try CCSD creation
    mycc = cc.CCSD(mf, frozen=n_frozen)
    mycc.conv_tol = 1e-8
    print(f"✓ CCSD object created with frozen={n_frozen}")
    print(f"  nocc = {mycc.nocc}, nvir = {mycc.nmo - mycc.nocc}")
    
    print("\n=== ALL CHECKS PASSED ===")
    
except Exception as e:
    print(f"\n✗ FAIL: Au13- test")
    traceback.print_exc()
    sys.exit(1)
