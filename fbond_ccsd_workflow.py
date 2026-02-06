#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fbond workflow using HF + CCSD natural orbitals

- HF → O_MOS from canonical orbitals
- CCSD → 1-RDM → natural occupations → S_E,max
- Fbond = 0.5 * O_MOS * S_E,max

Designed to be consistent with v2.0 Fbond (frozen-core FCI) but using CCSD
for large systems (e.g. Cs3Al8-, Cs3Al12-).
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from pyscf import gto, scf, cc
from scipy.linalg import eigh
import json
from datetime import datetime


@dataclass
class FbondResult:
    name: str
    formula: str
    basis: str
    charge: int
    spin: int
    nelec: int
    nao: int
    frozen: Optional[int]

    E_HF: float
    E_CCSD: float

    epsilon_HOMO: float
    epsilon_LUMO: float
    O_MOS: float

    S_E_max: float
    Fbond: float

    S_entropies: np.ndarray
    nat_occ: np.ndarray

    timestamp: str


def build_molecule(name: str,
                   geom_str: str,
                   charge: int,
                   spin: int,
                   basis: str,
                   ecp: Optional[Dict[str, str]] = None) -> gto.Mole:
    """
    Build PySCF molecule from XYZ-style string.

    geom_str: multiline string "El x y z" in Angstrom
    basis: e.g. 'sto-3g', 'def2-TZVP'
    ecp: optional {'Cs': 'def2-TZVP', ...} for heavy atoms
    """
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = geom_str
    mol.unit = "Angstrom"
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    if ecp is not None:
        mol.ecp = ecp
    mol.build()
    print(f"\n=== Molecule: {name} ===")
    print(f"Formula guess: {mol.atom_symbol(0)} ...")
    print(f"nelec = {mol.nelectron}, nao = {mol.nao}")
    return mol


def hartree_fock(mol: gto.Mole) -> scf.RHF:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    E_HF = mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF did not converge")
    print(f"E(HF) = {E_HF:.10f} Ha")
    return mf


def homo_lumo_gap(mf: scf.hf.RHF, mol: gto.Mole):
    nelec = mol.nelectron
    eps = mf.mo_energy
    homo_idx = nelec // 2 - 1
    lumo_idx = nelec // 2
    eps_h = eps[homo_idx]
    eps_l = eps[lumo_idx]
    gap = eps_l - eps_h
    print("\n--- Orbital energetics (HF canonical) ---")
    print(f"ε_HOMO = {eps_h:.6f} Ha (idx {homo_idx})")
    print(f"ε_LUMO = {eps_l:.6f} Ha (idx {lumo_idx})")
    print(f"O_MOS  = {gap:.6f} Ha")
    return eps_h, eps_l, gap


def run_ccsd(mf: scf.hf.RHF,
             frozen: Optional[int] = None) -> cc.CCSD:
    """
    Run closed-shell CCSD. 'frozen' is number of lowest-energy MOs to freeze.
    For systems with ECP on Cs, you might not need a large frozen value;
    for main-group, you can emulate your SI frozen-core scheme.
    """
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-8
    mycc.max_cycle = 100
    E_corr, t1, t2 = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    E_tot = mycc.e_tot
    print("\n--- CCSD ---")
    print(f"E(CCSD) = {E_tot:.10f} Ha")
    print(f"E_corr  = {E_corr*1000:.3f} mHa")
    return mycc


def natural_occupations_from_ccsd(mycc: cc.CCSD,
                                  mol: gto.Mole,
                                  frozen: Optional[int] = None):
    """
    Build 1-RDM in MO basis (active space only if frozen != None),
    diagonalize to get natural occupations.
    """
    # Full 1-RDM in MO basis
    dm1_mo = mycc.make_rdm1()  # (nmo, nmo)

    # If 'frozen' is integer number of lowest-energy MOs, restrict to active
    if frozen is not None and frozen > 0:
        dm1_active = dm1_mo[frozen:, frozen:]
        print(f"Using active space: nmo_active = {dm1_active.shape[0]} (frozen = {frozen})")
        occ, U = eigh(dm1_active)
    else:
        occ, U = eigh(dm1_mo)

    # Sort descending
    occ = occ[::-1]
    print("\n--- Natural occupations (first 10) ---")
    for i, n in enumerate(occ[:10]):
        dev = abs(n - 2.0) if n > 1.0 else abs(n)
        print(f"NO {i+1:3d}: n = {n:8.5f}, dev = {dev:8.5f}")
    return occ


def single_orbital_entropies(occ: np.ndarray) -> np.ndarray:
    """
    Von Neumann single-orbital entropies from occupations n_i (0..2):
    S_i = -p ln p - (1-p) ln (1-p), p = n_i/2
    """
    S = []
    for n in occ:
        if 0.0 < n < 2.0:
            p = n / 2.0
            if 0.0 < p < 1.0:
                S_i = -p * np.log(p) - (1 - p) * np.log(1 - p)
            else:
                S_i = 0.0
        else:
            S_i = 0.0
        S.append(S_i)
    S = np.array(S)
    print("\n--- Entanglement entropies (first 10 orbitals) ---")
    for i, s in enumerate(S[:10]):
        print(f"Orb {i+1:3d}: S_E = {s:8.5f} nats")
    print(f"\nS_E,max = {S.max():.6f} nats")
    return S


def fbond_from_gap_and_entropy(O_MOS: float, S_E_max: float,
                               N: float = 0.5) -> float:
    F = N * O_MOS * S_E_max
    print("\n--- Fbond ---")
    print(f"Fbond = {N:.3f} * {O_MOS:.6f} * {S_E_max:.6f} = {F:.6f} Ha^-1")
    return F


def run_fbond_workflow(name: str,
                       formula: str,
                       geom_str: str,
                       charge: int,
                       spin: int,
                       basis: str,
                       ecp: Optional[Dict[str, str]] = None,
                       frozen: Optional[int] = None) -> FbondResult:
    """
    High-level driver: HF → CCSD → NO occupations → S_E,max → Fbond.
    """
    mol = build_molecule(name, geom_str, charge, spin, basis, ecp)
    mf = hartree_fock(mol)
    eps_h, eps_l, O_MOS = homo_lumo_gap(mf, mol)
    mycc = run_ccsd(mf, frozen=frozen)
    occ = natural_occupations_from_ccsd(mycc, mol, frozen=frozen)
    S_list = single_orbital_entropies(occ)
    S_E_max = float(S_list.max())
    Fbond = fbond_from_gap_and_entropy(O_MOS, S_E_max, N=0.5)

    res = FbondResult(
        name=name,
        formula=formula,
        basis=basis,
        charge=charge,
        spin=spin,
        nelec=mol.nelectron,
        nao=mol.nao,
        frozen=frozen,
        E_HF=float(mf.e_tot),
        E_CCSD=float(mycc.e_tot),
        epsilon_HOMO=float(eps_h),
        epsilon_LUMO=float(eps_l),
        O_MOS=float(O_MOS),
        S_E_max=S_E_max,
        Fbond=float(Fbond),
        S_entropies=S_list,
        nat_occ=occ,
        timestamp=datetime.now().isoformat()
    )
    return res


def save_fbond_result(res: FbondResult, fname_prefix: str):
    data = asdict(res)
    # Convert numpy arrays to lists for JSON
    data["S_entropies"] = [float(x) for x in res.S_entropies]
    data["nat_occ"] = [float(x) for x in res.nat_occ]
    json_name = f"{fname_prefix}_fbond.json"
    with open(json_name, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[+] Saved Fbond result to {json_name}")


# ------------------- EXAMPLE USAGE TEMPLATES -------------------

if __name__ == "__main__":
    # 1) H2 reference (STO-3G)
    geom_H2 = """
    H 0.0000 0.0000 0.0000
    H 0.0000 0.0000 0.7414
    """
    res_H2 = run_fbond_workflow(
        name="H2",
        formula="H2",
        geom_str=geom_H2,
        charge=0,
        spin=0,
        basis="sto-3g",
        ecp=None,
        frozen=None
    )
    save_fbond_result(res_H2, "H2_sto3g_CCSD")

    # 2) Cs3Al8- with cc-pVDZ + def2-ECP (as stated in manuscript)
geom_Cs3Al8 = """
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

res_Cs3Al8 = run_fbond_workflow(
    name="Cs3Al8-",
    formula="Cs3Al8-",
    geom_str=geom_Cs3Al8,
    charge=-1,
    spin=0,
    basis={"Al": "cc-pvdz", "Cs": "cc-pvdz"},
    ecp={"Cs": "def2-ecp"},     # Stuttgart-Dresden ECP for Cs
    frozen=8                     # Freeze 8 Al 1s core orbitals
)
save_fbond_result(res_Cs3Al8, "Cs3Al8_ccpvdz_CCSD")

# 3) Cs3Al12- (add when geometry optimization finishes)
# geom_Cs3Al12 = """
# Al ...
# Cs ...
# """
# res_Cs3Al12 = run_fbond_workflow(
#     name="Cs3Al12-",
#     formula="Cs3Al12-",
#     geom_str=geom_Cs3Al12,
#     charge=-1,
#     spin=0,
#     basis={"Al": "cc-pvdz", "Cs": "cc-pvdz"},
#     ecp={"Cs": "def2-ecp"},
#     frozen=12  # Freeze 12 Al 1s core orbitals
# )
# save_fbond_result(res_Cs3Al12, "Cs3Al12_ccpvdz_CCSD")
