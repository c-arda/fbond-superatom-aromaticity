#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

# ---------- 1. Initial guess geometry (ROUGH, YOU CAN ADJUST) ----------

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

# ---------- 2. Build molecule with DFT settings ----------

mol = gto.M(
    atom=geom_Cs3Al8_guess,
    unit="Angstrom",
    charge=-1,
    spin=0,
    basis={"Al": "def2-SVP", "Cs": "def2-SVP"},
    ecp={"Cs": "def2-SVP"},
    verbose=4,
)

mf = dft.RKS(mol)
mf.xc = "b3lyp"   # or "pbe0" if you prefer
mf.conv_tol = 1e-8
mf.max_cycle = 200


# ---------- 3. Geometry optimization ----------

mol_opt = optimize(mf)  # returns an optimized Mole object

print("\n=== Optimized geometry: Cs3Al8- (wB97X-D/def2-SVP) ===")
coords = mol_opt.atom_coords(unit="Angstrom")
symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
for s, (x, y, z) in zip(symbols, coords):
    print(f"{s:2s}  {x:10.6f}  {y:10.6f}  {z:10.6f}")

print("\nTotal energy at optimized geometry:", mf.e_tot)
