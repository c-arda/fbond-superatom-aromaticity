#!/usr/bin/env python3
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

geom_Cs3Al12_guess = """
Al   0.000   0.000   0.000
Al   2.500   0.000   0.000
Al  -2.500   0.000   0.000
Al   0.000   2.500   0.000
Al   0.000  -2.500   0.000
Al   0.000   0.000   2.500
Al   0.000   0.000  -2.500
Al   2.500   2.500   0.000
Al  -2.500  -2.500   0.000
Al   2.500   0.000   2.500
Al  -2.500   0.000  -2.500
Al   0.000   2.500   2.500
Cs   0.000   0.000   6.500
Cs   4.500   0.000   4.000
Cs  -4.500   0.000   4.000
"""

mol = gto.M(
    atom=geom_Cs3Al12_guess,
    unit="Angstrom",
    charge=-1,
    spin=0,
    basis={"Al": "def2-SVP", "Cs": "def2-SVP"},
    ecp={"Cs": "def2-SVP"},
    verbose=4,
)

mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.conv_tol = 1e-6          # Relaxed from 1e-8 (still tight enough for gradients)
mf.conv_tol_grad = 1e-5     # Add gradient convergence criterion
mf.max_cycle = 300          # More SCF cycles if needed
mf.diis_space = 12          # Larger DIIS space for better convergence
mf.level_shift = 0.2        # Add level shift to stabilize SCF in early cycles

mol_opt = optimize(mf, maxsteps=100)  # Allow more geom steps if needed

print("\n=== Optimized geometry: Cs3Al12- (B3LYP/def2-SVP) ===")
coords = mol_opt.atom_coords(unit="Angstrom")
symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
for s, (x, y, z) in zip(symbols, coords):
    print(f"{s:2s}  {x:10.6f}  {y:10.6f}  {z:10.6f}")
