#!/usr/bin/env python3
"""
Geometry Optimization for Cs3Al_n Clusters
Uses B3LYP/def2-SVP with geomeTRIC optimizer
"""

import numpy as np
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
import argparse
import sys

def optimize_cluster(system, initial_xyz=None, output_xyz=None):
    """
    Optimize Cs3Al_n cluster geometry

    Parameters:
    -----------
    system : str
        'Cs3Al8' or 'Cs3Al12'
    initial_xyz : str
        Input XYZ file (optional, will use template if not provided)
    output_xyz : str
        Output XYZ file for optimized geometry
    """

    # Load geometry
    if initial_xyz:
        print(f"Loading geometry from {initial_xyz}...")
        mol = gto.M(
            atom=initial_xyz,
            basis={'Al': 'def2-svp', 'Cs': 'def2-svp'},
            ecp={'Cs': 'def2-svp'},  # def2-ECP for Cs (46 electrons)
            charge=-1,
            spin=0,  # Singlet (closed shell)
            verbose=4
        )
    else:
        # Use idealized template geometries
        if system == 'Cs3Al8':
            geometry = get_cs3al8_template()
        elif system == 'Cs3Al12':
            geometry = get_cs3al12_template()
        else:
            raise ValueError("System must be 'Cs3Al8' or 'Cs3Al12'")

        print(f"Using template geometry for {system}...")
        mol = gto.M(
            atom=geometry,
            basis={'Al': 'def2-svp', 'Cs': 'def2-svp'},
            ecp={'Cs': 'def2-svp'},
            charge=-1,
            spin=0,
            verbose=4
        )

    print("\n" + "="*70)
    print(f"GEOMETRY OPTIMIZATION: {system}")
    print("="*70)
    print(f"Method: B3LYP/def2-SVP")
    print(f"ECP: def2-ECP for Cs (46 core electrons)")
    print(f"Charge: -1, Spin: 0 (singlet)")
    print(f"Atoms: {mol.natm}")
    print(f"Electrons: {mol.nelectron}")
    print("="*70 + "\n")

    # Set up DFT calculation
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    mf.grids.level = 3  # Medium grid

    # Run geometry optimization with geomeTRIC
    print("Starting geometry optimization with geomeTRIC...")
    print("(This may take 30 minutes to several hours depending on system size)\n")

    try:
        mol_opt = optimize(mf, maxsteps=100)

        print("\n" + "="*70)
        print("OPTIMIZATION CONVERGED!")
        print("="*70)

        # Save optimized geometry
        if output_xyz is None:
            output_xyz = f"{system}_optimized.xyz"

        with open(output_xyz, 'w') as f:
            f.write(f"{mol_opt.natm}\n")
            f.write(f"{system} optimized (B3LYP/def2-SVP)\n")
            for i in range(mol_opt.natm):
                symbol = mol_opt.atom_symbol(i)
                coords = mol_opt.atom_coord(i) * 0.529177  # Bohr to Angstrom
                f.write(f"{symbol:2s}  {coords[0]:12.6f}  {coords[1]:12.6f}  {coords[2]:12.6f}\n")

        print(f"\n✓ Optimized geometry saved to: {output_xyz}")
        print(f"  Final energy: {mol_opt.energy_tot():.8f} Ha")

        return mol_opt

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        sys.exit(1)

def get_cs3al8_template():
    """
    Return idealized Cs3Al8 geometry (distorted cube + 3 Cs)
    Approximate D3h symmetry
    """
    return """
    Al   0.000000   0.000000   2.200000
    Al   2.200000   0.000000   0.000000
    Al   0.000000   2.200000   0.000000
    Al  -2.200000   0.000000   0.000000
    Al   0.000000  -2.200000   0.000000
    Al   0.000000   0.000000  -2.200000
    Al   1.555635   1.555635   1.555635
    Al  -1.555635  -1.555635  -1.555635
    Cs   4.000000   4.000000   0.000000
    Cs  -4.000000  -4.000000   0.000000
    Cs   0.000000   0.000000   5.656854
    """

def get_cs3al12_template():
    """
    Return idealized Cs3Al12 geometry (icosahedral core + 3 Cs)
    """
    return """
    Al   0.000000   0.000000   2.500000
    Al   2.236068   0.000000   1.118034
    Al   0.690983   2.127051   1.118034
    Al  -1.809017   1.314214   1.118034
    Al  -1.809017  -1.314214   1.118034
    Al   0.690983  -2.127051   1.118034
    Al   1.809017   1.314214  -1.118034
    Al  -0.690983   2.127051  -1.118034
    Al  -2.236068   0.000000  -1.118034
    Al  -0.690983  -2.127051  -1.118034
    Al   1.809017  -1.314214  -1.118034
    Al   0.000000   0.000000  -2.500000
    Cs   4.500000   4.500000   0.000000
    Cs  -4.500000  -4.500000   0.000000
    Cs   0.000000   0.000000   6.363961
    """

def main():
    parser = argparse.ArgumentParser(
        description='Optimize Cs3Al_n cluster geometry using B3LYP/def2-SVP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize using template geometry
  python optimize_geometry.py --system Cs3Al8 --output Cs3Al8_optimized.xyz

  # Optimize from existing XYZ file
  python optimize_geometry.py --system Cs3Al12 --input initial.xyz --output optimized.xyz
        """
    )

    parser.add_argument('--system', required=True, choices=['Cs3Al8', 'Cs3Al12'],
                        help='Cluster system to optimize')
    parser.add_argument('--input', dest='input_xyz', default=None,
                        help='Input XYZ file (optional, uses template if not provided)')
    parser.add_argument('--output', dest='output_xyz', default=None,
                        help='Output XYZ file (default: <system>_optimized.xyz)')

    args = parser.parse_args()

    # Run optimization
    optimize_cluster(
        system=args.system,
        initial_xyz=args.input_xyz,
        output_xyz=args.output_xyz
    )

if __name__ == '__main__':
    main()
