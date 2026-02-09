# Fbond Analysis for Superatom Aromaticity

Analysis scripts and data for computing Fbond descriptors on Cs₃Al₈⁻ and Cs₃Al₁₂⁻ superatom clusters. This project investigates superatom aromaticity using Coherence Field Dynamics (CFD) and quantum chemical calculations.

## Structure

- `visualizations/`: Interactive 3D orbital visualizations
  - `mixed_basis/`: Initial visualizations using mixed basis sets (STO-3G/def2-SVP)
  - `def2svp/`: High-resolution visualizations using the def2-SVP basis set
- `data/`: Computational results and Fbond metrics (JSON)
- `scripts/`: Python scripts for CCSD workflow, Fbond calculation, and Plotly visualization

## Requirements

- Python 3.11+
- PySCF 2.12+
- NumPy, SciPy, Plotly

## Usage

1. **Environment Setup:**
   ```bash
   conda create -n fbond-env python=3.11
   conda activate fbond-env
   pip install pyscf numpy scipy plotly
2. **Run Analysis:**
    Run Analysis:
   ```bash
python scripts/fbond_ccsd_workflow.py
3. **View Visualizations:** Open the HTML files in visualizations/ directly in any web browser to explore the 3D HOMO-LUMO orbitals.

**Methodology**

This workflow utilizes CCSD (Coupled Cluster Singles and Doubles) level theory to compute the quantum fidelity metric (Fbond​) as a descriptor for non-covalent bonding and superatomic stability.
