# Natural Orbital Correlation Analysis of Cluster Bonding

**From Aromatic Clusters to Metallic Superatoms with Quantum Topology Probes**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/PySCF-2.12-2C6DAC.svg)](https://pyscf.org)
[![Pasqal](https://img.shields.io/badge/Pasqal-Neutral_Atom-00C7B7.svg)](https://www.pasqal.com)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg?logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C.svg)](https://matplotlib.org)
[![LaTeX](https://img.shields.io/badge/LaTeX-Manuscript-008080.svg?logo=latex&logoColor=white)](https://www.latex-project.org)
[![PCCP](https://img.shields.io/badge/PCCP-Submitted-B31B1B.svg)](https://pubs.rsc.org/en/journals/journalissues/cp)
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434/chemrxiv.15000134-B31B1B.svg)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000134/v5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains all computational scripts, raw data, and reproducibility materials for the $N_D$ correlation analysis paper. The framework applies the **Takatsuka–Head-Gordon index of effectively unpaired electrons** ($N_D = \sum_i n_i(2-n_i)$) to diverse cluster systems, computed from CCSD natural orbital occupations over the **complete** correlated orbital space.

We introduce the per-electron correlation density:

$$f_e = \frac{N_D}{N_{\text{corr}}}$$

where $N_{\text{corr}}$ is the number of electrons in the CCSD correlation treatment (total electrons minus frozen core), enabling meaningful comparison across systems with different sizes and core treatments.

### Key Findings

- **Two distinct correlation regimes**: small clusters ($f_e \approx 0.08\text{--}0.14$) vs. metallic superatoms ($f_e \approx 0.04\text{--}0.05$)
- **Orbital space completeness is critical**: truncating to a small active window underestimates $N_D$ by up to 6,400×
- **$N_D$ is dominated by dynamic correlation**: the virtual orbital tail contributes the majority of the signal

### Molecular Topology on Quantum Hardware

In a separate investigation, we explore whether molecular bonding topology generates characteristic entanglement signatures when embedded as interaction graphs on a **Pasqal neutral-atom quantum processor**. We emphasize that the Rydberg spin Hamiltonian is physically distinct from the electronic Hamiltonian—results are interpreted as a study of molecular graph topology, not electronic wavefunctions.

Key topology findings across 9 molecular systems (4–16 qubits):
- Different chemical topology classes (aromatic, antiaromatic, cage, metallic) produce systematically distinct entanglement signatures
- Entanglement scales with graph connectivity, not register size
- Signatures are robust under realistic noise profiles

## Systems Studied

| System | $N_D$ | $f_e$ | Character |
|--------|-------|-------|-----------|
| C₆H₆ (benzene) | 2.49 | 0.083 | Organic aromatic |
| Al₄²⁻ (aromatic) | 3.84 | 0.083 | Metal aromatic |
| Al₄⁴⁻ (singlet) | 4.03 | 0.084 | Metal antiaromatic |
| Al₄⁴⁻ (triplet) | 4.17 | 0.087 | Open-shell |
| B₁₂ (planar) | 4.42 | 0.123 | Electron-deficient |
| B₁₂ (icosahedral) | 4.99 | 0.139 | Strained cage |
| B₆N₆ | 5.11 | 0.106 | Heteroatomic |
| Cs₃Al₈⁻ | 5.58 | 0.048 | Metallic superatom |
| Au₁₃⁻ | 6.76 | 0.030 | Noble-metal cluster |
| Cs₃Al₁₂⁻ | 7.10 | 0.044 | Metallic superatom |
| B₁₂N₁₂ (cage) | 7.18 | 0.075 | Heteroatomic cage |

## Repository Structure

```
fbond-superatom-aromaticity/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
│
├── automated_fbond_workflow.py      # Main CCSD/N_D calculation pipeline
├── optimize_geometry.py             # B3LYP geometry optimization
├── visualize_orbitals.py            # Generate orbital cube files and HTML
├── regenerate_figures.py            # Regenerate all manuscript figures
│
├── quantum/                         # Quantum topology study
│   ├── fbond_pasqal.py              # Pasqal neutral-atom simulation script
│   └── plot_pasqal_results.py       # Visualization of Rydberg results
│
├── basis_set_validation/            # def2-TZVP validation (ESI Table S6)
│   ├── benzene_tzvp_comparison.py   # C₆H₆ CCSD/def2-TZVP
│   ├── C6H6_benzene_def2tzvp_results.json
│   ├── al4_tzvp_comparison.py       # Al₄²⁻ CCSD/def2-TZVP
│   └── Al4_2minus_def2tzvp_results.json
│
├── data/                            # Raw computational data
│   └── fbond_pasqal_results_final.json  # Quantum simulation results
│
├── structures/                      # Optimized geometries (.xyz)
│   ├── C6H6_benzene_structure.xyz   # Benzene (D6h)
│   ├── Al4_2minus_structure.xyz     # Al₄²⁻ aromatic (D4h)
│   ├── Al4_4minus_structure.xyz     # Al₄⁴⁻ antiaromatic singlet (D2h)
│   ├── Al4_4minus_triplet_structure.xyz  # Al₄⁴⁻ triplet
│   ├── B12_planar_structure.xyz     # Planar B₁₂ (D3h)
│   ├── B12_icosahedral_structure.xyz # Icosahedral B₁₂ (Ih)
│   ├── B6N6_planar_structure.xyz    # Planar B₆N₆
│   ├── B12N12_cage_structure.xyz    # B₁₂N₁₂ cage (Td)
│   ├── Au13_minus_structure.xyz     # Au₁₃⁻ icosahedral
│   ├── Cs3Al8_structure.xyz         # Cs₃Al₈⁻ superatom
│   └── Cs3Al12_structure.xyz        # Cs₃Al₁₂⁻ superatom
│
└── example_output/                  # Classical calculation outputs
    └── fbond_results_combined.json  # Complete N_D results (all 11 systems)
```

## Installation

### Prerequisites
- Python ≥ 3.9
- PySCF 2.12.1+
- Pulser SDK (for quantum simulations)

### Setup
```bash
git clone https://github.com/c-arda/fbond-superatom-aromaticity.git
cd fbond-superatom-aromaticity
pip install -r requirements.txt
```

## Usage

### Classical $N_D$ Calculation
```bash
python automated_fbond_workflow.py
```
This runs the full CCSD/Λ-CCSD pipeline: geometry → SCF → CCSD → Lambda → 1-RDM → NOONs → $N_D$ → $f_e$.

### Quantum Topology Study
```bash
cd quantum/
python fbond_pasqal.py
```
Maps molecular coordinates onto Rydberg atom registers (uniform spatial scaling, 1 Å → 3 μm) and computes entanglement signatures.

## Computational Details

### Classical Methods
- **Level of theory**: CCSD/def2-SVP with def2-ECP for Cs and Au
- **Core treatment**: Frozen core (Al 1s, B 1s, N 1s); $f_e$ uses $N_{\text{corr}}$ (correlated electrons only)
- **Natural orbitals**: Full CCSD 1-RDM via Λ equations, **complete** occupation arrays retained
- **Software**: PySCF 2.12.1

### Quantum Methods
- **Platform**: Pasqal neutral-atom processor (QutipEmulator + EMU_FREE cloud)
- **Mapping**: Uniform spatial scaling of physical Cartesian coordinates
- **Protocol**: Adiabatic Rydberg blockade with calibrated noise profiles
- **Physics note**: The Rydberg Hamiltonian ($1/R^6$ van der Waals) is physically distinct from the electronic Hamiltonian; results characterize graph topology, not electronic correlation

## Citation

If you use this code or data, please cite:

```bibtex
@article{arda2026fbond,
  author  = {Arda, Celal},
  title   = {Natural Orbital Correlation Analysis of Cluster Bonding:
             From Aromatic Clusters to Metallic Superatoms with
             Quantum Topology Probes},
  journal = {Phys. Chem. Chem. Phys.},
  year    = {2026},
  note    = {Submitted},
  doi     = {10.26434/chemrxiv-2025-bnp0l-v2}
}
```

## Version History

### v6.0.0 (2026-04-14)
- **PCCP submission**: Reformatted for RSC Physical Chemistry Chemical Physics
- **Basis-set validation**: CCSD/def2-TZVP calculations for benzene (+7% $f_e$) and Al₄²⁻ (−24% $f_e$; $T_1$ drops 0.039→0.012), confirming regime classification is physical
- **Au₁₃⁻ clarification**: Corrected ECP/frozen-core accounting — 13 lowest MOs (5s/5p, 26e) frozen in CCSD
- **ESI Table S6**: Two-system basis-set comparison (benzene + Al₄²⁻)
- **Reference [20]**: Added ChemRxiv DOI
- **New directory**: `basis_set_validation/` with TZVP scripts and results

### v5.0.0 (2026-03-17)
- **ACS Omega resubmission**: Cover letter, final manuscript polish
- **Manuscript renamed**: `unified_fbond_manuscript_v5.tex` → `nd_cluster_bonding_v5.tex`
- **Figure fixes**: Corrected stale $f_e$ labels (benzene 0.069→0.083, B₁₂N₁₂ 0.050→0.075)
- **Graph connectivity analysis**: Added `graph_connectivity_analysis.py` and correlation figure
- **SI updates**: QutipEmulator vs EMU_FREE footnotes, full-space vs frontier $S_E$, wallclock times
- **µm encoding fix**: Replaced `\SI{5}{\micro\meter}` with `$\mu$m`
- **Badges**: Added technology stack (PySCF, Pasqal, PennyLane, NumPy, SciPy, Matplotlib, LaTeX)
- **ChemRxiv DOI badge**: [10.26434/chemrxiv.15000134/v5](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000134/v5)

### v4.0.0 (2026-02-28)
- **Critical review fixes**: Triplet data reconciliation, shot count consistency
- **Spearman statistics**: Corrected $\rho$=0.53 for heterogeneity in Section 5.2
- **Conclusion rewrite**: Point 6 — entanglement NOT explained by graph metrics alone
- **SI computational cost table**: Fixed swapped columns
- **Bridge paragraph**: Added Section 5 connection between classical and quantum analysis
- **Noise table footnote**: Added clarification

### v3.0.0 (2026-02-23)
- **Major revision**: Adopted standard $N_D$ (Takatsuka–Head-Gordon) nomenclature
- **Physics fix**: $f_e$ now uses $N_{\text{corr}}$ (correlated electrons) instead of $N_e$ (total electrons)
- **Quantum section reframed**: "Molecular Topology as Entanglement Graphs" — explicitly distinguishes Rydberg from electronic Hamiltonian
- Removed Formula A and meaningless B/A ratios
- Added Takatsuka (1978), Staroverov & Davidson (2000), Head-Gordon (2003) citations
- Fixed coordinate mapping description (uniform spatial scaling, not force-directed)
- Added bridge paragraph connecting classical correlation analysis to quantum topology study

### v2.0.0 (2026-02-17)
- Added quantum hardware validation (Pasqal neutral-atom simulation)
- Added `quantum/` directory with simulation scripts
- Expanded scope from superatoms-only to unified framework

### v1.0.0 (2026-02-11)
- Initial release: Classical Fbond workflow for Cs₃Al_n⁻ superatom clusters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Celal Arda** — [GitHub](https://github.com/c-arda) · [ORCID 0009-0006-4563-8325](https://orcid.org/0009-0006-4563-8325)
