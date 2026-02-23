# Natural Orbital Correlation Analysis of Cluster Bonding

**From Aromatic Clusters to Metallic Superatoms with Quantum Topology Probes**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains all computational scripts, raw data, and reproducibility materials for the $N_D$ correlation analysis paper. The framework applies the **Takatsuka–Head-Gordon index of effectively unpaired electrons** ($N_D = \sum_i n_i(2-n_i)$) to diverse cluster systems, computed from CCSD natural orbital occupations over the **complete** correlated orbital space.

We introduce the per-electron correlation density:

$$f_e = \frac{N_D}{N_{\text{corr}}}$$

where $N_{\text{corr}}$ is the number of electrons in the CCSD correlation treatment (total electrons minus frozen core), enabling meaningful comparison across systems with different sizes and core treatments.

### Key Findings

- **Two distinct correlation regimes**: small clusters ($f_e \approx 0.08\text{--}0.14$) vs. metallic superatoms ($f_e \approx 0.04\text{--}0.05$)
- **Orbital space completeness is critical**: truncating to a small active window underestimates $N_D$ by up to 6,200×
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
| Al₄²⁻ (aromatic) | 3.84 | 0.083 | Metal aromatic |
| Al₄⁴⁻ (singlet) | 4.03 | 0.084 | Metal antiaromatic |
| B₁₂ (planar) | 4.42 | 0.123 | Electron-deficient |
| B₁₂ (icosahedral) | 4.99 | 0.139 | Strained cage |
| B₆N₆ | 5.11 | 0.106 | Heteroatomic |
| Cs₃Al₈⁻ | 5.58 | 0.048 | Metallic superatom |
| Al₄⁴⁻ (triplet) | 6.00 | 0.188 | Open-shell |
| Cs₃Al₁₂⁻ | 7.10 | 0.044 | Metallic superatom |

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
│
├── quantum/                         # Quantum topology study
│   ├── fbond_pasqal.py              # Pasqal neutral-atom simulation script
│   └── plot_pasqal_results.py       # Visualization of Rydberg results
│
├── data/                            # Raw computational data
│   └── fbond_pasqal_results_final.json  # Quantum simulation results
│
├── example_output/                  # Classical calculation outputs
│   ├── fbond_results_combined.json  # Complete N_D results
│   ├── Cs3Al8_structure.xyz         # Optimized Cs₃Al₈⁻ geometry
│   └── Cs3Al12_structure.xyz        # Optimized Cs₃Al₁₂⁻ geometry
│
└── manuscript/                      # Manuscript sources
    ├── unified_fbond_manuscript_v5.tex  # Main manuscript (revised)
    ├── references_unified.bib           # Bibliography
    ├── Supporting_Information.tex       # SI LaTeX source
    └── Supporting_Information.pdf       # Compiled SI
```

## Installation

### Prerequisites
- Python ≥ 3.9
- PySCF 2.12.1+
- Pulser SDK (for quantum simulations)

### Setup
```bash
git clone https://github.com/unearthlyimprint/fbond-superatom-aromaticity.git
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
  journal = {Preprint},
  year    = {2026},
  note    = {In preparation}
}
```

## Version History

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

**Celal Arda** — [GitHub](https://github.com/unearthlyimprint)
