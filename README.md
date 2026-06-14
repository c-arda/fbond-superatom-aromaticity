# Natural Orbital Correlation Analysis of Cluster Bonding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Repository for:**
> *Natural Orbital Correlation Analysis of Cluster Bonding: From Aromatic Clusters to Metallic Superatoms with Quantum Topology Probes*
>
> Celal Arda — ACS Omega, manuscript ao-2026-06677s (2026)

---

## Overview

This repository contains all computational scripts, raw data, and reproducibility materials for the paper. The analysis uses two correlation measures computed from CCSD natural orbital occupations:

- **N<sub>D</sub>** (Takatsuka–Head-Gordon index, extensive): total deviation from idempotency, N<sub>D</sub> = Σ n<sub>i</sub>(2 − n<sub>i</sub>), summed over the **complete** natural orbital space.
- **f<sub>e</sub> = N<sub>D</sub> / N<sub>corr</sub>** (per-electron correlation density, intensive): enables comparison across systems of different sizes.

### Key Findings

| System | N<sub>e</sub> | N<sub>D</sub> | f<sub>e</sub> |
|--------|------|------|------|
| Al₄²⁻ (aromatic) | 54 | 3.84 | 0.083 |
| Al₄⁴⁻ (antiaromatic) | 56 | 4.03 | 0.084 |
| B₁₂ (planar) | 60 | 4.42 | 0.123 |
| B₁₂ (icosahedral) | 60 | 4.99 | 0.139 |
| B₆N₆ (planar) | 72 | 5.11 | 0.106 |
| Cs₃Al₈⁻ | 132 | 5.58 | 0.048 |
| Cs₃Al₁₂⁻ | 184 | 7.10 | 0.044 |

**Correlation-density trend:** small clusters show a higher per-electron correlation density (f<sub>e</sub> ≈ 0.08–0.14) than metallic superatoms (f<sub>e</sub> ≈ 0.04–0.05); presented as a trend (clearest at the extremes and in matched-element comparisons), not a sharp two-regime boundary.

### Quantum Topology Probes (Exploratory)

As a complementary, hypothesis-generating study, the molecular bonding topologies are embedded as interaction graphs on Pasqal neutral-atom (Rydberg) **cloud emulators** (EMU_FREE; EMU_TN for the largest registers; 500 shots per system), and topology-dependent entanglement signatures are examined. No single graph metric reaches statistical significance, so this is presented as exploratory, not as a validation of the classical results.

| System | Quantum S<sub>E</sub><sup>Q</sup> (nats) |
|--------|------|
| Al₄²⁻ (aromatic) | 0.503 |
| Al₄⁴⁻ (antiaromatic) | 0.621 |
| B₁₂ (planar) | 0.585 |
| B₆N₆ (planar) | 0.577 |
| Cs₃Al₈⁻ (superatom) | 0.674 |

---

## Repository Structure

```
fbond-superatom-aromaticity/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── automated_fbond_workflow.py        # Main CCSD/F_bond calculation
├── optimize_geometry.py               # B3LYP geometry optimization
├── visualize_orbitals.py              # Generate orbital cube files and HTML
│
├── quantum/                           # Quantum hardware validation
│   ├── fbond_pasqal.py                # Pasqal neutral-atom simulation script
│   └── plot_pasqal_results.py         # Visualization of quantum results
│
├── data/                              # Raw computational data
│   └── fbond_pasqal_results_final.json# Quantum simulation results (500 shots)
│
├── example_output/                    # Classical calculation outputs
│   ├── fbond_results_combined.json    # Complete F_bond results
│   ├── Cs3Al8_structure.xyz           # Optimized Cs₃Al₈⁻ geometry
│   └── Cs3Al12_structure.xyz          # Optimized Cs₃Al₁₂⁻ geometry
│
└── manuscript/                        # Supporting Information
    ├── Supporting_Information.tex      # SI LaTeX source
    └── Supporting_Information.pdf      # Compiled SI
```

---

## Installation

### Prerequisites
- Python ≥ 3.11
- PySCF 2.12.1
- Pasqal Pulser SDK (for quantum validation)

### Setup
```bash
git clone https://github.com/c-arda/fbond-superatom-aromaticity.git
cd fbond-superatom-aromaticity
pip install -r requirements.txt
```

---

## Usage

### Classical F<sub>bond</sub> Calculation
```bash
# Full CCSD workflow (geometry optimization → CCSD → F_bond)
python automated_fbond_workflow.py
```

### Quantum Hardware Validation
```bash
# Local simulation (no cloud credentials needed)
python quantum/fbond_pasqal.py --mode local --shots 100

# Cloud simulation via Pasqal SDK (requires credentials)
export PASQAL_PROJECT_ID="your-project-id"
export PASQAL_USERNAME="your-username"
export PASQAL_PASSWORD="your-password"
python quantum/fbond_pasqal.py --mode cloud --emulator EMU_FREE --shots 500

# Plot results
python quantum/plot_pasqal_results.py
```

---

## Computational Details

### Classical Methods
- **Level of theory:** CCSD/def2-SVP (frozen core)
- **Software:** PySCF 2.12.1
- **Key insight:** Complete natural orbital space retention is essential.
  Truncating to a small active space underestimates N<sub>D</sub>
  by up to 8,200×.

### Quantum Methods
- **Platform:** Pasqal neutral-atom (Rydberg) cloud emulators
- **Protocol:** Adiabatic Rydberg blockade evolution
- **Backend:** EMU_FREE (≤12-qubit registers) and EMU_TN tensor-network (largest registers), 500 shots per system
- **Mapping:** Force-directed 2D layout preserving bonding topology (R > 5 μm)

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{arda2026fbond,
  author  = {Arda, Celal},
  title   = {Natural Orbital Correlation Analysis of Cluster Bonding:
             From Aromatic Clusters to Metallic Superatoms
             with Quantum Topology Probes},
  journal = {ACS Omega},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Version History

### v2.0.0 (2026-02-17)
- **Major upgrade:** Added quantum hardware validation (Pasqal neutral-atom simulation)
- Added `quantum/` directory with `fbond_pasqal.py` and `plot_pasqal_results.py`
- Added `data/fbond_pasqal_results_final.json` (500-shot MPS emulator results)
- Added `manuscript/Supporting_Information.tex` and `.pdf`
- Updated README to reflect v4 manuscript (ACS Omega submission)
- Expanded scope from superatoms-only to unified framework (Al₄, B₁₂, B₆N₆, Cs₃Al_n⁻)

### v1.0.0 (2026-02-11)
- Initial release: Classical F<sub>bond</sub> workflow for Cs₃Al_n⁻ superatom clusters
- Automated CCSD/Lambda-CCSD/NOON pipeline
- Structure optimization and orbital visualization scripts

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Celal Arda — celal.arda@outlook.de
