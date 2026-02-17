# F<sub>bond</sub>: A Unified Measure of Electron Correlation on Classical and Quantum Processors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Repository for:**
> *F<sub>bond</sub> as a Unified Measure of Electron Correlation: From Aromatic Clusters to Metallic Superatoms on Classical and Quantum Processors*
>
> Celal Arda — Submitted to ACS Omega (2026)

---

## Overview

This repository contains all computational scripts, raw data, and reproducibility materials for the F<sub>bond</sub> framework paper. The framework introduces two complementary measures of electron correlation:

- **F<sub>bond</sub><sup>(A)</sup>** (intensive, frontier-based): Captures HOMO–LUMO entanglement.
- **F<sub>bond</sub><sup>(B)</sup>** (extensive, total): Measures total deviation from idempotency across the **complete** natural orbital space.
- **f<sub>e</sub>** (per-electron correlation density): Enables meaningful comparison across systems of different sizes.

### Key Findings

| System | N<sub>e</sub> | F<sub>bond</sub><sup>(B)</sup> | f<sub>e</sub> |
|--------|------|------|------|
| Al₄²⁻ (aromatic) | 54 | 3.84 | 0.071 |
| Al₄⁴⁻ (antiaromatic) | 56 | 4.03 | 0.072 |
| B₁₂ (planar) | 60 | 4.42 | 0.074 |
| B₁₂ (icosahedral) | 60 | 4.99 | 0.083 |
| B₆N₆ (planar) | 72 | 5.11 | 0.071 |
| Cs₃Al₈⁻ | 132 | 5.58 | 0.042 |
| Cs₃Al₁₂⁻ | 184 | 7.10 | 0.039 |

**Two distinct correlation regimes:** small clusters (f<sub>e</sub> ≈ 0.07) vs. metallic superatoms (f<sub>e</sub> ≈ 0.04).

### Quantum Hardware Validation

We validated the F<sub>bond</sub> framework using **analog quantum simulation** on a Pasqal neutral-atom processor (MPS emulator). The quantum entanglement topology reproduces the chemical bonding character:

| System | Classical S<sub>E,max</sub> | Quantum S<sub>E</sub><sup>Q</sup> |
|--------|------|------|
| Al₄²⁻ (aromatic) | 0.028 | 0.514 |
| Al₄⁴⁻ (antiaromatic) | 0.019 | 0.611 |
| B₁₂ (planar) | 0.030 | 0.593 |
| B₆N₆ (planar) | 0.035 | 0.577 |
| Cs₃Al₈⁻ (superatom) | 0.013 | 0.674 |

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
git clone https://github.com/unearthlyimprint/fbond-superatom-aromaticity.git
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
python quantum/fbond_pasqal.py --mode cloud --emulator EMU_MPS --shots 500

# Plot results
python quantum/plot_pasqal_results.py
```

---

## Computational Details

### Classical Methods
- **Level of theory:** CCSD/def2-SVP (frozen core)
- **Software:** PySCF 2.12.1
- **Key insight:** Complete natural orbital space retention is essential.
  Truncating to a small active space underestimates F<sub>bond</sub><sup>(B)</sup>
  by up to 6,200×.

### Quantum Methods
- **Platform:** Pasqal neutral-atom processor (MPS emulator)
- **Protocol:** Adiabatic Rydberg blockade evolution
- **Backend:** Matrix Product State (MPS), 500 shots per system
- **Mapping:** Force-directed 2D layout preserving bonding topology (R > 5 μm)

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{arda2026fbond,
  author  = {Arda, Celal},
  title   = {F_bond as a Unified Measure of Electron Correlation:
             From Aromatic Clusters to Metallic Superatoms
             on Classical and Quantum Processors},
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
