# Natural Orbital Correlation Analysis of Cluster Bonding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20493151.svg)](https://doi.org/10.5281/zenodo.20493151)
[![Preprint: ChemRxiv](https://img.shields.io/badge/Preprint-ChemRxiv-orange.svg)](https://doi.org/10.26434/chemrxiv-2025-bnp0l-v2)
[![ACS Omega: ao-2026-06677s](https://img.shields.io/badge/ACS%20Omega-ao--2026--06677s-b30000.svg)](https://pubs.acs.org/journal/acsodf)

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
| Al₄²⁻ (aromatic) | 54 | 2.54 | 0.055 |
| Al₄⁴⁻ (antiaromatic, singlet) | 56 | 2.53 | 0.053 |
| Al₄⁴⁻ (triplet) | 56 | 4.34 | 0.090 |
| B₁₂ (planar) | 60 | 4.42 | 0.123 |
| B₁₂ (icosahedral) | 60 | 4.99 | 0.139 |
| B₆N₆ (planar) | 72 | 5.11 | 0.106 |
| Cs₃Al₈⁻ | 132 | 5.58 | 0.048 |
| Cs₃Al₁₂⁻ | 184 | 7.10 | 0.044 |

**Correlation-density trend:** small covalent clusters (C₆H₆, B₁₂, B₆N₆) show a higher per-electron correlation density (f<sub>e</sub> ≈ 0.08–0.14) than metallic superatoms (f<sub>e</sub> ≈ 0.03–0.05). The aromatic/antiaromatic aluminium clusters (Al₄²⁻, Al₄⁴⁻ singlet) fall at the boundary between these groups (f<sub>e</sub> ≈ 0.05, with multireference character), while the open-shell Al₄⁴⁻ triplet (f<sub>e</sub> = 0.090) sits with the covalent clusters — so f<sub>e</sub> tracks the *character* of correlation rather than system size. The separation is clearest at the regime extremes, so this is presented as a trend, not a sharp two-regime boundary.

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

## Reproducing Table 1

The full correlation diagnostics in **Table 1** (the Takatsuka-Head-Gordon index
N<sub>D</sub> and the per-electron correlation density f<sub>e</sub>) reproduce
directly from the deposited CCSD natural-orbital occupations, with no
recomputation required.

`example_output/fbond_results_combined.json` holds, for each of the 11 systems,
the full per-orbital occupation array `natural_orbital_occupations_n_i` together
with the derived `N_D`, `n_correlated`, and `f_e`. The index is the sum over
those occupations, N<sub>D</sub> = Σ<sub>i</sub> n<sub>i</sub>(2 - n<sub>i</sub>),
and f<sub>e</sub> = N<sub>D</sub> / N<sub>corr</sub>, so every row of Table 1
follows from a few lines:

```python
import json

for s in json.load(open("example_output/fbond_results_combined.json")):
    n   = s["natural_orbital_occupations_n_i"]
    N_D = sum(x * (2 - x) for x in n)      # Takatsuka-Head-Gordon index
    f_e = N_D / s["n_correlated"]          # per-electron correlation density
    print(f"{s['system']:20s} N_D={N_D:6.3f}  f_e={f_e:.3f}")
```

This reproduces every Table 1 value to better than 0.001. To regenerate the
occupations from scratch, run the CCSD workflow (see **Usage**) on the input
geometries in `geometries/` and `structures/`; the corrected Al₄ geometries and
the per-system backing files are documented in
`cloud_results/PROVENANCE_NOTES.md`.

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

### v2.1.1 (2026-06-27)
- **Repository hygiene:** this archive now carries reproducibility **data and code only**. The manuscript and Supporting Information are hosted on ChemRxiv and (under review) ACS Omega, and are no longer mirrored in this repository.
- **Deposit enriched:** `example_output/fbond_results_combined.json` now includes the per-orbital natural-orbital occupation array (`natural_orbital_occupations_n_i`) for all 11 systems, so every Table 1 N<sub>D</sub> reproduces exactly via N<sub>D</sub> = Σ<sub>i</sub> n<sub>i</sub>(2 − n<sub>i</sub>).
- **Removed superseded pre-correction Al₄ files** from `cloud_results/` (computed on the old Bohr-mislabeled geometry; they contradicted the corrected Table 1). The authoritative `Al4_corrected_table1_results.json` and its generator `recompute_al4_table1.py` are retained; removed files remain in git history.
- **Provenance notes** updated to the corrected nine-distinct-register graph statistics (Section 3.7 / Table 4): interaction heterogeneity ρ = 0.68, p = 0.042 (nominal only, does not survive multiple-comparison correction).
- **Added a "Reproducing Table 1" section** with the exact N<sub>D</sub> = Σ<sub>i</sub> n<sub>i</sub>(2 − n<sub>i</sub>) recompute from the deposit.

### v2.1.0 (2026-06-16)
- **Reproducibility deposit:** added `cloud_results/` (complete CCSD natural-orbital occupation arrays + `extract_fe_from_checkpoints.py`) and `cloud_results/PROVENANCE_NOTES.md`, so the full N<sub>D</sub> / f<sub>e</sub> column reproduces from the deposited wavefunctions.
- **Correction — Al₄ geometry units:** the three Al₄ structures were stored in Bohr but read as Ångström, stretching the clusters ~1.9×. Recomputed at the equilibrium geometry: Al₄²⁻ N<sub>D</sub> 3.84 → 2.54 (f<sub>e</sub> 0.083 → 0.055), Al₄⁴⁻ singlet 4.03 → 2.53 (0.084 → 0.053), Al₄⁴⁻ triplet 4.17 → 4.34 (0.087 → 0.090). All other systems are genuine Ångström and unchanged.
- **Reframed** the correlation-density trend accordingly: the Al₄ singlets are now an informative **boundary** case (superatom-like f<sub>e</sub> with multireference character); the trend rests on the regime extremes, and the earlier "matched-element ≈2×" comparison was removed.
- Added multireference characterization (CASSCF / AVAS active space, leading-weight analysis) to the Supporting Information.
- Raw CCSD checkpoints archived on Zenodo (DOI [10.5281/zenodo.20493151](https://doi.org/10.5281/zenodo.20493151)).
- Added DOI, preprint, and journal-status banners.

### v2.0.0 (2026-02-17)
- **Major upgrade:** Added quantum hardware validation (Pasqal neutral-atom simulation)
- Added `quantum/` directory with `fbond_pasqal.py` and `plot_pasqal_results.py`
- Added `data/fbond_pasqal_results_final.json` (500-shot MPS emulator results)
- Added `manuscript/Supporting_Information.tex` and `.pdf` *(removed in v2.1.1; the paper and SI live on ChemRxiv / ACS Omega, not this data repo)*
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
