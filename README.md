# F_bond Analysis of Cs₃Al_n⁻ Superatom Clusters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/PySCF-2.12+-green.svg)](https://pyscf.org/)

Analysis scripts for computing F_bond quantum descriptors on cesium-aluminum superatom clusters (Cs₃Al₈⁻ and Cs₃Al₁₂⁻).

##  Associated Publication

**Title:** Size-Independent Quantum Entanglement Signatures in Cs₃Al_n⁻ Superatom Clusters: An F_bond Analysis

**Authors:** Celal Arda

**Preprint:** [ChemRxiv DOI: XXX] (add after upload)

**Abstract:** We demonstrate that the F_bond quantum descriptor exhibits remarkable size independence in cesium-aluminum superatom clusters, with values differing by only 0.87% (0.01270 vs 0.01280) despite a 36% increase in cluster size. This constancy emerges from compensating trends: the HOMO-LUMO gap increases by 24% while entanglement entropy decreases by 19%, suggesting F_bond ≈ 0.0127 is a characteristic signature of Cs₃Al_n⁻ aromaticity.

---

##  Key Findings

- **F_bond = 0.01270** for Cs₃Al₈⁻ (40 valence electrons)
- **F_bond = 0.01280** for Cs₃Al₁₂⁻ (56 valence electrons)
- **0.87% difference** despite 36% size increase
- **Compensating trends:** +24% HOMO-LUMO gap, -19% entanglement entropy
- **Universal signature:** F_bond captures intrinsic superatom aromaticity

---

##  Repository Structure

```
fbond-superatom-aromaticity/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── automated_fbond_workflow.py        # Main CCSD/F_bond calculation
├── optimize_geometry.py               # B3LYP geometry optimization
├── visualize_orbitals.py             # Generate orbital cube files and HTML
├── example_output/
│   ├── fbond_results_combined.json   # Complete results for both systems
│   ├── Cs3Al8_structure.xyz          # Optimized geometry
│   └── Cs3Al12_structure.xyz         # Optimized geometry
└── docs/
    └── calculation_workflow.md        # Detailed methodology

```

---

##  Installation

### Prerequisites

- Python 3.11 or later
- 32 GB RAM recommended (minimum 16 GB)
- Linux or macOS (Windows via WSL)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/unearthlyimprint/fbond-superatom-aromaticity.git
   cd fbond-superatom-aromaticity
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n fbond python=3.11
   conda activate fbond
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Key packages:
   - `pyscf>=2.12.1` - Quantum chemistry calculations
   - `numpy>=1.24.0` - Numerical arrays
   - `scipy>=1.11.0` - Scientific computing
   - `matplotlib>=3.7.0` - Visualization
   - `geometric>=1.0` - Geometry optimization

---

## 💻 Usage

### Quick Start: Reproduce Published Results

```bash
# Run complete workflow for Cs₃Al₈⁻
python automated_fbond_workflow.py --system Cs3Al8 --geometry example_output/Cs3Al8_structure.xyz

# Run complete workflow for Cs₃Al₁₂⁻
python automated_fbond_workflow.py --system Cs3Al12 --geometry example_output/Cs3Al12_structure.xyz
```

**Expected output:**
- JSON file with F_bond components
- Natural orbital cube files (HOMO, LUMO)
- Checkpoint files for restart capability

**Runtime:**
- Cs₃Al₈⁻: ~8-10 hours (132 electrons, CCSD)
- Cs₃Al₁₂⁻: ~30-36 hours (184 electrons, CCSD)

### Step-by-Step Workflow

#### 1. Geometry Optimization (Optional)

If starting from scratch:

```bash
python optimize_geometry.py --system Cs3Al8 --output optimized.xyz
```

**Method:** B3LYP/def2-SVP with def2-ECP for Cs

**Convergence:** Gradient RMS < 3×10⁻⁴ Ha/bohr

#### 2. F_bond Calculation

```bash
python automated_fbond_workflow.py \
    --system Cs3Al8 \
    --geometry optimized.xyz \
    --basis def2-svp \
    --frozen-core 8 \
    --output results/
```

**Options:**
- `--system`: System identifier (Cs3Al8 or Cs3Al12)
- `--geometry`: XYZ coordinate file
- `--basis`: Basis set (default: def2-svp)
- `--frozen-core`: Number of frozen core orbitals
- `--output`: Output directory

**Workflow:**
1. Hartree-Fock calculation
2. Frozen-core CCSD (T-amplitudes)
3. Lambda-CCSD (density matrix)
4. Natural orbital extraction
5. F_bond computation
6. Cube file generation

#### 3. Visualization

```bash
python visualize_orbitals.py \
    --cube-homo results/Cs3Al8_HOMO_NO.cube \
    --cube-lumo results/Cs3Al8_LUMO_NO.cube \
    --output orbital_visualization.html
```

Opens interactive 3D orbital viewer in your browser.

---

##  Output Files

### JSON Data (`fbond_results.json`)

```json
{
  "system": "Cs3Al8",
  "n_electrons": 132,
  "n_atoms": 11,
  "E_HF": -1994.282,
  "E_CCSD": -1995.118,
  "E_corr": -0.836,
  "eps_HOMO": -0.027336,
  "eps_LUMO": 0.061674,
  "O_MOS": 0.089010,
  "S_E_max": 0.285297,
  "F_bond": 0.012697
}
```

### Cube Files

Gaussian CUBE format for visualization:
- `Cs3Al8_HOMO_NO.cube` - HOMO natural orbital
- `Cs3Al8_LUMO_NO.cube` - LUMO natural orbital

Grid spacing: 0.2 Bohr  
Recommended isovalue: ±0.03 a.u.

**Visualization tools:**
- [Avogadro](https://avogadro.cc/)
- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
- [PyMOL](https://pymol.org/)
- Provided HTML script (interactive web-based)

---

##  Computational Details

### Basis Sets

- **Aluminum (Al):** def2-SVP [3s2p1d]
- **Cesium (Cs):** def2-SVP with def2-ECP (46-electron pseudopotential)

### Frozen Core Approximation

- **Cs₃Al₈⁻:** 8 Al 1s electrons frozen
- **Cs₃Al₁₂⁻:** 12 Al 1s electrons frozen
- Cs core electrons handled by ECP

### Convergence Criteria

| Property | Threshold |
|----------|-----------|
| SCF energy | 1×10⁻⁹ Ha |
| CCSD energy | 1×10⁻⁷ Ha |
| CCSD amplitudes | 1×10⁻⁵ |
| Lambda amplitudes | 1×10⁻⁵ |

### System Requirements

| System | Electrons | CCSD Time | RAM Usage |
|--------|-----------|-----------|-----------|
| Cs₃Al₈⁻ | 132 | ~8 hours | 16-24 GB |
| Cs₃Al₁₂⁻ | 184 | ~34 hours | 24-32 GB |

*Times for single-core CPU. Parallel execution reduces by 2-4×.*

---

##  Results Summary

| System | Valence e⁻ | O_MOS (Ha) | S_E,max (nats) | **F_bond** |
|--------|-----------|------------|----------------|-----------|
| Cs₃Al₈⁻ | 40 | 0.0890 | 0.2853 | **0.01270** |
| Cs₃Al₁₂⁻ | 56 | 0.1106 | 0.2316 | **0.01280** |
| **Change** | +40% | +24% | -19% | **+0.87%** |

**Key insight:** Opposite trends in gap (+24%) and entanglement (-19%) compensate to yield constant F_bond, suggesting it captures an intrinsic property of Cs₃Al_n⁻ superatom aromaticity.

---

##  References

**Primary reference for F_bond methodology:**
> Arda, C. "Quantum entanglement signatures of aromaticity..." (2025) [Previous F_bond paper]

**Superatom chemistry:**
> Li, X. et al. "Observation of all-metal aromatic clusters..." *Science* (2012)

**PySCF software:**
> Sun, Q. et al. "PySCF: The Python-based simulations of chemistry framework" *WIREs Comput. Mol. Sci.* (2018)

---

##  Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

##  Contact

**Celal Arda**  
Independent Researcher  
📧 celal.arda@outlook.de  
🔗 [ORCID: 0009-0006-4563-8325](https://orcid.org/0009-0006-4563-8325)

---

##  Citation

If you use this code in your research, please cite:

```bibtex
@article{arda2026superatom,
  title={Size-Independent Quantum Entanglement Signatures in Cs₃Al_n⁻ Superatom Clusters: An F_bond Analysis},
  author={Arda, Celal},
  journal={ChemRxiv},
  year={2026},
  doi={XXX}  % Add after publication
}
```

And the software repository:

```bibtex
@software{arda2026fbond_code,
  author={Arda, Celal},
  title={F_bond Analysis Scripts for Cesium-Aluminum Superatom Clusters},
  year={2026},
  url={https://github.com/unearthlyimprint/fbond-superatom-aromaticity}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PySCF development team** for quantum chemistry software
- **geomeTRIC** for geometry optimization tools
- **ChemRxiv** for preprint hosting

---

## Version History

### v1.0.0 (2026-02-11)
- Initial release
- Cs₃Al₈⁻ and Cs₃Al₁₂⁻ calculations
- def2-SVP/CCSD level of theory
- Size-independence discovery

---

**Last updated:** February 11, 2026
