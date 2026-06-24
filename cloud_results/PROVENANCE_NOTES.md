# Provenance notes: cluster-bonding application paper

This repository is cited in the Supporting Information of "Natural Orbital
Correlation Analysis of Cluster Bonding: From Aromatic Clusters to Metallic
Superatoms with Quantum Topology Probes" as the public reproducibility archive
for the CCSD/def2-SVP correlation indices (Table 1) and the Pasqal emulator
entanglement values (Table 4).

## Canonical extracted metrics

`*_results_corrected.json` carry `natural_occupations_full`, `N_D`, `n_corr`,
and `f_e`. The index is reproduced by `N_D = sum_i n_i (2 - n_i)` over
`natural_occupations_full`; 8 of the 11 Table 1 rows reproduce to within 0.01
directly from these files. (C6H6 = 2.48 from the `_corrected` file is the value
used in the paper.)

## System-specific provenance and superseded files

### Al4 systems (all three)  —  2026-06 geometry-units correction
The three Al4 input geometries were stored in Bohr but read as Angstrom,
stretching the clusters ~1.89x (near-dissociated). Corrected compact
(equilibrium-geometry) CCSD values are now reported in Table 1:
  - Al4(2-)     N_D = 2.54, f_e = 0.055   (was 3.84)
  - Al4(4-) S   N_D = 2.53, f_e = 0.053   (was 4.03)
  - Al4(4-) T   N_D = 4.34, f_e = 0.090   (was 4.17 / 5.196), UCCSD, E_CCSD = -966.7322 Ha

**Single source of truth for Al4:** `Al4_corrected_table1_results.json` (full
natural occupations; reproduces via `recompute_al4_table1.py`) and the corrected
Zenodo Tier-1 checkpoints `Al4_*_checkpoint_ccsd.pkl` in
`manuscript/v6/al4_correction_2026-06-14/corrected_checkpoints/`. The geometry-units
correction is described in the manuscript SI.

SUPERSEDED (pre-correction, stretched geometry — retained for transparency only,
do NOT use for reported values):
  - `Al4_2minus_results_corrected.json` (N_D 3.84) and
    `Al4_4minus_results_corrected.json` (N_D 4.03)
  - `Al4_4minus_triplet_results_corrected.json` (N_D 5.196, higher-energy soln)
    and `Al4_4minus_triplet_results_FINAL_4.17.json` (N_D 4.17)
  - the three Al4 entries in `fbond_results_corrected.json`. That file is the
    **cached combined output** of `extract_fe_from_checkpoints.py` (written at its
    line 315), not a hand-maintained source; re-running that script over the
    corrected checkpoints regenerates it with the right Al4 arrays. Non-Al4
    entries (B / C / Cs3Al / Au13 / B12N12) are unaffected by the units bug.

### Au13-  (Table 1: N_D = 6.76, f_e = 0.030)
- Reported values use LANL2DZ with the Hay-Wadt LANL2DZ-ECP. Au13- is the only
  system not treated at def2-SVP; this is stated in the main-text Computational
  Details. Backing file: `Au13_minus_lanl2dz_production.json` (full natural
  occupations, 286 orbitals, N_D = 6.7576).
- NON-PRODUCTION: `Au13_minus_results.json` is a def2-SVP run that only reached
  the HF stage (stored natural occupations are all 2.0, so its N_D = 0). It does
  **not** back the reported value and should not be used for it.

### B12N12 cage  (Table 1: N_D = 7.18, f_e = 0.075)
- Backed by `B12N12_cage_results.json` (`N_D` = 7.18, `N_D_from_run_log` =
  7.183). The converged CCSD run log and a 20 MB CCSD checkpoint are archived on
  Zenodo; the full CCSD amplitudes were not retained, as disclosed in the SI.

## Quantum entanglement (Table 4, ten systems)

All tabulated S_E^Q values are Pasqal **cloud-emulator** (device-level
simulation) measurements at **500 shots** per system: EMU_FREE for the
<=12-qubit registers, EMU_TN (tensor-network) for the 13- and 16-qubit
systems. Local QutipEmulator (state-vector) runs were kept only as
consistency checks and are not the tabulated values. (An earlier version of
this note said "2000 shots / QutipEmulator state-vector" for the <=12-qubit
rows; that was incorrect, the production rows are 500-shot cloud runs.)

`../quantum/pasqal_results/table4_quantum_results_10systems.json` carries the
ten Table-4 rows (the `C6H6_carbon_only` 6-qubit entry in that file is a
diagnostic sub-register, not a Table-4 row). Row -> S_E^Q:
C6H6 0.641, Al4(2-) 0.503, Al4(4-) 0.621, Al4(4-) triplet 0.625,
B12 planar 0.585, B12 ico 0.575, **B6N6 0.577**, B12N12 0.581,
Au13- 0.630, Cs3Al8- 0.674. Raw EMU_TN batch counters for the two largest
registers are `0d20062d` (Au13-, 0.630) and `55a02bc7` (B12N12, 0.581),
500 shots each.

`B6N6 0.577` is the planar heteroatomic ring (12-qubit register), measured in
the same emulator campaign and added as the tenth Table-4 system; it is also
the B6N6 row of SI Table S3. SI Table S3 is an independent replicate EMU_FREE
batch (500 shots) of five <=12-qubit registers; for Al4(2-), Al4(4-) and
planar B12 its values differ from Table 4 by 1-2% (finite-shot sampling),
while B6N6 and Cs3Al8- have a single cloud measurement shared by both tables.
The ten-system graph-connectivity Spearman statistics (mean coordination
rho = -0.06, p = 0.88; interaction heterogeneity rho = 0.57, p = 0.08; qubit
count rho = 0.01, p = 0.97; none significant) are reproduced by
`molecular_F_bond_structure/manuscript/v5/scripts/graph_connectivity_analysis.py`.

## Large checkpoints

CCSD checkpoints exceeding GitHub's practical file-size limit (Al4(4-) triplet,
26 MB; B12N12 cage, 20 MB) are not stored in git. They are archived on Zenodo
(DOI 10.5281/zenodo.20493151) and available from the authors on request.
