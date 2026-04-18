#!/bin/bash
# ============================================================================
# TZVP Basis-Set Validation Pipeline
# ============================================================================
# Runs B12 icosahedral and Cs3Al8- CCSD/def2-TZVP calculations sequentially
# on the local workstation (AMD Ryzen 9 9950X, 16c/32t, 60GB RAM).
#
# Expected total runtime: ~10-24 hours
#   Job 1: B12 icosahedral    ~2-8 hours
#   Job 2: Cs3Al8-            ~8-20 hours
#
# Usage: nohup bash run_tzvp_pipeline.sh &> tzvp_pipeline.log &
# ============================================================================

set -euo pipefail

# Activate conda environment with PySCF
source /home/ardac/miniconda3/etc/profile.d/conda.sh
conda activate fbond-env

# Redirect PySCF scratch to /data (896 GB NVMe) instead of /tmp (31 GB tmpfs)
# The CCSD AO-to-MO integral transformation writes ~30-100 GB temp files
export PYSCF_TMPDIR="/data/pyscf_scratch"
export TMPDIR="/data/pyscf_scratch"
mkdir -p "$PYSCF_TMPDIR"

# Allow PySCF to use more RAM (in MB) — Ryzen 9 has 60 GB
export PYSCF_MAX_MEMORY=48000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  TZVP BASIS-SET VALIDATION PIPELINE"
echo "  Host: $(hostname)"
echo "  CPU:  $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  RAM:  $(free -h | awk '/Mem:/{print $2}')"
echo "  Cores: $(nproc)"
echo "  Started: $(date -Iseconds)"
echo "============================================================"
echo ""

# ── Job 1: B12 icosahedral ──────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  JOB 1/2: B12 icosahedral CCSD/def2-TZVP               │"
echo "│  Expected: ~2-8 hours                                   │"
echo "│  Start: $(date -Iseconds)                               │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

python3 b12_ico_tzvp_comparison.py
B12_EXIT=$?

if [ $B12_EXIT -eq 0 ]; then
    echo ""
    echo "  ✓ B12 icosahedral COMPLETED at $(date -Iseconds)"
    echo ""
else
    echo ""
    echo "  ✗ B12 icosahedral FAILED (exit code $B12_EXIT)"
    echo "  Continuing to next job..."
    echo ""
fi

# ── Job 2: Cs3Al8- ─────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  JOB 2/2: Cs3Al8- CCSD/def2-TZVP                       │"
echo "│  Expected: ~8-20 hours                                  │"
echo "│  Start: $(date -Iseconds)                               │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

python3 cs3al8_tzvp_comparison.py
CS3AL8_EXIT=$?

if [ $CS3AL8_EXIT -eq 0 ]; then
    echo ""
    echo "  ✓ Cs3Al8- COMPLETED at $(date -Iseconds)"
    echo ""
else
    echo ""
    echo "  ✗ Cs3Al8- FAILED (exit code $CS3AL8_EXIT)"
    echo ""
fi

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PIPELINE SUMMARY"
echo "  Finished: $(date -Iseconds)"
echo "  B12 icosahedral: $([ $B12_EXIT -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "  Cs3Al8-:         $([ $CS3AL8_EXIT -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""
echo "  Results saved to:"
echo "    B12_icosahedral_def2tzvp_results.json"
echo "    Cs3Al8_minus_def2tzvp_results.json"
echo "============================================================"
