#!/bin/bash
# ============================================================================
# Fbond GCE Batch — Collect results from GCS
# ============================================================================
#
# Usage:
#   ./batch/collect_results.sh                    # Download all results
#   ./batch/collect_results.sh --status            # Show job statuses
#   ./batch/collect_results.sh --merge             # Download + merge into one JSON
#
# ============================================================================

set -euo pipefail

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="us-central1"
RESULTS_BUCKET="fbond-results-${PROJECT_ID}"
CHECKPOINTS_BUCKET="fbond-checkpoints-${PROJECT_ID}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS="${SCRIPT_DIR}/cloud_results"

MODE="${1:---download}"

# ─── Show job statuses ──────────────────────────────────────────────────────

if [[ "$MODE" == "--status" ]]; then
    echo "============================================"
    echo "FBOND BATCH JOB STATUS"
    echo "============================================"
    gcloud batch jobs list \
        --location="$REGION" \
        --filter="name:fbond-" \
        --format="table(name.basename(), status.state, createTime, status.runDuration)" \
        --sort-by="~createTime" \
        --limit=25
    echo "============================================"
    exit 0
fi

# ─── Download results ────────────────────────────────────────────────────────

echo "============================================"
echo "COLLECTING RESULTS FROM GCS"
echo "============================================"
echo "  Source:  gs://${RESULTS_BUCKET}/"
echo "  Target:  ${LOCAL_RESULTS}/"
echo "============================================"

mkdir -p "$LOCAL_RESULTS"

echo ""
echo "[1/2] Downloading results..."
gcloud storage cp -r "gs://${RESULTS_BUCKET}/*" "$LOCAL_RESULTS/" 2>/dev/null || {
    echo "  (no results found in bucket yet)"
}

echo ""
echo "[2/2] Downloading checkpoints..."
mkdir -p "${LOCAL_RESULTS}/checkpoints"
gcloud storage cp -r "gs://${CHECKPOINTS_BUCKET}/*" "${LOCAL_RESULTS}/checkpoints/" 2>/dev/null || {
    echo "  (no checkpoints found in bucket yet)"
}

echo ""
echo "[✓] Results downloaded to: ${LOCAL_RESULTS}/"
echo ""

# List what we got
echo "Downloaded files:"
find "$LOCAL_RESULTS" -name "*.json" -o -name "*.xyz" -o -name "*.cube" -o -name "*.pkl" 2>/dev/null | sort

# ─── Merge results ────────────────────────────────────────────────────────────

if [[ "$MODE" == "--merge" ]]; then
    echo ""
    echo "[MERGE] Combining all *_results.json into cloud_batch_results.json..."

    python3 -c "
import json, glob, sys

results = []
for f in sorted(glob.glob('${LOCAL_RESULTS}/*_results.json')):
    with open(f) as fh:
        results.append(json.load(fh))

if not results:
    print('No result files found to merge.')
    sys.exit(0)

outfile = '${LOCAL_RESULTS}/cloud_batch_results.json'
with open(outfile, 'w') as fh:
    json.dump(results, fh, indent=2)

print(f'[✓] Merged {len(results)} results into {outfile}')

# Print summary table
print()
print(f'{\"System\":<25} {\"Basis\":<12} {\"F_bond\":<10} {\"O_MOS\":<10} {\"S_E_max\":<10} {\"E_CCSD (Ha)\":<16}')
print('-' * 83)
for r in results:
    print(f'{r[\"system\"]:<25} {r[\"basis\"]:<12} {r[\"F_bond\"]:<10.6f} {r[\"O_MOS\"]:<10.6f} {r[\"S_E_max\"]:<10.6f} {r[\"E_CCSD\"]:<16.8f}')
"
fi

echo ""
echo "============================================"
echo "COLLECTION COMPLETE"
echo "============================================"
