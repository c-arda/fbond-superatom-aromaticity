#!/bin/bash
# ============================================================================
# Fbond GCE Batch — Submit a single chemistry calculation
# ============================================================================
#
# Usage:
#   ./batch/submit_batch.sh <geometry_name> [basis] [machine_type]
#
# Examples:
#   ./batch/submit_batch.sh C6H6_benzene                    # def2-SVP on c2-highmem-16
#   ./batch/submit_batch.sh Cs3Al8 def2-SVP                 # explicit basis
#   ./batch/submit_batch.sh Cs3Al12 def2-TZVP n2-highmem-32 # large basis, more RAM
#
# Batch job all-in:
#   for g in geometries/*.json; do
#     name=$(basename "$g" .json)
#     ./batch/submit_batch.sh "$name" def2-SVP
#   done
#
# ============================================================================

set -euo pipefail

# ─── Arguments ────────────────────────────────────────────────────────────────

SYSTEM_NAME="${1:?Usage: $0 <geometry_name> [basis] [machine_type]}"
BASIS="${2:-def2-SVP}"
MACHINE_TYPE="${3:-c2-highmem-16}"

# ─── Configuration ────────────────────────────────────────────────────────────

REGION="us-central1"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REPO="research-images"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/fbond:latest"

RESULTS_BUCKET="fbond-results-${PROJECT_ID}"
CHECKPOINTS_BUCKET="fbond-checkpoints-${PROJECT_ID}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="fbond-$(echo "${SYSTEM_NAME}" | tr '[:upper:]_' '[:lower:]-')-${TIMESTAMP}"

TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${TEMPLATE_DIR}/fbond-batch-job.json"

# ─── Validate ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GEOM_FILE="${SCRIPT_DIR}/geometries/${SYSTEM_NAME}.json"

if [[ ! -f "$GEOM_FILE" ]]; then
    echo "ERROR: Geometry file not found: ${GEOM_FILE}"
    echo "Available geometries:"
    ls -1 "${SCRIPT_DIR}/geometries/"*.json 2>/dev/null | xargs -n1 basename | sed 's/.json$//'
    exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: Batch template not found: ${TEMPLATE}"
    exit 1
fi

# ─── RAM heuristic ────────────────────────────────────────────────────────────
# Override machine type for large basis sets if not explicitly set

if [[ "$BASIS" == "def2-TZVP" && "$MACHINE_TYPE" == "c2-highmem-16" ]]; then
    echo "[!] def2-TZVP requires more RAM — auto-upgrading to n2-highmem-32 (256 GB)"
    MACHINE_TYPE="n2-highmem-32"
fi

# ─── Generate job config ─────────────────────────────────────────────────────

TMPJOB=$(mktemp /tmp/fbond-batch-XXXXXX.json)
trap "rm -f $TMPJOB" EXIT

sed \
    -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
    -e "s|__SYSTEM_NAME__|${SYSTEM_NAME}|g" \
    -e "s|__BASIS__|${BASIS}|g" \
    -e "s|__MACHINE_TYPE__|${MACHINE_TYPE}|g" \
    -e "s|__RESULTS_BUCKET__|${RESULTS_BUCKET}|g" \
    -e "s|__CHECKPOINTS_BUCKET__|${CHECKPOINTS_BUCKET}|g" \
    "$TEMPLATE" > "$TMPJOB"

# Adjust memory request based on machine type
case "$MACHINE_TYPE" in
    c2-highmem-16)
        sed -i 's/"cpuMilli": [0-9]*/"cpuMilli": 16000/' "$TMPJOB"
        sed -i 's/"memoryMib": [0-9]*/"memoryMib": 122880/' "$TMPJOB"
        ;;
    n2-highmem-32)
        sed -i 's/"cpuMilli": [0-9]*/"cpuMilli": 32000/' "$TMPJOB"
        sed -i 's/"memoryMib": [0-9]*/"memoryMib": 245760/' "$TMPJOB"
        ;;
    n2-highmem-64)
        sed -i 's/"cpuMilli": [0-9]*/"cpuMilli": 64000/' "$TMPJOB"
        sed -i 's/"memoryMib": [0-9]*/"memoryMib": 491520/' "$TMPJOB"
        ;;
esac

# ─── Print summary ────────────────────────────────────────────────────────────

echo "============================================"
echo "FBOND BATCH JOB SUBMISSION"
echo "============================================"
echo "  Job name:      ${JOB_NAME}"
echo "  System:        ${SYSTEM_NAME}"
echo "  Basis:         ${BASIS}"
echo "  Machine:       ${MACHINE_TYPE}"
echo "  VM type:       Spot (preemptible)"
echo "  Image:         ${IMAGE_URI}"
echo "  Results:       gs://${RESULTS_BUCKET}/"
echo "  Checkpoints:   gs://${CHECKPOINTS_BUCKET}/"
echo "  Region:        ${REGION}"
echo "============================================"

# ─── Cost estimate ────────────────────────────────────────────────────────────

case "$MACHINE_TYPE" in
    c2-highmem-16)   echo "  Est. cost:     ~\$0.24/hr × 4-6h = \$1.00-1.50" ;;
    n2-highmem-32)   echo "  Est. cost:     ~\$0.55/hr × 6-10h = \$3.30-5.50" ;;
    n2-highmem-64)   echo "  Est. cost:     ~\$1.10/hr × 8-12h = \$8.80-13.20" ;;
    *)               echo "  Est. cost:     (unknown machine type)" ;;
esac

echo "============================================"
echo ""

# ─── Submit ───────────────────────────────────────────────────────────────────

echo "[SUBMITTING] gcloud batch jobs submit ${JOB_NAME} ..."
gcloud batch jobs submit "${JOB_NAME}" \
    --location="${REGION}" \
    --config="${TMPJOB}"

echo ""
echo "[✓] Job submitted: ${JOB_NAME}"
echo ""
echo "Monitor with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --location=${REGION}"
echo ""
echo "Stream logs:"
echo "  gcloud batch jobs logs ${JOB_NAME} --location=${REGION}"
echo ""
echo "Cancel if needed:"
echo "  gcloud batch jobs delete ${JOB_NAME} --location=${REGION}"
echo "============================================"
