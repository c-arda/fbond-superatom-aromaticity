#!/bin/bash
# ============================================================================
# Fbond Round 2 — Resubmit all systems with frozen-core fix
# ============================================================================
# Uses the WORKING config from the previous batch (us-east1, gcr.io, /mnt/disks/gcs/)
#
# Usage:
#   ./batch/submit_round2.sh              # Submit all 10 (skip Au13)
#   ./batch/submit_round2.sh --with-au13  # Submit all 11 (Au13 with TZVP)
#   ./batch/submit_round2.sh --dry-run    # Show what would be submitted
# ============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

REGION="us-east1"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
IMAGE_URI="gcr.io/${PROJECT_ID}/fbond:latest"
RESULTS_BUCKET="fbond-results-${PROJECT_ID}"
CHECKPOINTS_BUCKET="fbond-checkpoints-${PROJECT_ID}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse args
DRY_RUN=false
WITH_AU13=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --with-au13)  WITH_AU13=true ;;
    esac
done

# ─── Systems ──────────────────────────────────────────────────────────────────

# All def2-SVP systems (10 systems, skip Au13 by default)
SVP_SYSTEMS=(
    "Al4_2minus"
    "Al4_4minus"
    "Al4_4minus_triplet"
    "B12_planar"
    "B12_icosahedral"
    "B6N6_planar"
    # "B12N12_cage"  # deferred: geometry has 12 atoms (BN ring), not 24-atom cage. See plans/load-skillsets-for-this-magical-mccarthy.md
    "C6H6_benzene"
    "Cs3Al8"
    "Cs3Al12"
)

# ─── Submit function ─────────────────────────────────────────────────────────

submit_job() {
    local SYSTEM_NAME="$1"
    local BASIS="${2:-def2-SVP}"
    local MACHINE_TYPE="${3:-n2-highmem-16}"

    local TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    local JOB_NAME="fbond-$(echo "${SYSTEM_NAME}" | tr '[:upper:]_' '[:lower:]-')-${TIMESTAMP}"

    # Truncate job name to 63 chars (GCP limit)
    JOB_NAME="${JOB_NAME:0:63}"

    # Auto-detect compute resources
    local VCPUS=$(echo "$MACHINE_TYPE" | grep -oE '[0-9]+$')
    local CPU_MILLI=$((VCPUS * 1000))
    local MEM_MIB
    if [[ "$MACHINE_TYPE" == *"highmem"* ]]; then
        MEM_MIB=$((VCPUS * 8 * 1024 - 2048))
    else
        MEM_MIB=$((VCPUS * 4 * 1024 - 2048))
    fi
    local MAX_MEMORY=$((MEM_MIB - 8000))  # Reserve 8GB for OS

    echo "  ${SYSTEM_NAME:0:25}  basis=${BASIS}  machine=${MACHINE_TYPE}  job=${JOB_NAME}"

    if $DRY_RUN; then
        return 0
    fi

    # Generate job JSON inline
    local TMPJOB=$(mktemp /tmp/fbond-batch-XXXXXX.json)
    trap "rm -f $TMPJOB" RETURN

    cat > "$TMPJOB" <<EOF
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "${IMAGE_URI}",
              "commands": [
                "--system-file", "/app/geometries/${SYSTEM_NAME}.json",
                "--output-dir", "/mnt/disks/gcs/output",
                "--checkpoint-dir", "/mnt/disks/gcs/checkpoints",
                "--scratch-dir", "/tmp/pyscf_scratch",
                "--basis", "${BASIS}",
                "--max-memory", "${MAX_MEMORY}"
              ],
              "volumes": [
                "/mnt/disks/gcs/output",
                "/mnt/disks/gcs/checkpoints",
                "/mnt/disks/gcs/output:/mnt/disks/gcs/output:rw",
                "/mnt/disks/gcs/checkpoints:/mnt/disks/gcs/checkpoints:rw"
              ]
            }
          }
        ],
        "computeResource": {
          "cpuMilli": ${CPU_MILLI},
          "memoryMib": ${MEM_MIB}
        },
        "volumes": [
          {
            "gcs": {
              "remotePath": "${RESULTS_BUCKET}"
            },
            "mountPath": "/mnt/disks/gcs/output"
          },
          {
            "gcs": {
              "remotePath": "${CHECKPOINTS_BUCKET}"
            },
            "mountPath": "/mnt/disks/gcs/checkpoints"
          }
        ],
        "maxRunDuration": "86400s",
        "maxRetryCount": 2
      },
      "taskCount": 1,
      "parallelism": 1
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "machineType": "${MACHINE_TYPE}",
          "provisioningModel": "STANDARD",
          "bootDisk": {
            "sizeGb": "100",
            "type": "pd-balanced"
          }
        }
      }
    ],
    "location": {
      "allowedLocations": [
        "regions/us-east1",
        "zones/us-east1-b",
        "zones/us-east1-c",
        "zones/us-east1-d"
      ]
    }
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

    gcloud batch jobs submit "${JOB_NAME}" \
        --location="${REGION}" \
        --config="${TMPJOB}" \
        --quiet 2>&1 | tail -1

    rm -f "$TMPJOB"
    # Stagger submissions to avoid quota spike
    sleep 3
}

# ─── Main ─────────────────────────────────────────────────────────────────────

echo "============================================"
echo "FBOND ROUND 2 — FROZEN-CORE FIX"
echo "============================================"
echo "  Region:    ${REGION}"
echo "  Image:     ${IMAGE_URI}"
echo "  Results:   gs://${RESULTS_BUCKET}/"
echo "  Systems:   ${#SVP_SYSTEMS[@]} SVP"
if $WITH_AU13; then
    echo "  + Au13⁻:   def2-TZVP on n2-highmem-32"
fi
if $DRY_RUN; then
    echo "  MODE:      DRY RUN (no jobs submitted)"
fi
echo "============================================"
echo ""

# Submit SVP systems
echo "[SUBMITTING] def2-SVP systems..."
for sys in "${SVP_SYSTEMS[@]}"; do
    submit_job "$sys" "def2-SVP" "n2-highmem-16"
done

# Submit Au13 with TZVP if requested
if $WITH_AU13; then
    echo ""
    echo "[SUBMITTING] Au13⁻ with def2-TZVP..."
    submit_job "Au13_minus" "def2-TZVP" "n2-highmem-32"
fi

echo ""
echo "============================================"
echo "[✓] All jobs submitted!"
echo ""
echo "Monitor:"
echo "  gcloud batch jobs list --location=${REGION} --format='table(name.basename(),status.state)'"
echo ""
echo "Stream logs for a job:"
echo "  gcloud logging read 'textPayload=~\"CCSD|F_bond|ERROR\" AND resource.type=~\"batch\"' --limit=20 --freshness=1h"
echo "============================================"
