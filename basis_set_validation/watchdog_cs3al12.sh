#!/bin/bash
# Watchdog for cs3al12_tzvp_extract.py
# Checks every 15 min if the process is alive. If dead, logs reason & restarts.
# Usage: nohup bash watchdog_cs3al12.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="cs3al12_tzvp_extract.py"
LOGFILE="$SCRIPT_DIR/cs3al12_tzvp_extract.log"
WATCHLOG="$SCRIPT_DIR/watchdog_cs3al12.log"
CHECK_INTERVAL=900  # 15 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCHLOG"; }

start_calc() {
    log "Starting $SCRIPT..."
    source /home/ardac/miniconda3/etc/profile.d/conda.sh
    conda activate fbond-env
    cd "$SCRIPT_DIR"
    nohup python3 "$SCRIPT" >> "$LOGFILE" 2>&1 &
    local pid=$!
    log "Started with PID $pid"
    echo "$pid"
}

log "═══ Watchdog started ═══"

# Check if already running
PID=$(pgrep -f "python.*$SCRIPT" | head -1)
if [ -z "$PID" ]; then
    log "Process not found. Starting..."
    PID=$(start_calc)
else
    log "Process already running (PID $PID)"
fi

while true; do
    sleep "$CHECK_INTERVAL"

    # Check if results JSON exists (job finished)
    if [ -f "$SCRIPT_DIR/Cs3Al12_minus_def2tzvp_results.json" ]; then
        log "✓ Results JSON found — calculation COMPLETE. Watchdog exiting."
        exit 0
    fi

    # Check if process is alive
    if pgrep -f "python.*$SCRIPT" > /dev/null 2>&1; then
        MEM=$(pgrep -f "python.*$SCRIPT" | head -1 | xargs -I{} ps -o rss= -p {} 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}')
        log "✓ Process alive (RSS: ${MEM:-?} GB)"
    else
        # Process died — check why
        TAIL=$(tail -5 "$LOGFILE" 2>/dev/null)
        log "✗ Process DIED. Last log lines:"
        log "$TAIL"

        # Check for OOM
        if dmesg -T 2>/dev/null | tail -20 | grep -qi "killed process"; then
            log "  → Likely OOM kill detected in dmesg"
        fi

        # Check if CCSD amplitudes were saved (partial success)
        if [ -f "$SCRIPT_DIR/cs3al12_tzvp_ccsd_amplitudes.npz" ]; then
            log "  → CCSD amplitudes checkpoint EXISTS — restart will skip CCSD"
        fi

        log "  → Restarting..."
        PID=$(start_calc)
    fi
done
