#!/bin/bash
# cycle.sh — Auto-fetch Claude batch results in a self-healing loop.
#
# Polls every $INTERVAL seconds. Each cycle calls retrieve_batch.py, which
# fetches whatever batches have ended and merges them into
# results/claude_batch_results.json (latest wins per custom_id, so re-runs
# never erase past answers). Exits cleanly once every batch in
# batches/batch_state.json has ended.
#
# Usage (run from anywhere; the script cds to the repo root):
#   bash src/evaluation/closed_source/claude/cycle.sh                 # 180s default
#   bash src/evaluation/closed_source/claude/cycle.sh 60              # custom interval
#
# Background it so you don't have to keep a terminal open:
#   nohup bash src/evaluation/closed_source/claude/cycle.sh \
#       > src/evaluation/closed_source/claude/cycle.log 2>&1 &
#   tail -f src/evaluation/closed_source/claude/cycle.log
#
# Ctrl+C / kill at any time — rerun and it resumes where it left off.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

STATE="${STATE:-$SCRIPT_DIR/batches/batch_state.json}"
RESULTS="${RESULTS:-$SCRIPT_DIR/results/claude_batch_results.json}"
LOG="$SCRIPT_DIR/cycle.log"

INTERVAL="${1:-180}"
MAX_CONSECUTIVE_FAILURES=10

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

# Returns "ok|ended|total|pending_summary" by querying the API.
batch_status() {
    "$PYTHON" - <<PY 2>/dev/null
import json, os
from dotenv import load_dotenv
for n in (".env", "secrets.env"):
    p = os.path.join("$SCRIPT_DIR", n)
    if os.path.isfile(p):
        load_dotenv(p, override=False)
import anthropic
try:
    state = json.load(open("$STATE"))
except Exception:
    print("no_state|0|0|missing")
    raise SystemExit(0)
batches = state.get("batches", [])
if not batches:
    print("no_batches|0|0|empty")
    raise SystemExit(0)
client = anthropic.Anthropic()
ended = 0
pending = []
for b in batches:
    try:
        info = client.messages.batches.retrieve(b["batch_id"])
        s = info.processing_status
    except Exception:
        pending.append(f"{b['batch_id']}=API_ERR")
        continue
    if s == "ended":
        ended += 1
    else:
        pending.append(f"{b['batch_id'][-8:]}={s}")
print(f"ok|{ended}|{len(batches)}|{','.join(pending) or 'all_ended'}")
PY
}

run_retrieve() {
    log "Retrieving results..."
    "$PYTHON" "$SCRIPT_DIR/retrieve_batch.py" \
        --state "$STATE" \
        --output "$RESULTS" 2>&1 | tee -a "$LOG"
    return ${PIPESTATUS[0]}
}

result_summary() {
    "$PYTHON" - <<PY 2>/dev/null
import json
try:
    r = json.load(open("$RESULTS"))
except Exception:
    print("0|0|0"); raise SystemExit(0)
ok = sum(1 for x in r if x.get("answer") and not x.get("error"))
err = sum(1 for x in r if x.get("error") or not x.get("answer"))
print(f"{ok}|{err}|{len(r)}")
PY
}

# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
log "============================================================"
log "Starting cycle.sh (interval=${INTERVAL}s, state=$STATE)"
log "============================================================"

consecutive_failures=0

while true; do
    status_line=$(batch_status)
    tag="${status_line%%|*}"
    rest="${status_line#*|}"
    done_n="${rest%%|*}"
    rest="${rest#*|}"
    total="${rest%%|*}"
    info="${rest#*|}"

    if [ "$tag" = "no_state" ] || [ "$tag" = "no_batches" ]; then
        log "No batches found in $STATE. Submit one with run_claude.py first."
        exit 1
    fi

    if [ "$tag" != "ok" ]; then
        consecutive_failures=$((consecutive_failures + 1))
        log "API status check failed ($consecutive_failures/$MAX_CONSECUTIVE_FAILURES)."
        if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
            log "Too many consecutive failures. Exiting."
            exit 1
        fi
        sleep "$INTERVAL"
        continue
    fi
    consecutive_failures=0

    log "Batches: $done_n/$total ended. Pending: $info"

    # Always retrieve — picks up any newly-ended batches and merges them.
    run_retrieve || log "retrieve_batch.py exited non-zero (will retry)."

    if [ "$done_n" = "$total" ]; then
        log "All $total batch(es) ended. Done."
        summary=$(result_summary)
        s_ok="${summary%%|*}"; rest="${summary#*|}"
        s_err="${rest%%|*}"; s_tot="${rest#*|}"
        log "Final: $s_tot results — $s_ok OK, $s_err errors."
        log "Results: $RESULTS"
        exit 0
    fi

    sleep "$INTERVAL"
done
