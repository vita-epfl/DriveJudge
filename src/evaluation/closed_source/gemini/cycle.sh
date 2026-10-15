#!/bin/bash
# cycle.sh — Auto-fetch Gemini batch results in a self-healing loop.
#
# Polls every $INTERVAL seconds. Each cycle calls retrieve_batch.py, which
# fetches whatever batches are SUCCEEDED and merges them into
# results/gemini_batch_results.json (latest wins per custom_id, so re-runs
# never erase past answers). Exits cleanly once every batch in
# batches/batch_state.json is in a terminal state.
#
# Usage (run from anywhere; the script cds to its own folder):
#   bash src/evaluation/closed_source/gemini/cycle.sh                 # 180s default
#   bash src/evaluation/closed_source/gemini/cycle.sh 60              # custom interval
#
# Background it so you don't have to keep a terminal open:
#   nohup bash src/evaluation/closed_source/gemini/cycle.sh \
#       > src/evaluation/closed_source/gemini/cycle.log 2>&1 &
#   tail -f src/evaluation/closed_source/gemini/cycle.log
#
# Ctrl+C / kill at any time — rerun and it resumes where it left off.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="uv run python"

STATE="$SCRIPT_DIR/batches/batch_state.json"
RESULTS="$SCRIPT_DIR/results/gemini_batch_results.json"
LOG="$SCRIPT_DIR/cycle.log"

INTERVAL="${1:-180}"
MAX_CONSECUTIVE_FAILURES=10

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

# Returns "done|total|nonterminal_state_summary" by querying the API.
batch_status() {
    $PYTHON - <<PY 2>/dev/null
import json, os
from dotenv import load_dotenv
for n in (".env", "secrets.env"):
    p = os.path.join("$SCRIPT_DIR", n)
    if os.path.isfile(p):
        load_dotenv(p, override=False)
from google import genai
TERMINAL = {"JOB_STATE_SUCCEEDED","JOB_STATE_FAILED","JOB_STATE_CANCELLED","JOB_STATE_EXPIRED"}
try:
    state = json.load(open("$STATE"))
except Exception:
    print("no_state|0|0|missing")
    raise SystemExit(0)
batches = state.get("batches", [])
if not batches:
    print("no_batches|0|0|empty")
    raise SystemExit(0)
client = genai.Client()
done = 0
non_terminal = []
for b in batches:
    try:
        info = client.batches.get(name=b["batch_name"])
        s = info.state.name if hasattr(info.state, "name") else str(info.state)
    except Exception as e:
        non_terminal.append(f"{b['batch_name']}=API_ERR")
        continue
    if s in TERMINAL:
        done += 1
    else:
        non_terminal.append(f"{b['batch_name'].split('/')[-1]}={s}")
print(f"ok|{done}|{len(batches)}|{','.join(non_terminal) or 'all_terminal'}")
PY
}

run_retrieve() {
    log "Retrieving results..."
    $PYTHON "$SCRIPT_DIR/retrieve_batch.py" \
        --state "$STATE" \
        --output "$RESULTS" 2>&1 | tee -a "$LOG"
    return ${PIPESTATUS[0]}
}

result_summary() {
    $PYTHON - <<PY 2>/dev/null
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
    done="${rest%%|*}"
    rest="${rest#*|}"
    total="${rest%%|*}"
    info="${rest#*|}"

    if [ "$tag" = "no_state" ] || [ "$tag" = "no_batches" ]; then
        log "No batches found in $STATE. Submit one with run_gemini.py first."
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

    log "Batches: $done/$total terminal. Pending: $info"

    # Always retrieve — picks up any newly-succeeded batches and merges them.
    run_retrieve || log "retrieve_batch.py exited non-zero (will retry)."

    if [ "$done" = "$total" ]; then
        log "All $total batch(es) terminal. Done."
        summary=$(result_summary)
        s_ok="${summary%%|*}"; rest="${summary#*|}"
        s_err="${rest%%|*}"; s_tot="${rest#*|}"
        log "Final: $s_tot results — $s_ok OK, $s_err errors."
        log "Results: $RESULTS"
        exit 0
    fi

    sleep "$INTERVAL"
done
