#!/bin/bash
# cycle.sh — Robust, self-healing batch submission loop.
#
# Handles:
#   - Batches still in progress → waits and retries
#   - Transient API / network errors → retries with backoff
#   - Per-request 500s inside completed batches → auto-builds a new retry
#     round and keeps going until everything succeeds or a hard cap is hit
#   - Ctrl+C → clean exit; rerun to resume exactly where you left off
#
# Usage:
#   bash src/evaluation/closed_source/gpt/cycle.sh              # one cycle
#   bash src/evaluation/closed_source/gpt/cycle.sh --loop       # keep going (3 min default)
#   bash src/evaluation/closed_source/gpt/cycle.sh --loop 300   # custom interval (seconds)
#
# Run from the repo root with the venv activated.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# Use uv to run python with the correct environment.
PYTHON="uv run python"

RETRY_STATE="$SCRIPT_DIR/retry/retry_state.json"
RETRY_RESULTS="$SCRIPT_DIR/results/gpt_retry_results.json"
BATCH_RESULTS="$SCRIPT_DIR/results/gpt_batch_results.json"
ALL_RESULTS="$SCRIPT_DIR/results/gpt_all_results.json"
LOG="$SCRIPT_DIR/cycle.log"
MAX_RETRY_ROUNDS=5
MAX_CONSECUTIVE_FAILURES=10

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

run_py() {
    # Run a python command, capture output, log it, return exit code.
    local output
    output=$( "$@" 2>&1 ) || {
        local rc=$?
        log "FAIL (rc=$rc): $*"
        log "$output"
        echo "$output"
        return $rc
    }
    echo "$output"
    return 0
}

get_pending_count() {
    $PYTHON -c "
import json, sys
try:
    s = json.load(open('$RETRY_STATE'))
    submitted = {b['jsonl_path'] for b in s.get('batches', [])}
    pending = [p for p in s.get('parts', []) if p['jsonl_path'] not in submitted]
    print(len(pending))
except Exception as e:
    print(-1)
" 2>/dev/null
}

get_last_batch_status() {
    $PYTHON -c "
import json, os, sys
from dotenv import load_dotenv
load_dotenv('$SCRIPT_DIR/.env', override=False)
load_dotenv('$SCRIPT_DIR/secrets.env', override=False)
from openai import OpenAI
client = OpenAI()
s = json.load(open('$RETRY_STATE'))
batches = s.get('batches', [])
if not batches:
    print('no_batches')
    sys.exit(0)
last = batches[-1]
b = client.batches.retrieve(last['batch_id'])
rc = b.request_counts
print(f'{b.status}|{rc.completed}|{rc.total}|{rc.failed}')
" 2>/dev/null || echo "api_error"
}

count_errors_in_results() {
    $PYTHON -c "
import json
try:
    r = json.load(open('$ALL_RESULTS'))
    err = sum(1 for x in r if x.get('error') or not x.get('answer'))
    ok = sum(1 for x in r if x.get('answer') and not x.get('error'))
    print(f'{ok}|{err}|{len(r)}')
except:
    print('0|0|0')
" 2>/dev/null
}

# -------------------------------------------------------
# One cycle: retrieve → merge → maybe submit
# -------------------------------------------------------
one_cycle() {
    log "======== cycle start ========"

    # Step 1: Check if the last submitted batch is done
    local status_line
    status_line=$(get_last_batch_status)
    local status="${status_line%%|*}"

    if [ "$status" = "no_batches" ]; then
        log "No batches submitted yet. Submitting first one."
    elif [ "$status" = "api_error" ]; then
        log "API error checking batch status — will retry next cycle."
        return 1
    elif [ "$status" != "completed" ] && [ "$status" != "failed" ] && \
         [ "$status" != "expired" ] && [ "$status" != "cancelled" ]; then
        log "Last batch still $status ($status_line). Waiting."
        return 1
    else
        log "Last batch: $status_line"
    fi

    # Step 2: Retrieve results
    log "Retrieving results..."
    local retrieve_out
    retrieve_out=$(run_py $PYTHON "$SCRIPT_DIR/retrieve_batch.py" \
        --state "$RETRY_STATE" \
        --output "$RETRY_RESULTS") || {
        log "Retrieve failed — will retry next cycle."
        return 1
    }
    echo "$retrieve_out" | tail -3

    # Step 3: Merge
    if [ -f "$BATCH_RESULTS" ] && [ -f "$RETRY_RESULTS" ]; then
        log "Merging results..."
        local merge_out
        merge_out=$(run_py $PYTHON "$SCRIPT_DIR/merge_results.py" \
            "$BATCH_RESULTS" "$RETRY_RESULTS" \
            -o "$ALL_RESULTS") || {
            log "Merge failed — will retry next cycle."
            return 1
        }
        echo "$merge_out"
    elif [ -f "$RETRY_RESULTS" ]; then
        cp "$RETRY_RESULTS" "$ALL_RESULTS"
    fi

    # Step 4: Check how many parts are left to submit
    local pending
    pending=$(get_pending_count)
    log "Pending parts: $pending"

    if [ "$pending" = "0" ]; then
        log "All parts submitted."
        return 0
    fi

    # Step 5: Submit next part
    log "Submitting next batch..."
    local submit_out
    submit_out=$(run_py $PYTHON "$SCRIPT_DIR/retry_failed.py" submit) || {
        log "Submit failed — will retry next cycle."
        return 1
    }
    echo "$submit_out" | tail -3

    log "======== cycle done ========"
    return 0
}

# -------------------------------------------------------
# Post-completion: check for per-request errors and auto-retry
# -------------------------------------------------------
auto_retry_errors() {
    local round_num=$1
    local counts
    counts=$(count_errors_in_results)
    local ok="${counts%%|*}"
    local rest="${counts#*|}"
    local err="${rest%%|*}"
    local total="${rest#*|}"

    log "Results: $ok OK, $err errors, $total total."

    if [ "$err" = "0" ]; then
        log "No errors — all done! 🎉"
        return 0
    fi

    if [ "$round_num" -ge "$MAX_RETRY_ROUNDS" ]; then
        log "Hit max retry rounds ($MAX_RETRY_ROUNDS). $err errors remain."
        log "Inspect with: $PYTHON $SCRIPT_DIR/diagnose_batches.py --state $RETRY_STATE"
        return 1
    fi

    log "$err errors remain. Starting retry round $((round_num + 1))..."

    # Back up current retry state and build a new one from itself
    local prev_state="$RETRY_STATE"
    local new_retry_dir="$SCRIPT_DIR/retry_round$((round_num + 1))"

    log "Building retry round $((round_num + 1))..."
    local build_out
    build_out=$(run_py $PYTHON "$SCRIPT_DIR/retry_failed.py" build \
        --state "$prev_state" \
        --retry-dir "$new_retry_dir") || {
        log "Retry build failed."
        return 1
    }
    echo "$build_out"

    # Point state to the new round
    RETRY_STATE="$new_retry_dir/retry_state.json"
    RETRY_RESULTS="$SCRIPT_DIR/results/gpt_retry${round_num}_results.json"

    return 2  # signal: more work to do
}

# -------------------------------------------------------
# Parse args
# -------------------------------------------------------
LOOP=false
INTERVAL=180

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loop)
            LOOP=true
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                INTERVAL="$2"
                shift
            fi
            shift
            ;;
        *)
            echo "Usage: $0 [--loop [SECONDS]]"
            exit 1
            ;;
    esac
done

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if [ "$LOOP" = false ]; then
    one_cycle
    exit $?
fi

log "Starting loop (interval=${INTERVAL}s, max_retry_rounds=$MAX_RETRY_ROUNDS)"
consecutive_failures=0
retry_round=0

while true; do
    one_cycle
    rc=$?

    if [ $rc -eq 0 ]; then
        consecutive_failures=0
        # All parts submitted — check if everything is done
        pending=$(get_pending_count)
        if [ "$pending" = "0" ]; then
            # Wait for the very last batch to finish
            log "All parts submitted. Waiting for last batch to complete..."
            sleep "$INTERVAL"
            # One more retrieve+merge
            one_cycle 2>/dev/null || true

            # Check for per-request errors and auto-retry
            auto_retry_errors "$retry_round"
            auto_rc=$?
            if [ $auto_rc -eq 0 ]; then
                break  # all done
            elif [ $auto_rc -eq 2 ]; then
                retry_round=$((retry_round + 1))
                log "Continuing with retry round $retry_round..."
                continue
            else
                break  # hit max rounds
            fi
        fi
    else
        consecutive_failures=$((consecutive_failures + 1))
        if [ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]; then
            log "ERROR: $MAX_CONSECUTIVE_FAILURES consecutive failures. Exiting."
            log "Fix the issue and rerun — it will resume where it left off."
            exit 1
        fi
    fi

    sleep "$INTERVAL"
done

# Final summary
log ""
log "========== FINAL SUMMARY =========="
counts=$(count_errors_in_results)
ok="${counts%%|*}"
rest="${counts#*|}"
err="${rest%%|*}"
total="${rest#*|}"
log "Total: $total results — $ok OK, $err errors"
log "Results: $ALL_RESULTS"
log "Log: $LOG"
