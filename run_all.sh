#!/usr/bin/env bash
# Runs every demo under MOCK_LLM and saves transcripts/. Needs no API keys.
set -uo pipefail
cd "$(dirname "$0")"
export MOCK_LLM=true CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true
export PYTHONUNBUFFERED=1            # stream output through tee instead of buffering it
mkdir -p transcripts logs
: > logs/requests.jsonl
PY="${PYTHON:-python}"
echo "[preflight] using: $($PY -c 'import sys; print(sys.executable)')"

FAILED=()
run() {
  echo; echo "=== $(date +%H:%M:%S) >>> $1 (first run of 02 downloads the embedding model: be patient)"
  $PY -u "scripts/$1.py" 2>&1 | tee "transcripts/$1.txt"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "!!! $1 FAILED (exit $rc) - see transcripts/$1.txt"; FAILED+=("$1"); fi
  return 0
}

echo "=== $(date +%H:%M:%S) >>> tests"
$PY -u -m pytest -q tests 2>&1 | tee transcripts/00_tests.txt
run 01_dataset
run 02_calibrate      # writes calibration.json (needed by everything below)
run 03_chunking_eval  # writes transcripts/chunking_eval.json
run 04_grounded_demo
run 05_crew_demo
run 06_memory_demo
run 07_guardrails_demo
run 08_api_demo
run 09_eval
run 10_review_demo
run 11_governance_demo
run 12_cache_demo
echo; echo "Done. Failed steps: ${FAILED[*]:-none}"
