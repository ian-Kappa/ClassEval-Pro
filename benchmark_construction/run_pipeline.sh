#!/usr/bin/env bash
#
# Complete benchmark data-generation script.
# Runs the full construction pipeline (stages 1..5):
#   mine -> extract -> merge/dedup -> compose -> generate
#
# Usage:
#   cd <repo root>
#   bash benchmark_construction/run_pipeline.sh            # all stages
#   bash benchmark_construction/run_pipeline.sh --from 4   # resume at stage 4
#   bash benchmark_construction/run_pipeline.sh --only 4 5
#
# Fill in the credentials / model deployment names below (or export them
# beforehand). Nothing is hardcoded in the source.

set -euo pipefail

# ---- Repo root (parent of this script's directory) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ---- Secrets (stage 1 & 5) ----
export GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_GITHUB_TOKEN}"
export AZURE_API_KEY="${AZURE_API_KEY:-YOUR_API_KEY}"
export AZURE_ENDPOINT="${AZURE_ENDPOINT:-YOUR_API_ENDPOINT}"

# ---- Model deployment names (stage 5) ----
export SKELETON_MODEL="${SKELETON_MODEL:-YOUR_SKELETON_MODEL}"
export TESTCASE_MODEL="${TESTCASE_MODEL:-YOUR_TESTCASE_MODEL}"
export SOLUTION_MODEL="${SOLUTION_MODEL:-YOUR_SOLUTION_MODEL}"
export JUDGE_MODEL_1="${JUDGE_MODEL_1:-YOUR_JUDGE_MODEL_1}"
export JUDGE_MODEL_2="${JUDGE_MODEL_2:-YOUR_JUDGE_MODEL_2}"
export JUDGE_MODEL_3="${JUDGE_MODEL_3:-YOUR_JUDGE_MODEL_3}"

# ---- Fail fast if placeholders were left unfilled ----
for v in GITHUB_TOKEN AZURE_API_KEY AZURE_ENDPOINT \
         SKELETON_MODEL TESTCASE_MODEL SOLUTION_MODEL \
         JUDGE_MODEL_1 JUDGE_MODEL_2 JUDGE_MODEL_3; do
    val="${!v}"
    if [[ "$val" == YOUR_* ]]; then
        echo "ERROR: environment variable '$v' is still a placeholder ('$val')."
        echo "       Export it or edit benchmark_construction/run_pipeline.sh."
        exit 1
    fi
done

echo "======================================================================"
echo "ClassEval-Pro benchmark construction"
echo "Repo root : $REPO_ROOT"
echo "Stages    : ${*:-1..5 (all)}"
echo "Output    : benchmark_construction/artifacts/dataset.json"
echo "======================================================================"

# ---- Run the orchestrator (passes through --from/--to/--only) ----
python -m benchmark_construction.run "$@"

echo "======================================================================"
echo "Done. Final benchmark -> benchmark_construction/artifacts/dataset.json"
echo "Intermediate artifacts -> benchmark_construction/artifacts/*.json"
echo "======================================================================"
