#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

inRoot="${inRoot:-${1:-eval/results/unified13_final}}"
goldRoot="${goldRoot:-${2:-dataset/test_data_with_level}}"
outCsv="${outCsv:-${3:-eval/results/$(basename "$inRoot")_vs_gold_decision.csv}}"

python eval/calculate_gold_decision_accuracy.py "$inRoot" \
  --gold_dir "$goldRoot" \
  -o "$outCsv"
