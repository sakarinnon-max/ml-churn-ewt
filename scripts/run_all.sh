#!/bin/bash
# Gate: execute every notebook top-to-bottom on sample data (headless).
# Usage: bash scripts/run_all.sh [chapter-number ...]   e.g. bash scripts/run_all.sh 00 04
set -u -o pipefail
cd "$(dirname "$0")/.."
# เรียก jupyter-nbconvert ของ venv ตรงๆ เท่านั้น — ห้ามใช้ "python -m jupyter nbconvert"
# เพราะมันค้นหา jupyter-nbconvert ผ่าน PATH แล้วไปเจอของ Anaconda (kernel คนละ sklearn!)
NBC=./venv/bin/jupyter-nbconvert
export ML_CHURN_DATA=sample

FAIL=0
if [ $# -gt 0 ]; then
  NBS=""
  for c in "$@"; do NBS="$NBS notebooks/ch${c}*.ipynb"; done
else
  NBS="notebooks/ch*.ipynb"
fi

for nb in $NBS; do
  [ -e "$nb" ] || continue
  echo "=== executing $nb"
  if $NBC --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=600 "$nb" 2>&1 | tail -3; then
    echo "    OK"
  else
    echo "    FAILED: $nb"
    FAIL=1
  fi
done
exit $FAIL
