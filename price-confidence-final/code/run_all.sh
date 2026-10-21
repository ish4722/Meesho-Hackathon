#!/usr/bin/env bash
# Price Confidence v2: regenerate everything from scratch and check it. Run from this folder (code/).
# Takes about 10 minutes, most of it the decision evaluation against the generator.
#   ./run_all.sh          full run
#   ./run_all.sh --quick  smaller evaluation (smoke test, about 1 minute)
set -euo pipefail
cd "$(dirname "$0")"
QUICK=""
if [[ "${1:-}" == "--quick" ]]; then QUICK="--quick"; fi

echo "== 1/7 synthetic data (seed 20261021)"
python generate_dataset.py --out .
echo "== 2/7 data-quality report"
python data_quality.py --data . --out reports
echo "== 3/7 models (all data) -> model.json"
python train_models.py --data . --out model.json
echo "== 4/7 tests (Python + engine.js)"
python -m pytest tests -q
echo "== 5/7 evaluation (train/validation/holdout, generator-judged decisions)"
python evaluate.py --out reports $QUICK
echo "== 6/7 offline app"
python build_app.py
cp dist/index.html ../OPEN_THIS_Price_Confidence_demo.html
echo "== 7/7 Excel copy of the dataset"
python build_workbook.py --out ../Price_Confidence_Synthetic_Dataset.xlsx
python - <<'PY'
import hashlib, json
m = json.load(open("dataset_manifest.json"))
bad = [f for f, h in m["files"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != h]
print("manifest hashes:", "OK" if not bad else f"MISMATCH {bad}")
PY
echo "done: reports/data_quality.md, reports/evaluation.md, ../OPEN_THIS_Price_Confidence_demo.html"
