#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/heyyubov/Desktop/Projects/Medical-Imaging"
DATA_SRC="/Users/heyyubov/Downloads/chest_xray/chest_xray"

cd "$PROJECT_DIR"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 не найден. Установи: brew install python@3.11"
  exit 1
fi

rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if [ ! -d "$DATA_SRC/train" ] || [ ! -d "$DATA_SRC/val" ] || [ ! -d "$DATA_SRC/test" ]; then
  echo "Не найден dataset в $DATA_SRC"
  exit 1
fi

rm -rf data/processed/chest_xray
mkdir -p data/processed/chest_xray
rsync -a "$DATA_SRC"/ data/processed/chest_xray/

sed -i '' 's/use_fake_data: true/use_fake_data: false/g' configs/centralized.yaml configs/fedavg.yaml configs/fedprox.yaml
sed -i '' 's|data_dir: .*|data_dir: data/processed/chest_xray|g' configs/centralized.yaml configs/fedavg.yaml configs/fedprox.yaml

rm -f data/splits/*.json
find results/metrics -mindepth 1 -not -name '.gitkeep' -delete
find results/plots -mindepth 1 -not -name '.gitkeep' -delete
find results/checkpoints -mindepth 1 -not -name '.gitkeep' -delete

bash scripts/run_all.sh

echo "Done:"
echo " - results/metrics/comparison_table.csv"
echo " - results/metrics/*_clinic_summary.csv"
echo " - results/plots/*_clinic_distribution.png"
echo " - REPORT.md"
