#!/bin/bash
set -e

pip install -r requirements.txt

rm -rf runs
mkdir -p runs

for i in $(seq 1 5); do
    echo "======== Run $i ========"
    SEED=$RANDOM
    OUT="runs/run_$i"
    mkdir -p "$OUT"

    python3 generate_dataset.py --n 120000 --seed $SEED \
        --out "$OUT/synthetic_dataset.csv"

    python3 train_models.py \
        --data   "$OUT/synthetic_dataset.csv" \
        --out    "$OUT/best_model.pkl" \
        --report "$OUT/training_report.txt"

    python3 optimize.py --seed $SEED --target both \
        --model        "$OUT/best_model.pkl" \
        --csv-oracle   "$OUT/optimization_results_oracle.csv" \
        --csv-ml       "$OUT/optimization_results_ml.csv" \
        --csv-crossval "$OUT/optimization_results_crossval.csv" \
        --gif-oracle   "$OUT/convergence_oracle.gif" \
        --gif-ml       "$OUT/convergence_ml.gif"

    pytest tests/ -v
done

python3 aggregate_runs.py
python3 plot_gap.py