#!/bin/bash
set -e

pip install -r requirements.txt

rm -f optimization_results_*.csv

for i in $(seq 1 5); do
    echo "======== Run $i ========"
    SEED=$RANDOM

    python3 generate_dataset.py --n 120000 --seed $SEED
    python3 train_models.py
    python3 optimize.py --seed $SEED \
        --plot-out convergence_$i.gif \
        --csv-out  optimization_results_$i.csv
    pytest tests/ -v
done

python3 aggregate_runs.py