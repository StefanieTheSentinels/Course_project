#!/bin/bash
set -e

pip install -r requirements.txt
for i in $(seq 1 5); do
    echo "±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± Run $i ±±±±±±±±±"
    SEED=$RANDOM
    python3 generate_dataset.py --n 120000 --seed $SEED
    python3 train_models.py
    python3 optimize.py --seed $SEED --plot-out convergence_$i.gif
    python3 sensitivity_analysis.py
    pytest tests/ -v
done