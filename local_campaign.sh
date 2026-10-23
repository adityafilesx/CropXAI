#!/bin/bash

# ==============================================================================
# CropXAI Laptop-Scale Scientific Campaign
# ==============================================================================
# This script executes a computationally feasible empirical grid suitable for 
# a local Apple Silicon Mac (M-series).
# It trains 4 representative architectures on EuroSAT across 3 statistical seeds
# for 5 epochs each, generating exactly 12 provenance records in the SQLite DB.
# ==============================================================================

echo "============================================================"
echo "Initializing Laptop-Scale Campaign (12 Permutations)"
echo "============================================================"

# Ensure virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment 'venv' not found."
        exit 1
    fi
fi

# Configuration
MODELS=("ResNet-18" "ConvNeXt-T" "ViT-Base" "Prithvi-100M")
DATASET="eurosat"
SEEDS=(42 100 999)
EPOCHS=5

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#SEEDS[@]} ))
CURRENT_RUN=1

export PYTHONPATH=.

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Executing Run [$CURRENT_RUN/$TOTAL_RUNS]: Model=$MODEL | Dataset=$DATASET | Seed=$SEED"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        ./cropxai campaign run --model "$MODEL" --dataset "$DATASET" --seed "$SEED" --epochs "$EPOCHS"
        
        if [ $? -ne 0 ]; then
            echo "Error: Run failed for $MODEL (seed=$SEED). Continuing to next run..."
        fi
        
        CURRENT_RUN=$((CURRENT_RUN + 1))
    done
done

echo ""
echo "============================================================"
echo "Laptop-Scale Campaign Complete!"
echo "All 12 runs have been hashed and written to results/experiment_db.sqlite"
echo ""
echo "You may now generate the final IEEE manuscript by running:"
echo "  ./cropxai paper build"
echo "============================================================"
