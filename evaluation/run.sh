#!/bin/bash

# Output directory for all results
DATA_DIR="../data"
OUTPUT_DIR="../results/extracted_quality_scores"
MODEL_PATH="../pretrained"

# Backbone types to evaluate
BACKBONES=("token" "crfiqa" "full")

# Early exit blocks to evaluate (0-11 + full model at 12)
EARLY_EXIT_BLOCKS=(12 11 10 9 8 7 6 5 4 3 2 1 0)

echo "Starting comprehensive evaluation of all backbones and early exit blocks..."

# Loop through all backbone types
for backbone in "${BACKBONES[@]}"
do
    echo "Processing backbone: $backbone"

    # Loop through all early exit blocks
    for exit_block in "${EARLY_EXIT_BLOCKS[@]}"
    do
        echo "Running with $backbone backbone, early exit block $exit_block"

        python3 getQualityScore.py \
            --data-dir "${DATA_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --model_path "${MODEL_PATH}" \
            --backbone "${backbone}" \
            --early-exit-block "${exit_block}" \

        echo "Completed $backbone backbone with early exit block $exit_block"
        echo "---------------------------------------------"
    done
done

echo "All evaluations completed! Results saved to ${OUTPUT_DIR}"
