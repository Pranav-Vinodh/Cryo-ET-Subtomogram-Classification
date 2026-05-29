#!/bin/bash

# Lambda sweep experiment: 3-shot training across multiple lambda values and seeds
# Tests lambda_residual values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Across 10 random seeds

# Configuration
N_SHOT=5
CUDA_DEVICE_JOINT=5
CUDA_DEVICE_BASELINE=6
LAMBDA_MMD=0.2

# Lambda values to test
LAMBDA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)

# 10 random seeds
SEEDS=(0 42 100 123 456 789 1024 2048 4096 8192)

echo "=========================================="
echo "Starting Lambda Sweep Experiment"
echo "N-shot: ${N_SHOT}"
echo "Lambda values: ${LAMBDA_VALUES[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Joint runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "Baseline runs: ${#SEEDS[@]} (lambda not applicable)"
echo "Total runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]} + ${#SEEDS[@]}))"
echo "=========================================="

# Counters for statistics
JOINT_SUCCESS=0
JOINT_FAIL=0
BASELINE_SUCCESS=0
BASELINE_FAIL=0
START_TIME=$(date +%s)

# Run joint training for all lambda values and seeds
echo ""
echo "=========================================="
echo "PHASE 1: JOINT TRAINING (Source + Target)"
echo "=========================================="

for lambda_res in "${LAMBDA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Running JOINT training: lambda_residual=${lambda_res}, seed=${seed}"
        python train_joint_nshot_swin3d.py \
            --n_shot ${N_SHOT} \
            --lambda_residual ${lambda_res} \
            --seed ${seed} \
            --cuda_device ${CUDA_DEVICE_JOINT} \
            --lambda_mmd ${LAMBDA_MMD}

        if [ $? -eq 0 ]; then
            echo "✓ Completed JOINT: lambda_residual=${lambda_res}, seed=${seed}"
            ((JOINT_SUCCESS++))
        else
            echo "✗ FAILED JOINT: lambda_residual=${lambda_res}, seed=${seed}"
            ((JOINT_FAIL++))
        fi
    done
done

# Run baseline training (lambda_residual is not applicable for baseline)
echo ""
echo "=========================================="
echo "PHASE 2: BASELINE TRAINING (Target Only)"
echo "Note: lambda_residual not used in baseline"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running BASELINE training: seed=${seed}"
    python train_baseline_nshot_swin3d.py \
        --n_shot ${N_SHOT} \
        --seed ${seed} \
        --cuda_device ${CUDA_DEVICE_BASELINE} \
        --lambda_mmd ${LAMBDA_MMD}

    if [ $? -eq 0 ]; then
        echo "✓ Completed BASELINE: seed=${seed}"
        ((BASELINE_SUCCESS++))
    else
        echo "✗ FAILED BASELINE: seed=${seed}"
        ((BASELINE_FAIL++))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE - STATISTICS"
echo "=========================================="
echo "Configuration:"
echo "  N-shot: ${N_SHOT}"
echo "  Lambda values: ${LAMBDA_VALUES[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo "  CUDA devices: Joint=${CUDA_DEVICE_JOINT}, Baseline=${CUDA_DEVICE_BASELINE}"
echo ""
echo "Joint Training (Source + Target):"
echo "  ✓ Successful: ${JOINT_SUCCESS} / $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "  ✗ Failed: ${JOINT_FAIL}"
echo ""
echo "Baseline Training (Target Only):"
echo "  ✓ Successful: ${BASELINE_SUCCESS} / ${#SEEDS[@]}"
echo "  ✗ Failed: ${BASELINE_FAIL}"
echo ""
echo "Overall:"
echo "  Total runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]} + ${#SEEDS[@]}))"
echo "  Total successful: $((JOINT_SUCCESS + BASELINE_SUCCESS))"
echo "  Total failed: $((JOINT_FAIL + BASELINE_FAIL))"
echo "  Success rate: $(awk "BEGIN {printf \"%.1f\", 100.0 * ($JOINT_SUCCESS + $BASELINE_SUCCESS) / (${#LAMBDA_VALUES[@]} * ${#SEEDS[@]} + ${#SEEDS[@]})}")%"
echo ""
echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
