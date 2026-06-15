#!/bin/bash

# Lambda sweep experiment: 3-shot training across multiple lambda values and seeds
# Tests lambda_residual values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Across 10 random seeds

# Locate python interpreter
PYTHON=${PYTHON:-"/shared/scratch/0/home/v_pranav_vinodh/miniconda3/envs/torch/bin/python"}
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

# Configuration defaults (supports environment variables)
DATASET=${DATASET:-"noble"}
N_SHOT=${N_SHOT:-5}
CUDA_DEVICE_JOINT=${CUDA_DEVICE_JOINT:-5}
CUDA_DEVICE_BASELINE=${CUDA_DEVICE_BASELINE:-6}
LAMBDA_MMD=${LAMBDA_MMD:-0.2}
LOSS_TYPE=${LOSS_TYPE:-"mmd"} # Can be set to "mmd" or "coral"

# Optional run toggles (default to false, set to true to execute)
RUN_RESNET=${RUN_RESNET:-"false"}
RUN_DA_BASELINE=${RUN_DA_BASELINE:-"false"}
RUN_SWIN_BASELINE=${RUN_SWIN_BASELINE:-"true"}

# Default lambda values to test
LAMBDA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)

# Default 10 random seeds
SEEDS=(0 42 100 123 456 789 1024 2048 4096 8192)

EXTRA_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --n_shot)
      N_SHOT="$2"
      shift 2
      ;;
    --loss_type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --gpu)
      export CUDA_VISIBLE_DEVICES="$2"
      CUDA_DEVICE_JOINT=0
      CUDA_DEVICE_BASELINE=0
      shift 2
      ;;
    --run_resnet)
      RUN_RESNET="true"
      shift
      ;;
    --run_da)
      RUN_DA_BASELINE="true"
      shift
      ;;
    --skip_swin_baseline)
      RUN_SWIN_BASELINE="false"
      shift
      ;;
    --lambdas)
      IFS=' ' read -r -a LAMBDA_VALUES <<< "$2"
      shift 2
      ;;
    --seeds)
      IFS=' ' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# If CUDA_VISIBLE_DEVICES is set, ensure default devices are 0
if [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_DEVICE_JOINT=${CUDA_DEVICE_JOINT:-0}
    CUDA_DEVICE_BASELINE=${CUDA_DEVICE_BASELINE:-0}
fi

echo "=========================================="
echo "Starting Lambda Sweep Experiment"
echo "Dataset: ${DATASET}"
echo "N-shot: ${N_SHOT}"
echo "Lambda values: ${LAMBDA_VALUES[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Loss Type: ${LOSS_TYPE}"
echo "Run ResNet Baseline: ${RUN_RESNET}"
echo "Run DA Baseline: ${RUN_DA_BASELINE}"
echo "Joint runs: $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "Swin3D Baseline runs: ${#SEEDS[@]}"
echo "=========================================="

# Counters for statistics
JOINT_SUCCESS=0
JOINT_FAIL=0
BASELINE_SUCCESS=0
BASELINE_FAIL=0
RESNET_SUCCESS=0
RESNET_FAIL=0
DA_SUCCESS=0
DA_FAIL=0
START_TIME=$(date +%s)

# Run joint training for all lambda values and seeds
echo ""
echo "=========================================="
echo "PHASE 1: JOINT TRAINING (Source + Target)"
echo "=========================================="

for lambda_res in "${LAMBDA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Running JOINT training: dataset=${DATASET}, lambda_residual=${lambda_res}, seed=${seed}, loss_type=${LOSS_TYPE}"
        $PYTHON train_joint_nshot_swin3d.py \
            --dataset ${DATASET} \
            --n_shot ${N_SHOT} \
            --lambda_residual ${lambda_res} \
            --seed ${seed} \
            --cuda_device ${CUDA_DEVICE_JOINT} \
            --lambda_mmd ${LAMBDA_MMD} \
            --loss_type ${LOSS_TYPE} \
            "${EXTRA_ARGS[@]}"

        if [ $? -eq 0 ]; then
            echo "✓ Completed JOINT: lambda_residual=${lambda_res}, seed=${seed}"
            ((JOINT_SUCCESS++))
        else
            echo "✗ FAILED JOINT: lambda_residual=${lambda_res}, seed=${seed}"
            ((JOINT_FAIL++))
        fi
    done
done

# Run baseline training
echo ""
echo "=========================================="
echo "PHASE 2: BASELINE TRAINING (Target Only)"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    if [ "${RUN_SWIN_BASELINE}" = "true" ]; then
        echo ""
        echo ">>> Running SWIN3D BASELINE training: dataset=${DATASET}, seed=${seed}"
        # Standard Swin3D Baseline:
        $PYTHON train_baseline_nshot_swin3d.py \
            --dataset ${DATASET} \
            --n_shot ${N_SHOT} \
            --seed ${seed} \
            --cuda_device ${CUDA_DEVICE_BASELINE}

        if [ $? -eq 0 ]; then
            echo "✓ Completed SWIN3D BASELINE: seed=${seed}"
            ((BASELINE_SUCCESS++))
        else
            echo "✗ FAILED SWIN3D BASELINE: seed=${seed}"
            ((BASELINE_FAIL++))
        fi
    fi

    # Run ResNet-34 baseline if toggled
    if [ "${RUN_RESNET}" = "true" ]; then
        echo ""
        echo ">>> Running RESNET-34 BASELINE training: dataset=${DATASET}, seed=${seed}"
        $PYTHON train_baseline_nshot_resnet34.py \
            --dataset ${DATASET} \
            --n_shot ${N_SHOT} \
            --seed ${seed} \
            --cuda_device ${CUDA_DEVICE_BASELINE}
            
        if [ $? -eq 0 ]; then
            echo "✓ Completed RESNET-34 BASELINE: seed=${seed}"
            ((RESNET_SUCCESS++))
        else
            echo "✗ FAILED RESNET-34 BASELINE: seed=${seed}"
            ((RESNET_FAIL++))
        fi
    fi

    # Run DA baseline (feature-only alignment) if toggled
    if [ "${RUN_DA_BASELINE}" = "true" ]; then
        echo ""
        echo ">>> Running DA BASELINE training: dataset=${DATASET}, seed=${seed}, loss_type=${LOSS_TYPE}"
        $PYTHON train_da_baseline_nshot_swin3d.py \
            --dataset ${DATASET} \
            --n_shot ${N_SHOT} \
            --seed ${seed} \
            --cuda_device ${CUDA_DEVICE_BASELINE} \
            --loss_type ${LOSS_TYPE} \
            --lambda_align ${LAMBDA_MMD}

        if [ $? -eq 0 ]; then
            echo "✓ Completed DA BASELINE: seed=${seed}"
            ((DA_SUCCESS++))
        else
            echo "✗ FAILED DA BASELINE: seed=${seed}"
            ((DA_FAIL++))
        fi
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
echo "  Dataset: ${DATASET}"
echo "  N-shot: ${N_SHOT}"
echo "  Lambda values: ${LAMBDA_VALUES[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo "  CUDA devices: Joint=${CUDA_DEVICE_JOINT}, Baseline=${CUDA_DEVICE_BASELINE}"
echo ""
echo "Joint Training (Source + Target):"
echo "  ✓ Successful: ${JOINT_SUCCESS} / $((${#LAMBDA_VALUES[@]} * ${#SEEDS[@]}))"
echo "  ✗ Failed: ${JOINT_FAIL}"
echo ""
echo "Swin3D Baseline Training (Target Only):"
echo "  ✓ Successful: ${BASELINE_SUCCESS} / ${#SEEDS[@]}"
echo "  ✗ Failed: ${BASELINE_FAIL}"
echo ""
if [ "${RUN_RESNET}" = "true" ]; then
    echo "ResNet-34 Baseline Training (Target Only):"
    echo "  ✓ Successful: ${RESNET_SUCCESS} / ${#SEEDS[@]}"
    echo "  ✗ Failed: ${RESNET_FAIL}"
    echo ""
fi
if [ "${RUN_DA_BASELINE}" = "true" ]; then
    echo "DA Baseline Training (Feature-only Alignment):"
    echo "  ✓ Successful: ${DA_SUCCESS} / ${#SEEDS[@]}"
    echo "  ✗ Failed: ${DA_FAIL}"
    echo ""
fi
echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
