#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of A100 Nodes with 4 A100 GPUs (single GPU 128 SBUs/hour, 512 SBUs/hour for an entire node)
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=9                       # Specify the number of CPUs/task (18/GPU, 9/GPU_mig)
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=...                         # Specify the node partition
#SBATCH --time=120:00:00                        # Specify the maximum time the job can run
#SBATCH --mail-type=BEGIN,END                   # Specify when to receive notifications on email
#SBATCH --mail-user=...                         # Specify email address to receive notifications

# STEP 1: Export folders for further use
export HOME_FOLDER=...
export BASE_FOLDER=...
export EXP_FOLDER=experiments

# STEP 2: Navigate to the location of the code
cd $BASE_FOLDER || return

# STEP 3: CREATE .SIF FILE
apptainer pull docker://chjkusters4/wle-cade-nbi-cadx:V2

# STEP 3: Setup WANDB logging
export WANDB_API_KEY=...
export WANDB_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_CONFIG_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_CACHE_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

# STEP 4: Define Loss functions, Mask Content, Seeds
MASK_CONTENT=('Hard')
LOSS_FUNCTION=('BCE')
SEEDS=(0)


# STEP 5: Execute experiments by for-loops
for i in "${!MASK_CONTENT[@]}"
do
    for j in "${!LOSS_FUNCTION[@]}"
    do
        for k in "${!SEEDS[@]}"
        do
            export OUTPUT_FOLDER=${MASK_CONTENT[$i]}_${LOSS_FUNCTION[$j]}_${SEEDS[$k]}
            if [ -f "${HOME_FOLDER}"/$EXP_FOLDER/"${OUTPUT_FOLDER}" ]; then
                echo "Folder for ${OUTPUT_FOLDER} already exists"
                echo "============Skipping============"
            else
                echo "Folder for ${OUTPUT_FOLDER} does not exist"
                echo "============Starting============"
                srun apptainer exec --nv $BASE_FOLDER/wle-cade-nbi-cadx_V2.sif \
                python3 train.py --experimentname "${OUTPUT_FOLDER}" \
                                 --seed "${SEEDS[$k]}" \
                                 --backbone ResNet-50-UNet \
                                 --optimizer Adam \
                                 --scheduler Plateau \
                                 --cls_criterion BCE \
                                 --cls_criterion_weight 1.0 \
                                 --seg_criterion "${LOSS_FUNCTION[$j]}" \
                                 --seg_metric Dice \
                                 --label_smoothing 0.01 \
                                 --focal_alpha_cls -1.0 \
                                 --focal_gamma_cls 1.0 \
                                 --batchsize 32 \
                                 --mask_content "${MASK_CONTENT[$i]}" \
                                 --num_epochs 150 \
                                 --train_lr 1e-4

                cp -r $BASE_FOLDER/$EXP_FOLDER/"${OUTPUT_FOLDER}" "${HOME_FOLDER}"/$EXP_FOLDER
                echo "============Finished============"
            fi
        done
    done
done
