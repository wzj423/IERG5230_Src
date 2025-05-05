#!/bin/bash
#SBATCH --job-name=dropout_test
#SBATCH --partition=S1_llmit2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

MASTER_PORT=30123
RDV_ADDR=$(hostname)
WORLD_SIZE=$SLURM_JOB_NUM_NODES

export http_proxy=http://opst:2C8nt8fVEN@10.1.8.50:33128
export https_proxy=http://opst:2C8nt8fVEN@10.1.8.50:33128
export no_proxy=localhost,127.0.0.1,.sensetime.com,.pjlab.org.cn,



mkdir -p logs
mkdir -p traces
# # For single node multi-gpu ddp, deepspeed launcher can be used. 
# srun deepspeed --num_gpus=8 train_DS_multinode.py --drop_weight \
#     --output_dir gpt2_output \
#     --run_name gpt2_normal \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --num_train_epochs 1 \
#     --save_steps 1000 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --logging_steps 5 \
#     --learning_rate 6e-4 \
#     --gradient_accumulation_steps 1 \
#     --save_safetensors True \
#     --fp16 True \
#     --fp16_full_eval True \
#     --deepspeed ds_config_single_GPU_no_offload.json \
#     &> $LOG_FILE

DS_CONFIG_FILE=$1
if [ "$DS_CONFIG_FILE" == "" ]; then
    DS_CONFIG_FILE=ds_configs/ds_config_z1.json
fi
RUN_NAME=$2
if [ "$RUN_NAME" == "" ]; then
    RUN_NAME=llama_vanilla_z1b6
fi
LOG_FILE=logs/$RUN_NAME.log
TRACE_NAME=traces/$RUN_NAME
# As slurm cannot support inter-node ssh, we are not using deepspeed launcher here.
srun torchrun --nproc_per_node=1 \
   --nnodes=$WORLD_SIZE \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
    test.py --drop_weight\
    --output_dir llama_output \
    --run_name llama_normal \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 5 \
    --learning_rate 4e-4 \
    --gradient_accumulation_steps 1 \
    --save_safetensors True \
    --fp16 True \
    --fp16_full_eval True \
    --deepspeed $DS_CONFIG_FILE \
    --tensorboard_trace_handler $TRACE_NAME \
    &> $LOG_FILE