export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Instruments_copytwo
DATA_PATH=./data
OUTPUT_DIR=./ckpt_Our/$DATASET/
RESULTS_FILE=./results/$DATASET/our2_1.json
BASE_MODEL="LLM path"
CKPT_PATH=./ckpt_Our/$DATASET/

torchrun --nproc_per_node=4 --master_port=4324 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 8 \
    --test_prompt_ids 0 \
    --index_file .item_sid_output.json
