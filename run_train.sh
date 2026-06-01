DATASET=Instruments_copytwo
BASE_MODEL="LLM path"
DATA_PATH=./data
OUTPUT_DIR=./ckpt_Our/$DATASET/

torchrun --nproc_per_node=4 --master_port=3325  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .item_sid_output.json\
    --wandb_run_name test_Our\
    --temperature 1.0
