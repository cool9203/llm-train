accelerate launch src/llm_train/train_vlm.py \
	`# Dataset` \
	--dataset_name data/tmpco-table-reasoning-code_block-row_comment-20250502-html.json \
	--max_length 2048 \
	`# Model` \
	--model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
	--attn_implementation flash_attention_2 \
	--image_max_pixels 1631220 \
	`# Lora` \
	--use_peft \
	--lora_dropout 0.0 \
	--lora_target_modules ".*proj" \
	`# Output` \
	--output_dir saves/qwen2.5_vl-3b/tmpco/table/html/lora/sft/20250512 \
	--logging_steps 10 \
	--save_strategy epoch \
	`# Train` \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 4 \
	--bf16 \
	--torch_dtype bfloat16 \
	--gradient_checkpointing \
	--learning_rate 1.0e-4 \
	--num_train_epochs 10.0 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.1 \
    --ddp_timeout 180000000
