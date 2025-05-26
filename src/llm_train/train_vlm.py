# coding: utf-8

# This script copy from https://github.com/huggingface/trl/blob/fd04a5461a158a09818c93ac3c3e3ca8488ccdbb/examples/scripts/sft_vlm.py

"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from peft import get_peft_model
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from llm_train import utils


@dataclass
class ExtendArguments:
    image_max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Image max pixels"},
    )
    image_min_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Image min pixels"},
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, ExtendArguments))
    script_args, training_args, model_args, extend_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    peft_config = get_peft_config(model_args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset_names = str(script_args.dataset_name).split(",")
    data = list()
    show_messages_count = 3
    if np.array(
        [
            True if Path(dataset_name).is_file() and Path(dataset_name).suffix == ".json" else False
            for dataset_name in dataset_names
        ]
    ).all():
        for dataset_name in dataset_names:
            with Path(dataset_name).open(mode="r", encoding="utf-8") as f:
                for payload in tqdm.tqdm(list(json.load(fp=f))):
                    images = list()
                    messages = list()
                    for image_path in payload["images"]:
                        image = Image.open(image_path)
                        precessed_image = utils.preprocess_image(
                            image=image,
                            **vars(extend_args),
                        )
                        img_byte_arr = io.BytesIO()
                        precessed_image.save(img_byte_arr, format=image.format)
                        image = Image.open(
                            img_byte_arr
                        )  # this is very important it forces parquet to save bytes instead of the path
                        images.append(image)

                    for message in payload["messages"]:
                        if isinstance(message["content"], str):
                            contents = list()
                            for i, text in enumerate(re.split(r"<image>", message["content"])):
                                if i > 0:
                                    contents.append({"text": None, "type": "image"})
                                if text:
                                    contents.append({"text": text, "type": "text"})

                            messages.append({"role": message["role"], "content": contents})
                        elif isinstance(message["content"], list):
                            messages.append(message)
                        else:
                            raise TypeError(f"{dataset_name} messages.content not support type: {type(message['content'])}")

                    if show_messages_count:
                        print(messages)
                        show_messages_count -= 1

                    data.append(
                        {
                            "messages": messages,
                            "images": images,
                        }
                    )
        dataset = DatasetDict({"train": Dataset.from_list(data)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
