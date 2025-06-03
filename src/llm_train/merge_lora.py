# coding: utf-8

import argparse
from pathlib import Path

from peft import LoraModel, PeftModel
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    PreTrainedModel,
)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge lora to base model script")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Merge base model name or path")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Save path")
    parser.add_argument("-l", "--lora_model", type=str, required=True, help="Lora adapter name or path")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Device map, cuda or cpu")

    args = parser.parse_args()

    return args


def merge_lora(
    model_name: str,
    output_path: str,
    lora_model: str,
    device_map: str = "cuda:0",
):
    base_model: PreTrainedModel = AutoModelForVision2Seq.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
    )
    merged_model: LoraModel = PeftModel.from_pretrained(base_model, lora_model)
    tokenizer = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name)

    print("Start merge model")
    model = merged_model.merge_and_unload(progressbar=True)

    print("Save model")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Success merge lora model and saved")


if __name__ == "__main__":
    args = arg_parser()
    merge_lora(**vars(args))
