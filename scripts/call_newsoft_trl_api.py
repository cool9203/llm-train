# coding: utf-8

import argparse
import ast
import json
import os
import re
from pathlib import Path

import requests
from tqdm import tqdm


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NewSoft invoice result")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input data path")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output result filename")
    parser.add_argument("--api_url", type=str, required=True, help="Api url")
    parser.add_argument(
        "--prompt",
        type=str,
        default="請從下列發票圖像中擷取以下欄位資訊，並以 JSON 格式回傳：InvoiceNumber、CompanyName、BuUniformNumber、InvoiceDate、NetAmount、TaxAmount、TotalAmount、TotalAmountCH、VnUniformNumber。",
        help="Api parameter: prompt",
    )
    parser.add_argument("--max_tokens", type=int, default=512, help="Api parameter: max token")
    parser.add_argument("--batch_size", type=int, default=5, help="Run batch inference size")

    args = parser.parse_args()

    return args


def batch_generator(
    data: list,
    batch_size: int,
):
    """將 list 拆分為 batch 的 generator"""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def call_newsoft_trl_api(
    api_url: str,
    input_path: os.PathLike,
    output_path: os.PathLike,
    prompt: str,
    max_tokens: int = 512,
    batch_size: int = 5,
):
    output_path = Path(output_path) if not str(output_path).endswith(".json") else Path(str(output_path)[:-5])
    # 處理圖片前, 先取得過往已處理的資料
    existing_results = list()
    if output_path.exists():
        try:
            with output_path.open(mode="r", encoding="utf-8") as json_file:
                existing_results = json.load(json_file)
        except Exception:
            existing_results = list()

    # 取得已處理的檔名
    processed_files = set(item["filename"] for item in existing_results)
    all_result = existing_results  # 使用已存在的结果作为起点

    # 獲取所有圖片文件
    image_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))

    # Run
    unprocessed_images = [img for img in image_files if os.path.basename(img) not in processed_files]
    with tqdm(total=len(unprocessed_images), desc="處理圖片中") as progress_bar:
        for image_paths in batch_generator(unprocessed_images, batch_size=batch_size):
            images = [open(image_path, "rb") for image_path in image_paths]
            files = [("images", image) for image in images]
            data = {
                "model_name": "lora",
                "img_type": "png",
                "system_prompt": None,
                "max_tokens": max_tokens,
                "prompt": prompt,
            }
            responses = requests.post(api_url, files=files, data=data)
            responses = responses.json()
            for i in range(len(responses)):
                if "images" in responses[i]:
                    del responses[i]["images"]
                origin_content = str(responses[i]["origin_content"]).lower()
                origin_content = re.sub(r'"品項"\s*:\s*\[.*?\],?', "", origin_content, flags=re.DOTALL)
                try:
                    origin_content = ast.literal_eval(origin_content.replace("\n", ""))
                except Exception as e:
                    print(e)

                all_result.append(
                    {
                        "filename": Path(image_paths[i]).name,
                        "api_result": origin_content,
                        "used_time": responses[i]["used_time"],
                    }
                )
                images[i].close()

            progress_bar.update(len(image_paths))
            with output_path.open(mode="w", encoding="utf-8") as json_file:
                json.dump(all_result, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = arg_parser()
    call_newsoft_trl_api(**vars(args))
