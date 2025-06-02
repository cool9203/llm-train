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
    parser.add_argument("-o", "--output", type=str, required=True, help="Output result filename")
    parser.add_argument("--api_url", type=str, required=True, help="Api url")
    parser.add_argument(
        "--prompt",
        type=str,
        default="請從下列發票圖像中擷取以下欄位資訊，並以 JSON 格式回傳：InvoiceNumber、CompanyName、BuUniformNumber、InvoiceDate、NetAmount、TaxAmount、TotalAmount、TotalAmountCH、VnUniformNumber。",
        help="Api parameter: prompt",
    )
    parser.add_argument("--max_token", type=int, default=512, help="Api parameter: max token")

    args = parser.parse_args()

    return args


def call_newsoft_trl_api(
    api_url: str,
    input_path: os.PathLike,
    output_path: os.PathLike,
    prompt: str,
    max_tokens: int = 512,
):
    # 處理圖片前, 先取得過往已處理的資料
    existing_results = list()
    if Path(output_path).exists():
        try:
            with open(output_path, "r", encoding="utf-8") as json_file:
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
    for image_path in tqdm(unprocessed_images, desc="處理圖片中"):
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {
                "model_name": "lora",
                "img_type": "png",
                "system_prompt": None,
                "max_tokens": max_tokens,
                "prompt": prompt,
            }
            response = requests.post(api_url, files=files, data=data)
            response = response.json()
            if "images" in response:
                del response["images"]
            print(response)
            origin_content = str(response["origin_content"]).lower()
            origin_content = re.sub(r'"品項"\s*:\s*\[.*?\],?', "", origin_content, flags=re.DOTALL)
            try:
                origin_content = ast.literal_eval(origin_content)
            except Exception as e:
                print(e)

            all_result.append(
                {
                    "filename": os.path.basename(image_path),
                    "api_result": origin_content,
                    "used_time": response["used_time"],
                }
            )
            with Path(Path(output_path).stem + ".json").open(mode="w", encoding="utf-8") as json_file:
                json.dump(all_result, json_file, ensure_ascii=False, indent=4)
