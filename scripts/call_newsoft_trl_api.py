# coding: utf-8

import ast
import json
import os
import re

import requests
from tqdm import tqdm

# folder_path = "/mnt/e/力新AIOCR/整張單據/100張的完整單據_VLM-4-Testing"  # 請替換為實際的資料夾路徑
folder_path = "/home/localadmin/data_center/100張的完整單據_VLM-4-Testing"
# folder_path = "/mnt/e/力新AIOCR/整張單據/20250416_測試803張"

# url = "http://10.70.0.128:30303/api/inference_table"
# url = "http://10.70.0.232:7861/api/inference_table"
url = "http://127.0.0.1:7866/api/inference_table"
# output_json = 'yoga_5005pic_6000steps_v4_api_test803pics_result.json'
# output_json = 'trl_test100pics_3b_0526_max_pixel_result.json'
output_json = "trl_test100pics_3b_0527_ck-4455_result.json"
max_tokens = 512
# prompt = '請從下列發票圖像中擷取以下欄位資訊，並以 JSON 格式回傳：InvoiceNumber、CompanyName、BuUniformNumber、InvoiceDate、NetAmount、TaxAmount、TotalAmount、VnUniformNumber'
prompt = "請從下列發票圖像中擷取以下欄位資訊，並以 JSON 格式回傳：InvoiceNumber、CompanyName、BuUniformNumber、InvoiceDate、NetAmount、TaxAmount、TotalAmount、TotalAmountCH、VnUniformNumber。"
mapping_name = {
    "發票號碼": "InvoiceNumber",
    "買受人名稱": "CompanyName",
    "賣方公司統編": "VnUniformNumber",
    "買方公司統編": "BuUniformNumber",
    "發票日期": "InvoiceDate",
    # "品項": "item",
    "未稅金額": "NetAmount",
    "稅額": "TaxAmount",
    "總金額": "TotalAmount",
    "手寫金額": "TotalAmountCH",
}

# 在处理图片前，先检查是否存在之前的结果
existing_results = []
if os.path.exists(output_json):
    try:
        with open(output_json, "r", encoding="utf-8") as json_file:
            existing_results = json.load(json_file)
    except Exception:
        existing_results = []

# 获取已处理的文件名列表
processed_files = set(item["filename"] for item in existing_results)
all_result = existing_results  # 使用已存在的结果作为起点

# 獲取所有圖片文件
image_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(os.path.join(root, file))

# 使用tqdm创建进度条，但只显示未处理的文件
unprocessed_images = [img for img in image_files if os.path.basename(img) not in processed_files]
for image_path in tqdm(unprocessed_images, desc="處理圖片中"):
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        # data = {
        #     'detect_table': False,
        #     'repair_latex': False,
        #     'full_border': False,
        #     'unsqueeze': False,
        #     'system_prompt':None,
        #     # 'default_prompt':
        #     # 'prompt': '擷取該發票的資訊並輸出成JSON格式'
        #     'prompt':prompt
        # }
        data = {"model_name": "lora", "img_type": "png", "system_prompt": None, "max_tokens": max_tokens, "prompt": prompt}
        response = requests.post(url, files=files, data=data)
        response = response.json()
        if "images" in response:
            del response["images"]
        print(response)
        origin_content = response["origin_content"]
        origin_content = re.sub(r'"品項"\s*:\s*\[.*?\],?', "", origin_content, flags=re.DOTALL)
        try:
            # origin_content = json.loads(origin_content)
            origin_content = ast.literal_eval(origin_content)
        except Exception as e:
            print(e)
            pass
        # 根据mapping_name替换键名
        mapped_content = {}
        for zh_key, en_key in mapping_name.items():
            # print(zh_key, en_key)
            if zh_key in origin_content:
                mapped_content[en_key] = origin_content[zh_key]
            elif en_key in origin_content:
                mapped_content[en_key] = origin_content[en_key]
        # print(mapped_content)
        all_result.append(
            {"filename": os.path.basename(image_path), "api_result": mapped_content, "used_time": response["used_time"]}
        )
        with open(output_json, "w", encoding="utf-8") as json_file:
            json.dump(all_result, json_file, ensure_ascii=False, indent=4)
