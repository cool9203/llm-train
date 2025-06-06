# coding: utf-8

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

try:
    import cn2an

    support_cn2an = True
except ImportError:
    support_cn2an = False

_cn2an_replace_vocab = OrderedDict(
    [
        # Fix word error
        ("參", "叁"),
        (r"弍|兩", "貳"),
        (r"o|×|0", "零"),
        ("元", ""),
        # Fix cn2an error
        (r"仟零佰", "仟零"),
        (r"零拾(?=[壹貳叁肆伍陸柒捌玖])", "零"),
        (r"零$|零[拾佰仟萬]", ""),
        (r"仟(?=[壹貳叁肆伍陸柒捌玖]拾)", "仟零"),
    ]
)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NewSoft invoice correct")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input data path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Output result filename")
    parser.add_argument("--answer_path", type=str, required=True, help="Answer data path")

    args = parser.parse_args()

    return args


def v4_compare_with_ans(
    answer_path: os.PathLike,
    input_path: os.PathLike,
    output_path: os.PathLike = None,
):
    if Path(answer_path).suffix == ".xlsx":
        df = pd.read_excel(answer_path)
    elif hasattr(pd, f"read_{Path(answer_path).suffix[1:]}"):
        df = getattr(pd, f"read_{Path(answer_path).suffix[1:]}")(
            answer_path,
            encoding="utf-8",
            dtype=str,
        )
    else:
        raise ValueError(f"Not support read '{Path(answer_path).suffix}'")

    df.columns = df.columns.str.lower()
    with Path(input_path).open(mode="r", encoding="utf-8") as f:
        api_results = json.load(f)

    correct_counter = dict(
        InvoiceNumber=0,
        CompanyName=0,
        VnUniformNumber=0,
        BuUniformNumber=0,
        InvoiceDate=0,
        NetAmount=0,
        TaxAmount=0,
        TotalAmount=0,
        TotalAmountCH=0,
    )
    correct_counter = {k.lower(): dict(name=k, count=v) for k, v in correct_counter.items()}

    print(f"是否支援中文數字轉換: {support_cn2an}")

    compare_results = list()
    for api_result in api_results:
        compare_result = dict()
        filename = api_result["filename"]
        item_ocr_result = {k.lower(): v for k, v in api_result["api_result"].items()}
        matched_row = df[df["filename"] == filename]
        for match_row_key in matched_row:
            match_row_key = match_row_key.lower()
            if match_row_key == "filename":
                compare_result[match_row_key] = filename
                continue

            compare_result[f"{match_row_key}_ans"] = str(matched_row[match_row_key].iloc[0])
            compare_result[f"{match_row_key}_iii"] = str(item_ocr_result.get(match_row_key, f"沒有這個key值: {match_row_key}"))
            compare_result[f"{match_row_key}_result"] = 0

            if match_row_key in item_ocr_result:
                gold_text = str(item_ocr_result[match_row_key]).lower()
                predict_text = str(matched_row[match_row_key].iloc[0]).lower()
                gold_text_float = None
                predict_text_float = None

                if re.fullmatch(r"[\d,]+\.\d+|[\d,]+", gold_text) and re.fullmatch(r"[\d,]+\.\d+|[\d,]+", predict_text):
                    gold_text_float = float(gold_text.replace(",", ""))
                    predict_text_float = float(predict_text.replace(",", ""))

                if gold_text == predict_text or (
                    gold_text_float is not None and predict_text_float is not None and gold_text_float == predict_text_float
                ):
                    compare_result[f"{match_row_key}_result"] = 1
                else:
                    if match_row_key == "InvoiceDate".lower():
                        if gold_text == f"中華民國{predict_text}":
                            compare_result[f"{match_row_key}_result"] = 1
                    if match_row_key == "TotalAmountCH".lower() and support_cn2an:
                        try:
                            for find_pattern, target_pattern in _cn2an_replace_vocab.items():
                                gold_text = re.sub(find_pattern, target_pattern, gold_text)

                            for find_pattern, target_pattern in _cn2an_replace_vocab.items():
                                predict_text = re.sub(find_pattern, target_pattern, predict_text)

                            if cn2an.cn2an(gold_text, "smart") == cn2an.cn2an(predict_text, "smart"):
                                compare_result[f"{match_row_key}_result"] = 1
                        except (ValueError, KeyError):
                            pass
            else:
                print(match_row_key, item_ocr_result)

            if compare_result[f"{match_row_key}_result"] == 1:
                correct_counter[match_row_key]["count"] += 1

        compare_results.append(compare_result)

    if output_path:
        with Path(Path(output_path).stem + ".json").open(mode="w", encoding="utf-8") as json_file:
            json.dump(compare_results, json_file, ensure_ascii=False, indent=4)
        print({v["name"]: v["count"] for _, v in correct_counter.items()})

        # 將結果轉換為 DataFrame
        results_df = pd.DataFrame(compare_results)
        results_df.to_csv(str(Path(Path(output_path).stem + ".csv")), index=False, encoding="utf-8")
        results_df.to_excel(Path(Path(output_path).stem + ".xlsx"), index=False, engine="openpyxl")


if __name__ == "__main__":
    args = arg_parser()
    v4_compare_with_ans(**vars(args))
