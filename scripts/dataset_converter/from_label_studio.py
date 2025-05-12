# coding: utf-8

import argparse
import ast
import json
import os
from pathlib import Path

import tqdm as TQDM


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert label studio format to OpenAI format")

    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input label studio annotation path")
    parser.add_argument("--image_path", type=str, default=None, help="Image path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Output path")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("-sp", "--system_prompt", type=str, default=None, help="System prompt")
    parser.add_argument("-c", "--check_format", type=str, choices=["json"], default=None, help="Check format")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()
    return args


def _check_format(
    text: str,
    format: str,
) -> str:
    if not format:
        return True

    try:
        if format == "json":
            text = json.dumps(ast.literal_eval(text), ensure_ascii=False)
    except Exception as e:
        raise e
    return text


def from_label_studio(
    input_path: os.PathLike,
    prompt: str,
    output_path: os.PathLike = None,
    system_prompt: str = "",
    image_path: os.PathLike = None,
    check_format: str = None,
    tqdm: bool = True,
) -> list[dict[str, list[str | dict[str, str]]]]:
    with Path(input_path).open(mode="r", encoding="utf-8") as f:
        labels = json.load(fp=f)

    converted_data = list()
    for label in TQDM.tqdm(labels) if tqdm else labels:
        if label["cancelled_annotations"] > 0:
            continue
        messages = list()
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": f"<image> {prompt}",
            }
        )

        for annotation in label["annotations"][-1]["result"]:
            if annotation["type"] in ["textarea"]:
                try:
                    text = _check_format(text=annotation["value"]["text"][0], format=check_format)
                except Exception as e:
                    print(f"id: {label['id']}")
                    print(annotation["value"]["text"][0])
                    raise e

                messages.append({"role": "assistant", "content": text})
        converted_data.append(
            {
                "messages": messages,
                "images": [
                    str(
                        Path(
                            image_path,
                            Path(label["data"]["image"]).name,
                        )
                        if image_path
                        else Path(label["data"]["image"])
                    )
                ],
            }
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                converted_data,
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"✅ 已處理 {len(converted_data)} 筆資料，JSON 儲存到 {output_path}")
    else:
        print(f"✅ 已處理 {len(converted_data)} 筆資料")
    return converted_data


if __name__ == "__main__":
    args = arg_parser()
    from_label_studio(**vars(args))
