# coding: utf-8

import argparse
import json
import os
import re
from pathlib import Path

import pypandoc
import tqdm as TQDM


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert label studio format to OpenAI format")

    parser.add_argument("-i", "--input_path", type=str, required=True, help="Input label studio annotation path")
    parser.add_argument("--image_path", type=str, default=None, help="Image path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Output path")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("-sp", "--system_prompt", type=str, default=None, help="System prompt")
    parser.add_argument(
        "--output_format", type=str, choices=["latex", "html", "markdown"], default=None, help="Check format"
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()
    return args


def text_replace(
    text: str,
    patterns: list[tuple[str, str]],
) -> str:
    result = text
    for find_pattern, target_pattern in patterns:
        result = re.sub(
            find_pattern,
            target_pattern,
            result,
        )
    return result


def from_img_and_txt(
    input_path: os.PathLike,
    prompt: str,
    output_path: os.PathLike = None,
    system_prompt: str = "",
    image_path: os.PathLike = None,
    output_format: str = None,
    tqdm: bool = True,
) -> list[dict[str, list[str | dict[str, str]]]]:
    from llamafactory import utils

    labels = list(Path(input_path).glob("*.txt"))

    converted_data = list()
    for label_file in TQDM.tqdm(labels) if tqdm else labels:
        messages = list()
        _image_path = None

        # Get image path
        for image_extension in [".png", ".jpg", ".jpeg"]:
            if Path(label_file.parent, label_file.stem + image_extension.lower()).exists():
                _image_path = Path(label_file.parent, label_file.stem + image_extension.lower())
            elif Path(label_file.parent, label_file.stem + image_extension.upper()).exists():
                _image_path = Path(label_file.parent, label_file.stem + image_extension.upper())

        assert _image_path is not None, f"未找到標記檔案對應之圖像: '{label_file!s}'"

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

        with label_file.open(mode="r", encoding="utf-8") as f:
            label_content = f.read()

        try:
            if output_format:
                texts = []
                # Convert to same format
                if utils.is_html_table(table_str=label_content) or utils.is_latex_table(table_str=label_content):
                    text = label_content
                elif utils.is_markdown_table(table_str=label_content):
                    text = pypandoc.convert_text(source=label_content, format="markdown", to="html")

                dfs = utils.convert_table_to_pandas(table_str=text, headers=True, unsqueeze=False)
                for df in dfs:
                    df.columns = [re.sub(r"\.\d+$", "", str(col)) for col in df.columns]

                    if output_format == "latex":
                        text = utils.convert_pandas_to_latex(df=df)
                    elif output_format == "markdown":
                        text = df.to_markdown(index=False, numalign=None, stralign=None)
                        text = text_replace(
                            text=text,
                            patterns=[
                                (r" \| |\| | \|", r"|"),
                                (r"\|-{2,}", r"|-"),
                            ],
                        )
                    elif output_format == "html":
                        text = re.sub(
                            r"<tr.*>",
                            "<tr>",
                            df.to_html(index=False).replace(' border="1"', "").replace(' class="dataframe"', ""),
                        )
                        text = text_replace(
                            text=text,
                            patterns=[
                                (r"\n *<th>", "<th>"),
                                (r"\n *<tr>", "<tr>"),
                                (r"\n *<td>", "<td>"),
                                (r"\n *<thead>", "<thead>"),
                                (r"\n *<tbody>", "<tbody>"),
                                (r"\n *</th>", "</th>"),
                                (r"\n *</tr>", "</tr>"),
                                (r"\n *</td>", "</td>"),
                                (r"\n *</thead>", "</thead>"),
                                (r"\n *</tbody>", "</tbody>"),
                                (r"\n *</table>", "</table>"),
                            ],
                        )
                    text = text_replace(
                        text=text,
                        patterns=[
                            (r"Unnamed: ?\d+", ""),
                        ],
                    )
                    texts.append(text)

        except Exception as e:
            print(f"file: {label_file!s}")
            print(label_content)
            raise e

        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(texts) if output_format else label_content,
            }
        )

        converted_data.append(
            {
                "messages": messages,
                "images": [
                    str(
                        Path(
                            image_path,
                            Path(_image_path).name,
                        )
                        if image_path is not None
                        else Path(_image_path)
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
    from_img_and_txt(**vars(args))
