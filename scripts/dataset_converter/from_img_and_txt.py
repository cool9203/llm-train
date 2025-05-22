# coding: utf-8

import argparse
import json
import os
import pprint
import re
import sys
from pathlib import Path

import pypandoc
import tqdm as TQDM


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert label studio format to OpenAI format")

    parser.add_argument(
        "-i",
        "--input_paths",
        nargs="+",
        type=str,
        required=True,
        help="Input label studio annotation path",
    )
    parser.add_argument("--image_path", type=str, default=None, help="Image path")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Output path")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("-sp", "--system_prompt", type=str, default=None, help="System prompt")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["latex", "html", "markdown"],
        default=None,
        help="Check format",
    )
    parser.add_argument("--reasoning", action="store_true", help="Add table reasoning content")
    parser.add_argument(
        "--code_block",
        action="store_true",
        help="Add code block tag to contain table content",
    )
    parser.add_argument(
        "--row_comment",
        type=int,
        default=0,
        help="Add row comment to contain table content, for hint row index",
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


def replace_nth_occurrence(
    text: str,
    sub: str | re.Pattern[str],
    repl: str,
    n: int,
    pos: int = 0,
    endpos: int = sys.maxsize,
) -> str:
    pattern = re.compile(re.escape(sub) if isinstance(sub, str) else sub)
    matches = list(pattern.finditer(string=text, pos=pos, endpos=endpos))
    if len(matches) < n:
        return text  # 沒有第 n 個出現
    start = matches[n - 1].start()
    end = start + len(pattern.pattern)
    return text[:start] + repl + text[end:]


def from_img_and_txt(
    input_paths: os.PathLike,
    prompt: str,
    output_path: os.PathLike = None,
    system_prompt: str = "",
    image_path: os.PathLike = None,
    output_format: str = None,
    reasoning: bool = False,
    code_block: bool = False,
    row_comment: int = 0,
    tqdm: bool = True,
) -> list[dict[str, list[str | dict[str, str]]]]:
    from llm_train import utils

    labels: list[Path] = list()
    for input_path in input_paths:
        for extension in [".txt", ".html"]:
            labels += list(Path(input_path).glob(f"*{extension}"))

    converted_data = list()
    for label_file in TQDM.tqdm(labels) if tqdm else labels:
        _image_path = None

        # Get image path
        for image_extension in [".png", ".jpg", ".jpeg"]:
            if Path(label_file.parent, label_file.stem + image_extension.lower()).exists():
                _image_path = Path(label_file.parent, label_file.stem + image_extension.lower())
            elif Path(label_file.parent, label_file.stem + image_extension.upper()).exists():
                _image_path = Path(label_file.parent, label_file.stem + image_extension.upper())

        assert _image_path is not None, f"未找到標記檔案對應之圖像: '{label_file!s}'"

        with label_file.open(mode="r", encoding="utf-8") as f:
            label_content = f.read()

        label_content = text_replace(
            text=label_content,
            patterns=[
                (r"\\#", "#"),
            ],
        )

        try:
            if output_format:
                texts = list()
                reasoning_contents = list()

                # Convert to same format
                if utils.is_html_table(table_str=label_content) or utils.is_latex_table(table_str=label_content):
                    text = label_content
                elif utils.is_markdown_table(table_str=label_content):
                    text = pypandoc.convert_text(source=label_content, format="markdown", to="html")

                dfs = utils.convert_table_to_pandas(table_str=text, headers=True, unsqueeze=False)
                for df_index, df in enumerate(dfs):
                    df.columns = [re.sub(r"\.\d+$", "", str(col)) for col in df.columns]

                    # Filter df rows
                    # for value in ["合計", "總計"]:
                    #     df = df[~df.isin([value]).any(axis=1)]

                    replaced_columns = text_replace(
                        text=str(list(df.columns)),
                        patterns=[
                            (r"Unnamed: ?\d+", ""),
                        ],
                    )

                    reasoning_contents.append(
                        f"確認表格{df_index + 1}結構，"
                        + f"看起來有{len(df.columns)}欄與{len(df) + 1 if output_format == 'latex' else len(df)}列。"
                        + (
                            f"其中欄位名稱分別為{replaced_columns}。"
                            if not df.columns.str.fullmatch(r"\d+").all()
                            else "看起來沒有欄位名稱"
                        )
                    )

                    if output_format == "latex":
                        text = utils.convert_pandas_to_latex(df=df)
                        if row_comment > 0:
                            latex_table_row_pattern = re.compile(r"[\s\S]*? ?\\\\")
                            latex_table_header_result = list(re.finditer(utils._latex_table_begin_pattern, text))
                            for row_index, sub in enumerate(
                                latex_table_row_pattern.finditer(
                                    string=text,
                                    pos=latex_table_header_result[0].end() if latex_table_header_result else 0,
                                )
                            ):
                                if (row_index + 1) % row_comment == 0:
                                    text = replace_nth_occurrence(
                                        text=text,
                                        sub=latex_table_row_pattern,
                                        repl=f"%第{row_index + 1}列開始\n{sub.group()}",
                                        n=row_index + 1,
                                        pos=latex_table_header_result[0].end() if latex_table_header_result else 0,
                                    )
                    elif output_format == "markdown":
                        text = df.to_markdown(index=False, numalign=None, stralign=None)
                        text = text_replace(
                            text=text,
                            patterns=[
                                (r" \| |\| | \|", r"|"),
                                (r"\|-{2,}", r"|-"),
                            ],
                        )
                        if row_comment > 0:
                            markdown_table_row_pattern = re.compile(r"\|[\s\S]*?\|(?:\n|$)")
                            markdown_table_header_result = list(re.finditer(r"\|(?:-\|)+", text))
                            for row_index, sub in enumerate(
                                markdown_table_row_pattern.finditer(
                                    string=text,
                                    pos=markdown_table_header_result[0].end() if markdown_table_header_result else 0,
                                )
                            ):
                                if (row_index + 1) % row_comment == 0:
                                    text = replace_nth_occurrence(
                                        text=text,
                                        sub=markdown_table_row_pattern,
                                        repl=f"<!-- 第{row_index + 1}列開始 -->{sub.group()}",
                                        n=row_index + 1,
                                        pos=(markdown_table_header_result[0].end() if markdown_table_header_result else 0),
                                    )
                    elif output_format == "html":
                        text = re.sub(
                            r"<tr.*>",
                            "<tr>",
                            df.to_html(
                                index=False,
                                header=not df.columns.str.fullmatch(r"\d+").all(),  # Ignore number's herder, if all is number
                            )
                            .replace(' border="1"', "")
                            .replace(' class="dataframe"', ""),
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
                        if row_comment > 0:
                            html_table_row_pattern = re.compile(r"<tr>")
                            html_table_header_result = list(re.finditer(r"<\/thead>", text))
                            for row_index, sub in enumerate(
                                html_table_row_pattern.finditer(
                                    string=text,
                                    pos=html_table_header_result[0].end() if html_table_header_result else 0,
                                )
                            ):
                                if (row_index + 1) % row_comment == 0:
                                    text = replace_nth_occurrence(
                                        text=text,
                                        sub=html_table_row_pattern,
                                        repl=f"<!-- 第{row_index + 1}列開始 -->{sub.group()}",
                                        n=row_index + 1,
                                        pos=html_table_header_result[0].end() if html_table_header_result else 0,
                                    )
                    text = text_replace(
                        text=text,
                        patterns=[
                            (r"Unnamed: ?\d+", ""),
                        ],
                    )

                    if code_block:
                        text = f"```{output_format}\n{text}```"
                    texts.append(text)
                reasoning_content = (
                    "<think>\n"
                    + f"首先仔細檢查圖片裡的表格數量，看起來總共有{len(dfs)}個表格。\n"
                    + "\n".join(reasoning_contents)
                    + f"\n根據user的要求，需要使用{output_format.lower()}語法解析表格。"
                    + "\n</think>"
                )
                if reasoning:
                    texts = [reasoning_content] + texts  # Insert reasoning_content into texts index-0

        except Exception as e:
            print(f"file: {label_file!s}")
            print(label_content)
            raise e

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
        messages.append(
            {
                "role": "assistant",
                "content": "\n\n".join(texts) if output_format else label_content,
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
    print(pprint.pformat(vars(args)))
    from_img_and_txt(**vars(args))
