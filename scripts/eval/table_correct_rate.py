# coding: utf-8

import argparse
import logging
import os
import platform
import pprint
import re
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import tqdm as TQDM

from llm_train import utils

_replace_vocab = {
    "内": "內",
    "×": "x",
    "·": ",",
    "，": ",",
    r"\#": "#",
    "□": " ",
    "（": "(",
    "）": ")",
    "撑": "撐",
}
_css = r"""<style>
    details {
        border: 1px solid #aaa;
        border-radius: 4px;
        padding: 0.5em;
    }

    table,
    th,
    td {
        border: 1px solid black;
        border-collapse: collapse;
        font-size: 24px;
        font-weight: normal;
    }

    table,
    img,
    details {
        max-width: 45%;
    }

    img,
    table,
    details {
        float: left;
    }

    img,
    details>table {
        max-width: 100%;
    }
</style>"""


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation latex table model")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        default=[],
        help="Evaluation dataset path",
    )
    parser.add_argument(
        "--detect_headers",
        type=str,
        nargs="+",
        default=[],
        help="Auto detect table header names",
    )
    parser.add_argument(
        "--ignore_headers",
        type=str,
        nargs="+",
        default=[],
        help="Ignore table header names, support regex",
    )
    parser.add_argument(
        "--ignore_values",
        type=str,
        nargs="+",
        default=[],
        help="Ignore table row contain values, support regex",
    )
    parser.add_argument("--output", type=str, default="eval_result", help="Eval detail output path")
    parser.add_argument(
        "--skip_space_row",
        dest="remove_all_space_row",
        action="store_false",
        help="Eval skip space row",
    )
    parser.add_argument(
        "--inference_result_folder",
        type=str,
        default=None,
        help="Save inference result folder name",
    )
    parser.add_argument("--target_platform", type=str, default=None, help="Platform")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


@dataclass
class EvalResult:
    predict_latex_error: bool = False
    gold_latex_error: bool = False
    cell_correct_count: int = 0
    cell_count: int = 0
    txt_filepath: str = None
    image_filepath: str = None
    gold_df: pd.DataFrame = None
    predict_df: pd.DataFrame = None
    error_indexes: list[tuple[int, int]] = field(default_factory=list)
    dataset_path: str = ""


def calc_correct_rate(
    results: list[EvalResult],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if len(results) == 0:
        return tuple([[0.0, 0.0, 0.0] for _ in range(4)])
    table_correct = sum(
        [
            1
            if not result.predict_latex_error and not result.gold_latex_error and result.cell_correct_count == result.cell_count
            else 0
            for result in results
        ]
    )
    cell_correct = sum([result.cell_correct_count for result in results])
    format_incorrect = sum([1 if result.predict_latex_error else 0 for result in results])
    label_format_incorrect = sum([1 if result.gold_latex_error else 0 for result in results])
    return (
        [
            table_correct,
            len(results),
            f"{table_correct / len(results):.3}",
        ],  # Table correct rate
        [
            cell_correct,
            sum([result.cell_count for result in results]),
            f"{cell_correct / sum([result.cell_count for result in results])}",
        ],  # Cell correct rate
        [
            format_incorrect,
            len(results),
            f"{format_incorrect / len(results):.3}",
        ],  # Format incorrect rate
        [
            label_format_incorrect,
            len(results),
            f"{label_format_incorrect / len(results):.3}",
        ],  # Label format incorrect rate
    )


def convert_filepath(
    filepath: PathLike,
    path_type: str | tuple[str, str],
) -> str:
    path_type = path_type.split(":") if isinstance(path_type, str) else path_type
    if path_type[0] in ["wsl"]:
        _filepath = str(filepath)[4:] if str(filepath).startswith("/mnt") else filepath
    else:
        _filepath = str(filepath)
    if path_type[0] in ["windows"] and path_type[1] in ["linux", "wsl"]:
        _filepath = PureWindowsPath(
            _filepath.replace("C:", "/c").replace("D:", "/d").replace("c:", "/c").replace("d:", "/d")
        ).as_posix()
    elif path_type[0] in ["linux", "wsl"] and path_type[1] in ["windows"]:
        if _filepath.startswith("/c"):
            _filepath = "\\".join(PurePosixPath(_filepath.replace("/c", "C:", 1)).parts)
        elif _filepath.startswith("/d"):
            _filepath = "\\".join(PurePosixPath(_filepath.replace("/d", "D:", 1)).parts)
        else:
            _filepath = "\\".join(PurePosixPath(_filepath).parts)
    if path_type[1] in ["wsl"]:
        if _filepath.startswith("/c"):
            _filepath = str(PurePosixPath(_filepath.replace("/c", "/mnt/c", 1)))
        elif _filepath.startswith("/d"):
            _filepath = str(PurePosixPath(_filepath.replace("/d", "/mnt/d", 1)))
        else:
            _filepath = str(PurePosixPath(_filepath))
    return str(_filepath)


def highlight_multiple(
    data: pd.DataFrame,
    targets: list[tuple[int, int]],
    style: str | tuple[str, str],
):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for col, row in targets:
        styles.iloc[row, col] = style if isinstance(style, str) else ":".join(style)
    return styles


def save_result(
    results: list[EvalResult],
    output_path: PathLike,
    style: str | tuple[str, str],
    path_type: str | tuple[str, str],
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if style:
        style: tuple[str, str] = style if isinstance(style, tuple) else tuple(style.split(":"))

    correct_rate = calc_correct_rate(results=results)

    with Path(output_path, "eval_detail.txt").open("w", encoding="utf-8") as f:
        f.write(
            ("-" * 25 + "\n").join(
                [
                    pprint.pformat(result).replace("gold_df=  ", "gold_df=\n").replace("predict_df=  ", "predict_df=\n")
                    for result in results
                ]
            )
            + "-" * 25
            + "\n\n"
        )
        f.writelines(
            [
                f"Table correct rate: {correct_rate[0][0]} / {correct_rate[0][1]} = {correct_rate[0][2]}\n",
                f"Cell correct rate: {correct_rate[1][0]} / {correct_rate[1][1]} = {correct_rate[1][2]}\n",
                f"Format incorrect rate: {correct_rate[2][0]} / {correct_rate[2][1]} = {correct_rate[2][2]}\n",
                f"Label format incorrect rate: {correct_rate[3][0]} / {correct_rate[3][1]} = {correct_rate[3][2]}\n",
            ]
        )

    # Save html
    for result in results:
        if not Path(output_path, Path(result.dataset_path).name).exists():
            Path(output_path, Path(result.dataset_path).name).mkdir()

        if result.cell_correct_count != result.cell_count:
            with Path(
                output_path,
                Path(result.dataset_path).name,
                f"{Path(result.txt_filepath).stem}.html",
            ).open("w", encoding="utf-8") as f:
                image_filepath = convert_filepath(filepath=result.image_filepath, path_type=path_type)

                error_indexes_styling = [
                    (index[0], index[1] + 1 if index[1] is not None else 0) for index in result.error_indexes
                ]
                if result.predict_df is not None:
                    df = pd.DataFrame([list(result.predict_df.columns)] + result.predict_df.values.tolist())
                    if style:
                        styling_df = df.style.apply(
                            highlight_multiple,
                            axis=None,
                            targets=error_indexes_styling,
                            style=style,
                        )
                    else:
                        styling_df = df
                    styling_df_html = styling_df.to_html(header=False, index=False)
                    f.write(f"""{_css}
Cell count: {result.cell_count}
<br>
Cell correct count: {result.cell_correct_count}
<br>
<details open="true">
    <summary>原始圖片</summary>
    <img src="{image_filepath}">
</details>
<details>
    <summary>標記表格</summary>
    {result.gold_df.to_html()}
</details>
{styling_df_html}""")


def detect_pandas_header(
    df: pd.DataFrame,
    detect_headers: list[str],
) -> pd.DataFrame:
    detected_df = None
    for row_index in range(len(df)):
        for detect_header in detect_headers:
            if detect_header in df.iloc[row_index].tolist():
                detected_df = df.rename(
                    columns=dict(list(zip(df.columns, df.iloc[row_index].tolist()))),
                )
                detected_df = detected_df[detected_df.index > row_index]
                break
        if detected_df is not None:
            break
    return detected_df if detected_df is not None else df


def get_pandas_ignore_header(
    df: pd.DataFrame,
    ignore_headers: list[str],
) -> np.ndarray:
    mask = np.array([False for _ in range(len(df.columns))], dtype=bool)
    for ignore_header in ignore_headers:
        mask = mask | df.columns.str.match(ignore_header)
    return mask


def table_correct_rate(
    dataset_path: PathLike,
    inference_result_folder: str,
    remove_all_space_row: bool = True,
    detect_headers: list[str] = [],
    ignore_headers: list[str] = [],
    ignore_values: list[str] = [],
    tqdm: bool = True,
) -> list[EvalResult]:
    data: list[tuple[PathLike, PathLike]] = list()
    results: list[EvalResult] = list()
    iter_data: list[Path] = list()

    # Pre-check dataset are correct pairs for label txt data and image data

    for extension in [".txt", ".html"]:
        iter_data += list(Path(dataset_path).glob(f"*{extension}"))

    for txt_filepath in iter_data:
        _data = None
        for image_extension in [".jpg", ".png"]:
            if Path(txt_filepath.parent, f"{txt_filepath.stem}{image_extension.lower()}").exists():
                _data = [
                    txt_filepath,
                    Path(
                        txt_filepath.parent,
                        f"{txt_filepath.stem}{image_extension.lower()}",
                    ),
                ]
            elif Path(txt_filepath.parent, f"{txt_filepath.stem}{image_extension.upper()}").exists():
                _data = [
                    txt_filepath,
                    Path(
                        txt_filepath.parent,
                        f"{txt_filepath.stem}{image_extension.upper()}",
                    ),
                ]

        if _data:
            inference_data_path = None
            for extension in [".txt", ".html"]:
                if Path(
                    dataset_path,
                    inference_result_folder,
                    f"{txt_filepath.stem}{extension}",
                ).exists():
                    inference_data_path = Path(
                        dataset_path,
                        inference_result_folder,
                        f"{txt_filepath.stem}{extension}",
                    )
                    break
            if inference_data_path:
                _data.append(inference_data_path)
                data.append(tuple(_data))
            else:
                raise ValueError(
                    f"Not have inference data: {Path(dataset_path, inference_result_folder, f'{txt_filepath.stem}.txt')!s}"
                )
        else:
            raise ValueError(f"Not have image data: {txt_filepath!s}")
    logger.debug(f"data: {data}")

    # Eval
    for txt_filepath, image_filepath, inference_filepath in TQDM.tqdm(data, desc="Eval") if tqdm else data:
        with Path(inference_filepath).open("r", encoding="utf-8") as f:
            predict_latex_table_text = f.read()

        with Path(txt_filepath).open(mode="r", encoding="utf-8") as f:
            gold_latex_table_text = f.read()

        (gold_df, predict_df) = (None, None)
        try:
            # Pre-process replace vocab
            for find_word, replace_word in _replace_vocab.items():
                gold_latex_table_text = gold_latex_table_text.replace(find_word, replace_word)

            for find_word, replace_word in _replace_vocab.items():
                predict_latex_table_text = predict_latex_table_text.replace(find_word, replace_word)

            # Convert to pandas
            gold_df = utils.convert_table_to_pandas(
                table_str=gold_latex_table_text,
                headers=True,
                unsqueeze=True,
                remove_all_space_row=remove_all_space_row,
            )[0]
            predict_df = utils.convert_table_to_pandas(
                table_str=predict_latex_table_text,
                headers=True,
                unsqueeze=True,
                remove_all_space_row=remove_all_space_row,
            )[0]
        except Exception as e:
            logger.error(txt_filepath)
            logger.exception(e)

        logger.debug(gold_df)
        logger.debug("-" * 25)
        logger.debug(predict_df)
        logger.debug("-" * 25)

        result = EvalResult(
            txt_filepath=str(txt_filepath),
            image_filepath=str(image_filepath),
            gold_df=gold_df,
            predict_df=predict_df,
            dataset_path=str(dataset_path),
        )

        if gold_df is not None:
            gold_df_non_ignore_header_count = sum(~get_pandas_ignore_header(df=gold_df, ignore_headers=ignore_headers))
            result.cell_count = gold_df_non_ignore_header_count * len(gold_df) + gold_df_non_ignore_header_count
            for ignore_value in ignore_values:
                gold_df = gold_df[~gold_df.apply(lambda row: row.astype(str).str.contains(ignore_value, regex=True)).any(axis=1)]

        if predict_df is not None:
            for ignore_value in ignore_values:
                predict_df = predict_df[
                    ~predict_df.apply(lambda row: row.astype(str).str.contains(ignore_value, regex=True)).any(axis=1)
                ]

        if predict_df is not None and gold_df is not None:
            # Detect header from rows
            if detect_headers:
                gold_df = detect_pandas_header(df=gold_df, detect_headers=detect_headers)
                predict_df = detect_pandas_header(df=predict_df, detect_headers=detect_headers)
                result.gold_df = gold_df
                result.predict_df = predict_df
                gold_df_non_ignore_header_count = sum(~get_pandas_ignore_header(df=gold_df, ignore_headers=ignore_headers))
                result.cell_count = gold_df_non_ignore_header_count * len(gold_df) + gold_df_non_ignore_header_count

        if predict_df is None:
            result.predict_latex_error = True
        elif gold_df is None:
            result.gold_latex_error = True
        else:
            # Compare header(column)
            for column_index in range(min(len(gold_df.columns), len(predict_df.columns))):
                if gold_df.columns[column_index] in ignore_headers or predict_df.columns[column_index] in ignore_headers:
                    continue

                gold_text = str(gold_df.columns[column_index]).replace(" ", "").replace("\n", "").replace(":", "")
                predict_text = str(predict_df.columns[column_index]).replace(" ", "").replace("\n", "").replace(":", "")

                # Post fix multi header error
                if re.match(r"\(.*\)", predict_text):
                    _result = re.findall(r"""'([^'"]+)'""", predict_text)
                    predict_text = _result[0] if _result else predict_text

                if gold_text == predict_text:
                    result.cell_correct_count += 1
                else:
                    result.error_indexes.append((column_index, None))

            # Compare row
            for column_index in range(min(len(gold_df.columns), len(predict_df.columns))):
                if gold_df.columns[column_index] in ignore_headers or predict_df.columns[column_index] in ignore_headers:
                    continue

                for row_index in range(min(len(gold_df), len(predict_df))):
                    gold_text = gold_df.iloc[row_index, column_index].strip().replace("\n", "").replace(":", "")
                    predict_text = predict_df.iloc[row_index, column_index].strip().replace("\n", "").replace(":", "")
                    gold_text_split = gold_text.split(" ")
                    predict_text_split = predict_text.split(" ")
                    if (len(gold_text_split) == len(predict_text_split) and set(gold_text_split) == set(predict_text_split)) or (
                        gold_text.replace(" ", "") == predict_text.replace(" ", "")
                    ):
                        result.cell_correct_count += 1
                    else:
                        result.error_indexes.append((column_index, row_index))

        results.append(result)

        if logger.level <= 10:
            break

    return results


if __name__ == "__main__":
    args = arg_parser()
    parameters = vars(args)

    logger.info(f"Used parameters:\n{pprint.pformat(parameters)}")

    dataset_paths: dict[str, str] = parameters.pop("datasets")
    output_path: str = parameters.pop("output")
    target_platform: str = parameters.pop("target_platform")

    results = []
    for dataset_path in dataset_paths:
        logger.info(f"Run {dataset_path!s}")
        dataset_result = table_correct_rate(
            dataset_path=dataset_path,
            **parameters,
        )
        correct_rate = calc_correct_rate(results=dataset_result)
        logger.info(f"Table correct rate: {correct_rate[0][0]} / {correct_rate[0][1]} = {correct_rate[0][2]}")
        logger.info(f"Cell correct rate: {correct_rate[1][0]} / {correct_rate[1][1]} = {correct_rate[1][2]}")
        logger.info(f"Format incorrect rate: {correct_rate[2][0]} / {correct_rate[2][1]} = {correct_rate[2][2]}")
        logger.info(f"Label format incorrect rate: {correct_rate[3][0]} / {correct_rate[3][1]} = {correct_rate[3][2]}")
        logger.info("-" * 25)
        results += dataset_result

    correct_rate = calc_correct_rate(results=results)
    logger.info(f"Table correct rate: {correct_rate[0][0]} / {correct_rate[0][1]} = {correct_rate[0][2]}")
    logger.info(f"Cell correct rate: {correct_rate[1][0]} / {correct_rate[1][1]} = {correct_rate[1][2]}")
    logger.info(f"Format incorrect rate: {correct_rate[2][0]} / {correct_rate[2][1]} = {correct_rate[2][2]}")
    logger.info(f"Label format incorrect rate: {correct_rate[3][0]} / {correct_rate[3][1]} = {correct_rate[3][2]}")
    logger.info("-" * 25)

    source_platform = platform.platform().lower()

    if "wsl" in source_platform:
        source_platform = "wsl"
    elif "linux" in source_platform:
        source_platform = "linux"
    else:
        source_platform = "windows"

    if target_platform is None:
        target_platform = source_platform

    path_type = f"{source_platform}:{target_platform}"
    logger.info(f"path_type: {path_type}")
    save_result(
        results=results,
        output_path=output_path,
        style=("border", "7px solid red"),
        path_type=path_type,
    )
