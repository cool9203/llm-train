# coding: utf-8

import math
import re
from inspect import signature
from io import StringIO
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import pandas as pd
from PIL import Image

_latex_table_begin_pattern = r"\\begin{tabular}{[lrc|]*}"
_latex_table_end_pattern = r"\\end{tabular}"
_latex_table_pattern = r"\\begin{tabular}{[lrc|]*}[\s\S]*?\\end{tabular}"
_latex_multicolumn_pattern = r"\\multicolumn{(\d+)}{([lrc|]+)}{(.*)}"
_latex_multirow_pattern = r"\\multirow{(\d+)}{([\*\d]*)}{(.*)}"
_html_table_begin_pattern = r"<table\b[^>]*?>"
_html_table_end_pattern = r"</table>"
_html_table_pattern = r"(?:^|\n) *<table\b[^>]*?>[\s\S]*?<\/table>"
_markdown_table_row_pattern = r"(?m)^(\|.+\|)"


class FormatError(Exception): ...


class LatexTableGeneratorError(Exception): ...


class NotDetectTableError(LatexTableGeneratorError): ...


class NotColumnMatchError(LatexTableGeneratorError): ...


class ImagePasteError(LatexTableGeneratorError): ...


class NotHtmlError(LatexTableGeneratorError, FormatError): ...


class NotLatexError(LatexTableGeneratorError, FormatError): ...


class NotSupportLatexError(LatexTableGeneratorError, FormatError): ...


class NotSupportMultiLatexTableError(NotSupportLatexError): ...


def _get_function_used_params(
    callable: Callable,
    **kwds: Dict,
) -> Dict[str, Any]:
    """Get `callable` need parameters from kwds.

    Args:
        callable (Callable): function

    Returns:
        Dict[str, Any]: parameters
    """
    parameters = dict()
    callable_parameters = signature(callable).parameters
    for parameter, value in kwds.items():
        if parameter in callable_parameters:
            if value:
                parameters.update({parameter: value})
            else:
                parameters.update({parameter: None})
    return parameters


def _remove_all_space_row(
    rows: list[list[Any]],
) -> list[list[str]]:
    cleaned_rows = list()
    for row in rows:
        _row_data = list()
        for cell in row:
            _row_data.append(str(cell).strip())
        if len("".join(_row_data)) == 0:
            continue
        else:
            cleaned_rows.append(_row_data)
    return cleaned_rows


def preprocess_latex_table_string(
    latex_table_str: str,
) -> str:
    processed_latex_table_str = re.sub(_latex_table_begin_pattern, "", latex_table_str)
    processed_latex_table_str = re.sub(_latex_table_end_pattern, "", processed_latex_table_str)
    processed_latex_table_str = processed_latex_table_str.replace("\n", " ").strip()

    # Fix multiple \hline and \hline not at start of row error
    rows = processed_latex_table_str.split(r"\\")
    new_rows = list()
    for row in rows:
        _row = row
        if row.count(r"\hline") > 0:
            _row = _row.replace(r"\hline", "").strip()
            _row = rf"\hline {_row}"
        new_rows.append(_row)

    return "\\\\\n".join(new_rows)


def is_latex_table(table_str: str) -> bool:
    return len(re.findall(_latex_table_begin_pattern, table_str)) >= 1


def is_html_table(table_str: str) -> bool:
    return len(re.findall(_html_table_begin_pattern, table_str)) >= 1


def is_markdown_table(table_str: str) -> bool:
    return len(re.findall(_markdown_table_row_pattern, table_str)) >= 1


def pre_check_latex_table_string(
    latex_table_str: str,
) -> Tuple[str, str]:
    results = re.findall(_latex_table_begin_pattern, latex_table_str)
    if not results:
        raise NotLatexError("Not latex table")

    begin_str = results[0]
    end_str = r"\end{tabular}"
    return (begin_str, end_str)


def convert_latex_table_to_pandas(
    latex_table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
    unsqueeze: bool = False,
    remove_all_space_row: bool = False,
    **kwds,
) -> pd.DataFrame:
    pre_check_latex_table_string(latex_table_str=latex_table_str)
    processed_latex_table_str = preprocess_latex_table_string(latex_table_str)
    rows = [
        row.replace("\n", "").strip()
        for row in processed_latex_table_str.split(r"\\")
        if ("&" in row or r"\multicolumn" in row) and row.replace("\n", "").strip()
    ]  # Filter unrelated row data

    # Split latex table to list table
    cleaned_data = list()
    table_data = [row.replace(r"\\", "").replace(r"\hline", "").replace(r"\cline", "").strip().split("&") for row in rows]
    for row in table_data:
        _row_data = list()
        for cell_text in row:
            _cell_text = cell_text.strip()
            if re.match(_latex_multicolumn_pattern, _cell_text):
                multicolumn_data = re.findall(_latex_multicolumn_pattern, _cell_text)[0]
                for index in range(int(multicolumn_data[0])):
                    if unsqueeze:
                        _row_data.append(multicolumn_data[2].strip())
                    else:
                        if index == 0:
                            _row_data.append(
                                rf"\multicolumn{{{multicolumn_data[0]}}}{{{multicolumn_data[1]}}}{{{multicolumn_data[2].strip()}}}"
                            )
                        else:
                            _row_data.append("")
            else:
                _row_data.append(_cell_text)
        cleaned_data.append(_row_data)

    # Process multirow
    for col in range(len(cleaned_data)):
        for row in range(len(cleaned_data[col])):
            # Clean multi row data
            multirow_result = re.findall(_latex_multirow_pattern, cleaned_data[col][row])
            if multirow_result:
                if unsqueeze:
                    for offset in range(int(multirow_result[0][0])):
                        cleaned_data[col + offset][row] = multirow_result[0][2].strip()
                else:
                    cleaned_data[col][row] = (
                        rf"\multirow{{{multirow_result[0][0]}}}{{{multirow_result[0][1]}}}{{{multirow_result[0][2].strip()}}}"
                    )
                    for offset in range(1, int(multirow_result[0][0])):
                        if col + offset >= len(cleaned_data):
                            break
                        cleaned_data[col + offset][row] = ""

    if remove_all_space_row:
        cleaned_data = _remove_all_space_row(rows=cleaned_data)

    try:
        if headers:
            if isinstance(headers, bool):
                headers = cleaned_data[0]  # First row is header
                cleaned_data = cleaned_data[1:]  # Other row is row data

            # Filling every row length to headers length
            for i in range(len(cleaned_data)):
                if len(cleaned_data[i]) > len(headers):
                    cleaned_data[i] = cleaned_data[i][: len(headers)]
                elif len(cleaned_data[i]) < len(headers):
                    cleaned_data[i] += ["" for _ in range(len(headers) - len(cleaned_data[i]))]
            df = pd.DataFrame(cleaned_data, columns=headers)
        else:
            df = pd.DataFrame(cleaned_data)
    except ValueError as e:
        raise NotSupportLatexError("Not support this latex") from e

    return df


def convert_pandas_to_latex(
    df: pd.DataFrame,
    full_border: bool = False,
) -> str:
    _row_before_text = ""
    if full_border:
        _row_before_text = r"\hline "
        latex_table_str = f"\\begin{{tabular}}{{{'c'.join(['|' for _ in range(len(df.columns) + 1)])}}}\n"
    else:
        latex_table_str = f"\\begin{{tabular}}{{{''.join(['c' for _ in range(len(df.columns))])}}}\n"

    # Add header
    latex_table_str += _row_before_text + f"{'&'.join(list(str(column) for column in df.columns))}\\\\\n"  # noqa: C400

    # Add row data
    for i in range(len(df)):
        row = list()
        skip_count = 0
        for column_index in range(len(df.columns)):
            if skip_count > 0:
                skip_count -= 1
            else:
                multicolumn_result = re.findall(_latex_multicolumn_pattern, str(df.iloc[i, column_index]))
                skip_count = int(multicolumn_result[0][0]) - 1 if multicolumn_result and skip_count == 0 else skip_count
                row.append(str(df.iloc[i, column_index]))
        latex_table_str += _row_before_text + f"{'&'.join(row)}\\\\\n"

    if full_border:
        latex_table_str += "\\hline\n"
    latex_table_str += r"\end{tabular}"

    return latex_table_str


def convert_html_table_to_pandas(
    html_table_str: str,
    remove_all_space_row: bool = False,
    **kwds,
) -> list[pd.DataFrame]:
    try:
        with StringIO(html_table_str) as f:
            dfs = pd.read_html(
                io=f,
                keep_default_na=False,
            )

        if remove_all_space_row:
            return [
                pd.DataFrame(
                    _remove_all_space_row(rows=df.values.tolist()),
                    columns=df.columns,
                )
                for df in dfs
            ]
        else:
            return dfs
    except Exception:
        raise NotHtmlError("This table str not is html")


def convert_table_to_pandas(
    table_str: str,
    headers: Union[bool, Sequence[str], None] = None,
    unsqueeze: bool = False,
    remove_all_space_row: bool = False,
    **kwds,
) -> list[pd.DataFrame]:
    if is_latex_table(table_str):
        return [
            convert_latex_table_to_pandas(
                latex_table_str=latex_table_str,
                headers=headers,
                unsqueeze=unsqueeze,
                remove_all_space_row=remove_all_space_row,
                **kwds,
            )
            for latex_table_str in re.findall(_latex_table_pattern, table_str)
        ]
    elif is_html_table(table_str):
        return convert_html_table_to_pandas(
            html_table_str=table_str,
            unsqueeze=unsqueeze,
            remove_all_space_row=remove_all_space_row,
            **kwds,
        )
    else:
        raise FormatError("Not Support convert the format table")


def preprocess_image(
    image: Image.Image,
    image_max_pixels: int,
    image_min_pixels: int,
    **kwds,
) -> Image.Image:
    r"""Pre-process a single image.
    Copy from https://github.com/hiyouga/LLaMA-Factory/blob/845af89ea4a8ee4003d72de1cbddbe85910c37df/src/llamafactory/data/mm_plugin.py#L210-L227
    """
    if image_max_pixels and (image.width * image.height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image_min_pixels and (image.width * image.height) < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image
