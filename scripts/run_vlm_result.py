# coding: utf-8

import argparse
import ast
import datetime as dt
import logging
import os
import pprint
import time
from os import PathLike
from pathlib import Path

import httpx
import tqdm as TQDM

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run VLM result script")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, default=[], help="Run dataset path")
    parser.add_argument("--api_url", type=str, default=None, help="Model gradio api url")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Model max tokens")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe the picture using HTML while considering merged columns and merged row cells.",
        help="Model prompt",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Model system prompt",
    )
    parser.add_argument("--inference_result_folder", type=str, default=None, help="Save inference result folder name")

    args = parser.parse_args()

    return args


def _vlm_inference(
    api_url: str,
    prompt: str,
    image_path: str,
    model_name: str,
    system_prompt: str,
    max_tokens: int,
    retry: int,
    timeout: float,
    **kwds,
) -> tuple[str, str]:
    _error = RuntimeError("inference error")
    _request_data = dict()
    _request_data.update(prompt=prompt) if prompt is not None else None
    _request_data.update(model_name=model_name) if model_name is not None else None
    _request_data.update(system_prompt=system_prompt) if system_prompt is not None else None
    _request_data.update(max_tokens=max_tokens) if max_tokens is not None else None

    for _ in range(retry):
        try:
            with httpx.Client(base_url=api_url, timeout=httpx.Timeout(timeout=timeout)) as client:
                resp = client.post(
                    url="/api/inference_table",
                    files={"image": Path(image_path).open("rb")},
                    data=_request_data,
                )
                response = resp.json()
            return (response["origin_content"], response["html_content"])
        except Exception as e:
            print(e)
            _error = e
            time.sleep(1)
    raise _error


def run_vlm_result(
    api_url: str,
    dataset_path: PathLike,
    inference_result_folder: str = None,
    prompt: str = None,
    model_name: str = None,
    system_prompt: str = None,
    max_tokens: int = 4096,
    retry: int = 5,
    timeout: float = 900,
    tqdm: bool = True,
):
    data: list[tuple[PathLike, PathLike]] = list()

    # Pre-check dataset are correct pairs for label txt data and image data
    for filepath in Path(dataset_path).iterdir():
        if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        if inference_result_folder:
            data.append(
                (
                    filepath,
                    Path(Path(dataset_path), inference_result_folder, f"{filepath.stem}"),
                )
            )
        else:
            data.append(
                (
                    filepath,
                    Path(Path(dataset_path), f"{filepath.stem}"),
                )
            )
    logger.debug(f"data: {data}")

    # Eval
    for image_filepath, inference_filepath in TQDM.tqdm(data, desc="Eval") if tqdm else data:
        if not Path(str(inference_filepath) + ".txt").exists() or not Path(str(inference_filepath) + ".html").exists():
            logger.info(f"Call api date: {dt.datetime.now()!s}")

            try:
                predict_latex_table_content, html_render_latex_table_content = _vlm_inference(
                    api_url=api_url,
                    prompt=prompt,
                    image_path=image_filepath,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    retry=retry,
                    timeout=timeout,
                )

                Path(inference_filepath).parent.mkdir(exist_ok=True, parents=True)
                with Path(str(inference_filepath) + ".txt").open("w", encoding="utf-8") as f:
                    f.write(predict_latex_table_content)
                with Path(str(inference_filepath) + ".html").open("w", encoding="utf-8") as f:
                    f.write(html_render_latex_table_content)
            except Exception:
                pass

            time.sleep(10)


if __name__ == "__main__":
    args = arg_parser()
    common_parameters = vars(args)

    logger.info(f"Used parameters:\n{pprint.pformat(common_parameters)}")

    dataset_paths: dict[str, str] = common_parameters.pop("datasets")

    for dataset_path in dataset_paths:
        dataset_path_split = dataset_path.split(":", maxsplit=1)
        passed_parameters = common_parameters.copy()
        if len(dataset_path_split) > 1:
            for parameter in dataset_path_split[1].replace(" ", "").split(","):
                parameter_split = parameter.split("=", maxsplit=1)
                passed_parameters.update(ast.literal_eval(f"{{'{parameter_split[0]}': {parameter_split[1]}}}"))

        logger.info(f"Eval {dataset_path_split[0]}\nparameter: {pprint.pformat(passed_parameters)}")
        run_vlm_result(
            dataset_path=dataset_path_split[0],
            timeout=900,
            retry=30,
            **passed_parameters,
        )
