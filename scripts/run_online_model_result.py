# coding: utf-8

import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
import logging
import os
import pprint
import time
from collections import OrderedDict
from pathlib import Path

import openai
import tqdm as TQDM
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from tenacity import Retrying, after_log, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_provider = dict(
    google="https://generativelanguage.googleapis.com/v1beta/openai",
    openai="https://api.openai.com/v1",
)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online vlm result")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, default=[], help="Run dataset path")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Online model api key, can set in ENV with XXX_API_KEY, ex: OPENAI_API_KEY",
    )
    parser.add_argument("--provider", type=str, required=True, help="Model provider")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Model prompt")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Model max tokens")
    parser.add_argument("--temperature", type=float, default=NOT_GIVEN, help="Model temperature")
    parser.add_argument("--top_p", type=float, default=NOT_GIVEN, help="Model top_p")
    parser.add_argument("--system_prompt", type=str, default="", help="Model system prompt")
    parser.add_argument("--inference_result_folder", type=str, default=None, help="Save inference result folder name")
    parser.add_argument(
        "--reasoning_effort", type=str, default=NOT_GIVEN, choices=["low", "medium", "high"], help="Reasoning budgets"
    )
    parser.add_argument("--icl_example_path", type=str, default=None, help="In context learning example path")
    parser.add_argument(
        "--icl_example_quantity", type=int, default=-1, help="In context learning example use quantity, set -1 are all use"
    )
    parser.add_argument("--icl_example_prompt", type=str, default=None, help="In context learning example prompt")
    parser.add_argument(
        "--unclassified_icl_example_category",
        action="store_true",
        help="Unclassified icl example category, will use all example to prompt",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun")
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def get_base64_image(image_path: os.PathLike) -> str:
    with Path(image_path).open(mode="rb") as image_file:
        extension = Path(image_path).suffix.lower()
        extension = ".jpeg" if extension in [".jpg"] else extension
        extension = extension[1:] if extension.startswith(".") else extension

        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/{extension};base64,{base64_image}"


def load_icl_example(
    icl_example_path: os.PathLike,
    icl_example_quantity: int,
    find_extensions: list = [".txt", ".json"],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    icl_example_path: Path = Path(icl_example_path)
    icl_example_iter_data = (
        list(
            icl_image_path
            for icl_image_path in icl_example_path.iterdir()
            if icl_image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
        if icl_example_path
        else []
    )
    icl_example_iter_data.sort()
    icl_example_data: list[dict[str, str]] = list()
    icl_example_data_info: list[dict[str, str]] = list()
    for icl_image_path in icl_example_iter_data:
        if icl_example_quantity >= 0 and len(icl_example_data) >= icl_example_quantity:
            break

        icl_text_path = None
        for extension in find_extensions:
            if Path(icl_example_path, f"{icl_image_path.stem}{extension}").exists():
                icl_text_path = Path(icl_example_path, f"{icl_image_path.stem}{extension}")
        if icl_text_path:
            logger.info(f"Usage example: {icl_image_path!s}")
            with icl_text_path.open(mode="r", encoding="utf-8") as f:
                icl_base64_image = get_base64_image(icl_image_path)
                icl_text = f.read()
                icl_example_data.append(
                    dict(
                        image_path=str(icl_image_path),
                        txt_path=str(icl_text_path),
                        base64_image=icl_base64_image,
                        text=icl_text,
                    )
                )
                icl_example_hash_md5 = hashlib.sha256(
                    str(
                        dict(
                            base64_image=icl_base64_image,
                            text=icl_text,
                        )
                    ).encode("utf-8")
                )
                icl_example_data_info.append(
                    dict(
                        image_path=str(icl_image_path),
                        txt_path=str(icl_text_path),
                        context_hash=icl_example_hash_md5.hexdigest(),
                    )
                )
    return (
        icl_example_data,
        icl_example_data_info,
    )


def run_online_model_result(
    dataset_path: str,
    api_key: str | None,
    model_name: str,
    provider: str,
    prompt: str,
    output_path: str = None,
    reasoning_effort: str = None,
    icl_example_path: str = None,
    icl_example_quantity: int = 1,
    icl_example_prompt: str = None,
    unclassified_icl_example_category: bool = False,
    inference_result_folder: str = None,
    system_prompt: str = "",
    max_tokens: int = 16384,
    temperature: float = NOT_GIVEN,
    top_p: float = NOT_GIVEN,
    retry: int = 3,
    tqdm: bool = True,
    force_rerun: bool = True,
    **kwds,
) -> None:
    dataset_path: Path = Path(dataset_path)
    icl_example_path: Path = Path(icl_example_path)
    icl_example_prompt = icl_example_prompt if icl_example_prompt else prompt

    if icl_example_path and not unclassified_icl_example_category and Path(icl_example_path, dataset_path.name).exists():
        icl_example_path = Path(icl_example_path, dataset_path.name)

    if Path(icl_example_path).exists():
        logger.info(f"icl_example_path: {icl_example_path}")
        logger.info(f"unclassified_icl_example_category: {unclassified_icl_example_category}")
    else:
        logger.info(f"Not found path will be ignore icl_example_path: '{icl_example_path}' ")
        icl_example_path = None

    # Get in context learning data
    if unclassified_icl_example_category:
        icl_example_data = list()
        icl_example_data_info = list()
        for icl_example_class_path in sorted(icl_example_path.iterdir()):
            icl_data = load_icl_example(
                icl_example_path=Path(icl_example_path, icl_example_class_path),
                icl_example_quantity=icl_example_quantity,
            )
            icl_example_data += icl_data[0]
            icl_example_data_info += icl_data[1]
    else:
        (icl_example_data, icl_example_data_info) = load_icl_example(
            icl_example_path=icl_example_path,
            icl_example_quantity=icl_example_quantity,
        )

    # Replace '\\n' to '\n'
    prompt = prompt.replace("\\n", "\n")
    icl_example_prompt = icl_example_prompt.replace("\\n", "\n")
    system_prompt = system_prompt.replace("\\n", "\n")

    run_task_info = OrderedDict(
        provider=provider,
        model_name=model_name,
        reasoning_effort=str(reasoning_effort),
        max_tokens=max_tokens,
        temperature=str(temperature),
        top_p=str(top_p),
        system_prompt=system_prompt,
        prompt=prompt,
        icl_example_prompt=icl_example_prompt,
        icl_example=icl_example_data_info,
        unclassified_icl_example_category=unclassified_icl_example_category,
    )

    if not inference_result_folder:
        hash_md5 = hashlib.sha256(str(run_task_info).encode("utf-8"))
        hash_value = hash_md5.hexdigest()[:16]
        logger.info(f"hash_value: {hash_value}")
        inference_result_folder = (
            f"{provider}-{model_name}-{hash_value}"
            if reasoning_effort == NOT_GIVEN
            else f"{provider}-{model_name}-{reasoning_effort}-{hash_value}"
        )
    api_key = api_key if api_key else os.getenv(f"{provider}_API_KEY".upper(), "")
    output_path: Path = Path(output_path if output_path else dataset_path.parent) / dataset_path.name / inference_result_folder
    assert api_key, "Not pass or set 'api_key'"

    if output_path.exists():
        with Path(output_path, ".task_info.txt").open(mode="r", encoding="utf-8") as f:
            exist_task_info = json.load(fp=f)
        assert run_task_info == exist_task_info or force_rerun, (
            f"inference_result_folder: '{inference_result_folder}' 發生衝突, 請重新給定名稱"
        )
    else:
        output_path.mkdir(parents=True, exist_ok=False)
        with Path(output_path, ".task_info.txt").open(mode="w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    obj=run_task_info,
                    ensure_ascii=False,
                    indent=4,
                )
            )

    client = openai.Client(
        base_url=_provider.get(provider, provider),
        api_key=api_key,
        max_retries=0,
        **kwds,
    )

    # Check model_name is exist
    try:
        for attempt in Retrying(
            reraise=True,
            stop=stop_after_attempt(retry),
            wait=wait_exponential(multiplier=1),
            retry=retry_if_exception_type(openai.RateLimitError),
            after=after_log(logger=logger, log_level=logging.INFO),
        ):
            with attempt:
                client.models.retrieve(model_name)
    except (openai.NotFoundError, openai.BadRequestError) as e:
        if isinstance(e, openai.NotFoundError):
            raise ValueError("沒有該模型")
        else:
            raise ValueError("API KEY 錯誤")

    image_filepaths: list[Path] = list()

    # Pre-check dataset are correct pairs for label txt data and image data
    for filepath in dataset_path.iterdir():
        if filepath.is_dir() or filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        image_filepaths.append(filepath)
    image_filepaths.sort()
    logger.debug(f"data: {image_filepaths}")

    # Eval
    for image_filepath in TQDM.tqdm(image_filepaths, desc=dataset_path.stem, smoothing=0) if tqdm else image_filepaths:
        inference_filepath = Path(output_path, f"{image_filepath.stem}")
        inference_filepath_exist = False
        if not force_rerun:
            for extension in [".txt", ".html", ".json"]:
                if Path(f"{inference_filepath!s}{extension}").exists():
                    inference_filepath_exist = True
                    break
            if inference_filepath_exist:
                continue

        logger.debug(f"Call api date: {dt.datetime.now()!s}")

        messages = list()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for icl_example in icl_example_data:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": icl_example["base64_image"],
                            },
                        },
                        {"type": "text", "text": icl_example_prompt},
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": icl_example["text"]},
                    ],
                }
            )

        contents = [
            {
                "type": "image_url",
                "image_url": {
                    "url": get_base64_image(image_filepath),
                },
            },
        ]
        if prompt:
            contents.append({"type": "text", "text": prompt})

        messages.append(
            {
                "role": "user",
                "content": contents,
            }
        )

        try:
            for attempt in Retrying(
                reraise=True,
                stop=stop_after_attempt(retry),
                wait=wait_exponential(multiplier=10),
                retry=retry_if_exception_type(openai.RateLimitError),
                after=after_log(logger=logger, log_level=logging.INFO),
            ):
                with attempt:
                    start_time = time.time()
                    responses: ChatCompletion = client.chat.completions.create(
                        messages=messages,
                        model=model_name,
                        max_completion_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        top_p=top_p,
                        temperature=temperature,
                    )
                    end_time = time.time()
        except Exception as e:
            raise e

        result = dict(
            generate_content=responses.choices[0].message.content,
            usage=responses.usage.model_dump(),
            time=end_time - start_time,
        )

        with Path(str(inference_filepath) + ".txt").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    obj=result,
                    ensure_ascii=False,
                    indent=4,
                )
            )

        time.sleep(1)


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
        run_online_model_result(
            dataset_path=dataset_path_split[0],
            **passed_parameters,
        )
