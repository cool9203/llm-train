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
    parser.add_argument("--system_prompt", type=str, default="", help="Model system prompt")
    parser.add_argument("--inference_result_folder", type=str, default=None, help="Save inference result folder name")
    parser.add_argument(
        "--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"], help="Reasoning budgets"
    )
    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    return args


def run_online_model_result(
    dataset_path: list[str],
    api_key: str | None,
    model_name: str,
    provider: str,
    prompt: str,
    reasoning_effort: str = "low",
    inference_result_folder: str = None,
    system_prompt: str = "",
    max_tokens: int = 512,
    retry: int = 3,
    tqdm: bool = True,
    **kwds,
) -> None:
    hash_md5 = hashlib.sha256(
        str(
            OrderedDict(
                provider=provider,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
                prompt=prompt,
            )
        ).encode("utf-8")
    )
    inference_result_folder = (
        inference_result_folder
        if inference_result_folder
        else f"{provider}-{model_name}-{reasoning_effort}-{hash_md5.hexdigest()[:16]}"
    )
    api_key = api_key if api_key else os.getenv(f"{provider}_API_KEY".upper(), "")
    assert api_key, "Not pass or set 'api_key'"

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
    for filepath in Path(dataset_path).iterdir():
        if filepath.is_dir() or filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        image_filepaths.append(filepath)
    image_filepaths.sort()
    logger.debug(f"data: {image_filepaths}")

    # Eval
    for image_filepath in TQDM.tqdm(image_filepaths, desc=Path(dataset_path).stem, smoothing=0) if tqdm else image_filepaths:
        inference_filepath = Path(Path(dataset_path), inference_result_folder, f"{image_filepath.stem}")
        if not Path(str(inference_filepath) + ".txt").exists() or not Path(str(inference_filepath) + ".html").exists():
            logger.debug(f"Call api date: {dt.datetime.now()!s}")

            messages = list()

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            with Path(image_filepath).open(mode="rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
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
                            top_p=1.0,
                        )
                        end_time = time.time()
            except Exception as e:
                raise e

            result = dict(
                generate_content=responses.choices[0].message.content,
                usage=responses.usage.model_dump(),
                time=end_time - start_time,
            )

            Path(inference_filepath).parent.mkdir(exist_ok=True, parents=True)
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
