# coding: utf-8

import argparse
import ast
import base64
import datetime as dt
import logging
import os
import pprint
import time
from os import PathLike
from pathlib import Path

import openai
import tqdm as TQDM
from openai.types.chat import ChatCompletion

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
        default=os.getenv("API_KEY", ""),
        help="Online model api key, can set in ENV with API_KEY",
    )
    parser.add_argument("--provider", type=str, required=True, help="Model provider")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Model max tokens")
    parser.add_argument("--prompt", type=str, required=True, help="Model prompt")
    parser.add_argument("--system_prompt", type=str, default="", help="Model system prompt")
    parser.add_argument("--inference_result_folder", type=str, required=True, help="Save inference result folder name")

    args = parser.parse_args()

    return args


def run_online_model_result(
    dataset_path: list[str],
    api_key: str,
    model_name: str,
    provider: str,
    prompt: str,
    inference_result_folder: str,
    system_prompt: str = "",
    max_tokens: int = 512,
    tqdm: bool = True,
    **kwds,
):
    client = openai.Client(
        base_url=_provider.get(provider, provider),
        api_key=api_key,
        **kwds,
    )

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
    for image_filepath, inference_filepath in TQDM.tqdm(data, desc="Eval", smoothing=0) if tqdm else data:
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

            responses: ChatCompletion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_completion_tokens=max_tokens,
                top_p=1.0,
                temperature=0.0,
            )
            generate_content = responses.choices[0].message.content

            Path(inference_filepath).parent.mkdir(exist_ok=True, parents=True)
            with Path(str(inference_filepath) + ".txt").open("w", encoding="utf-8") as f:
                f.write(generate_content)

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
