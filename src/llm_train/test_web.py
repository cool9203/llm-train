# coding: utf-8

import argparse
import base64
import io
import math
import os
import time
import traceback
from pathlib import Path
from typing import Annotated

import gradio as gr
import httpx
import pypandoc
import torch
from fastapi import FastAPI, File, Form, UploadFile
from peft import PeftConfig
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)

from llm_train import utils

_default_prompt = "latex table ocr"
_default_system_prompt = "You should follow the instructions carefully and explain your answers in detail."

__model: dict[str, PreTrainedModel | ProcessorMixin | PreTrainedTokenizer | str] = {
    "model": None,
    "tokenizer": None,
    "name": None,
    "device": None,
}

app = FastAPI()


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run test website to test unsloth training with llm or vlm")
    parser.add_argument("-m", "--model_name", type=str, default=None, help="Run model name or path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Run model generate max new tokens")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Run model device map")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4bit")
    parser.add_argument("--default_prompt", type=str, default=_default_prompt, help="Default prompt")
    parser.add_argument(
        "--default_system_prompt",
        type=str,
        default=_default_system_prompt,
        help="Default system prompt",
    )
    parser.add_argument("--example_folder", type=str, default="example", help="Example folder")
    parser.add_argument("--dev", dest="dev_mode", action="store_true", help="Dev mode")

    args = parser.parse_args()

    return args


class UsageToken(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class InferenceTableResponse(BaseModel):
    origin_content: str
    html_content: str
    images: list[str]
    tokens: UsageToken
    used_time: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _preprocess_image(
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


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    device_map: str = "cuda:0",
) -> tuple[PreTrainedModel, ProcessorMixin | PreTrainedTokenizer]:
    try:
        peft_config = PeftConfig.from_pretrained(model_name)
    except ValueError:
        peft_config = None

    model = AutoModelForVision2Seq.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    tokenizer = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=peft_config.base_model_name_or_path if peft_config else model_name,
        device_map=device_map,
    )

    return (model, tokenizer)


@torch.inference_mode()
def generate(
    image,
    prompt: str,
    system_prompt: str = _default_system_prompt,
    max_new_tokens: int = 1024,
    **kwds,
) -> dict[str, str | int]:
    model = __model.get("model")
    tokenizer = __model.get("tokenizer")
    messages = list()

    if system_prompt:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    )
    image = _preprocess_image(
        image,
        image_max_pixels=int(os.getenv("IMAGE_MAX_PIXELS", 1631220)),
        image_min_pixels=int(os.getenv("IMAGE_MIN_PIXELS", 0)),
    )
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with sdpa_kernel(
        backends=[
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    ):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwds,
        )

    # Reference: https://github.com/huggingface/transformers/issues/17117#issuecomment-1124497554
    return {
        "content": tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0],
        "tokens": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(outputs[:, inputs["input_ids"].shape[1] :][0]),
            "total_tokens": inputs["input_ids"].shape[1] + len(outputs[:, inputs["input_ids"].shape[1] :][0]),
        },
    }


@app.post("/api/inference_table")
def inference_table_api(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = _default_prompt,
    detect_table: Annotated[bool, Form()] = True,
    crop_table_padding: Annotated[int, Form()] = -60,
    max_tokens: Annotated[int, Form()] = 4096,
    model_name: Annotated[str, Form()] = None,
    system_prompt: Annotated[str, Form()] = _default_system_prompt,
    repair_latex: Annotated[bool, Form()] = False,
    full_border: Annotated[bool, Form()] = False,
    unsqueeze: Annotated[bool, Form()] = False,
    img_type: Annotated[str, Form()] = "png",
) -> InferenceTableResponse:
    output = _inference_table(
        image=image.file.read(),
        prompt=prompt,
        detect_table=detect_table,
        crop_table_padding=crop_table_padding,
        max_tokens=max_tokens,
        model_name=model_name,
        system_prompt=system_prompt,
        repair_latex=repair_latex,
        full_border=full_border,
        unsqueeze=unsqueeze,
    )

    for i in range(len(output.images)):
        with io.BytesIO(base64.b64decode(output.images[i].encode("utf-8"))) as origin_img_io:
            img: Image.Image = Image.open(origin_img_io)
            with io.BytesIO() as img_io:
                img.save(img_io, format=img_type)
                img_b64_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
                output.images[i] = f"data:{img_type};base64,{img_b64_str}"

    return output


def inference_table(
    image: Image.Image,
    prompt: str = _default_prompt,
    detect_table: bool = True,
    crop_table_padding: int = -60,
    max_tokens: int = 4096,
    model_name: str = None,
    system_prompt: str = _default_system_prompt,
    device_map: str = "auto",
    repair_latex: bool = False,
    full_border: bool = False,
    unsqueeze: bool = False,
):
    output = _inference_table(
        image=image,
        prompt=prompt,
        detect_table=detect_table,
        crop_table_padding=crop_table_padding,
        max_tokens=max_tokens,
        model_name=model_name,
        system_prompt=system_prompt,
        device_map=device_map,
        repair_latex=repair_latex,
        full_border=full_border,
        unsqueeze=unsqueeze,
    )
    return (
        output.origin_content,
        output.html_content,
        [Image.open(io.BytesIO(base64.b64decode(image.encode("utf-8")))) for image in output.images],
        output.tokens.completion_tokens / output.used_time if output.used_time > 0 else 0,
        output.tokens.completion_tokens,
        output.used_time,
    )


def _inference_table(
    image: str | bytes | Image.Image,
    prompt: str,
    detect_table: bool,
    crop_table_padding: int,
    max_tokens: int = 4096,
    model_name: str = None,
    system_prompt: str = _default_system_prompt,
    device_map: str = "auto",
    repair_latex: bool = False,
    full_border: bool = False,
    unsqueeze: bool = False,
) -> InferenceTableResponse:
    if isinstance(image, Image.Image):
        with io.BytesIO() as img_io:
            image.save(img_io, format="png")
            image = img_io.getvalue()
    elif isinstance(image, str):
        image = image.encode("utf-8")

    origin_responses = list()
    crop_images: list[bytes] = list()
    used_time = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if model_name and model_name != __model.get("name", None):
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
        )
        __model["name"] = model_name
        __model["device"] = str(__model["model"].device)

    if detect_table:
        with io.BytesIO(image) as file_io:
            resp = httpx.post(
                "http://10.70.0.232:9999/upload",
                files={"file": ("image.png", file_io)},
                data={
                    "action": "crop",
                    "padding": crop_table_padding,
                },
            )

        for crop_image_base64 in resp.json():
            crop_image_data = base64.b64decode(crop_image_base64)
            crop_images.append(crop_image_data)
    else:
        crop_images.append(image)

    try:
        for crop_image in crop_images:
            start_time = time.time()
            with io.BytesIO(crop_image) as img_io:
                generate_response = generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image=Image.open(img_io),
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    top_p=1.0,
                    top_k=None,
                    do_sample=False,
                    temperature=None,
                )
            end_time = time.time()
            used_time += end_time - start_time
            prompt_tokens += generate_response["tokens"]["prompt_tokens"]
            completion_tokens += generate_response["tokens"]["completion_tokens"]
            total_tokens += generate_response["tokens"]["total_tokens"]

            if repair_latex and utils.is_latex_table(generate_response["content"]):
                origin_responses.append(
                    utils.convert_pandas_to_latex(
                        df=utils.convert_latex_table_to_pandas(
                            latex_table_str=generate_response["content"],
                            headers=True,
                            unsqueeze=unsqueeze,
                        )[0],
                        full_border=full_border,
                    )
                )
            else:
                origin_responses.append(generate_response["content"])

        table_str = "".join(origin_responses)
        if utils.is_latex_table(table_str):
            html_response = pypandoc.convert_text(source=table_str, to="html", format="latex")
        elif utils.is_html_table(table_str):
            html_response = pypandoc.convert_text(source=table_str, to="html", format="html")
        else:
            html_response = pypandoc.convert_text(source=table_str, to="html", format="markdown")

    except Exception as e:
        html_response = "推論輸出無法解析"
        traceback.print_exception(e)

    return InferenceTableResponse(
        origin_content="\n\n".join(origin_responses),
        html_content=html_response,
        images=[base64.b64encode(crop_image).decode("utf-8") for crop_image in crop_images],
        tokens=UsageToken(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        used_time=used_time,
    )


def test_website(
    model_name: str = None,
    max_tokens: int = 4096,
    device_map: str = "cuda:0",
    dev_mode: bool = False,
    example_folder: str = "examples",
    default_prompt: str = _default_prompt,
    default_system_prompt: str = _default_system_prompt,
    **kwds,
) -> gr.Blocks:
    if model_name and __model.get("name") is None:
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
            load_in_4bit=kwds.get("load_in_4bit", False),
        )
        __model["name"] = model_name

    # Gradio 接口定義
    with gr.Blocks(
        title="VLM 生成表格測試網站",
        css="#component-6 { max-height: 85vh; }",
    ) as blocks:
        gr.Markdown("## VLM 生成表格測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="上傳圖片",
                    type="pil",
                    height="85vh",
                )

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")

        submit_button = gr.Button("生成表格")

        with gr.Row():
            with gr.Column():
                crop_table_results = gr.Gallery(label="偵測表格結果", format="png")

            with gr.Column():
                _model_name = gr.Textbox(
                    label="模型名稱或路徑",
                    value=__model.get("name", None),
                    visible=not model_name,
                )
                system_prompt_input = gr.Textbox(label="輸入系統文字提示", lines=2, value=default_system_prompt)
                prompt_input = gr.Textbox(label="輸入文字提示", lines=2, value=default_prompt)
                _max_tokens = gr.Slider(
                    label="Max tokens",
                    value=max_tokens,
                    minimum=1,
                    maximum=8192,
                    step=1,
                )
                detect_table = gr.Checkbox(label="是否自動偵測表格", value=True)
                crop_table_padding = gr.Slider(
                    label="偵測表格裁切框 padding",
                    value=-60,
                    minimum=-300,
                    maximum=300,
                    step=1,
                )
                repair_latex = gr.Checkbox(label="修復 latex", value=True, visible=dev_mode)
                full_border = gr.Checkbox(label="修復 latex 表格全框線", visible=dev_mode)
                unsqueeze = gr.Checkbox(label="修復 latex 並解開多行/列合併", visible=dev_mode)
                average_token = gr.Textbox(label="每秒幾個 token")
                all_complate_token = gr.Textbox(label="生成多少 token")
                usage_time = gr.Textbox(label="總花費時間")

        text_output = gr.Textbox(label="生成的文字輸出", visible=dev_mode)

        # Constant augments
        _device_map = gr.Textbox(value=device_map, visible=False)

        # Examples
        if Path(example_folder).exists():
            example_files = sorted(
                [
                    (str(path.resolve()), path.name)
                    for path in Path(example_folder).iterdir()
                    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ],
                key=lambda e: e[1],
            )
            gr.Examples(
                examples=[
                    [
                        Image.open(path),
                        _default_prompt,
                        True,
                        -60,
                        4096,
                        _model_name,
                        _default_system_prompt,
                        _device_map,
                        True,
                        False,
                        False,
                    ]
                    for path, name in example_files
                ],
                example_labels=[name for path, name in example_files],
                inputs=[
                    image_input,
                    prompt_input,
                    detect_table,
                    crop_table_padding,
                    _max_tokens,
                    _model_name,
                    system_prompt_input,
                    _device_map,
                    repair_latex,
                    full_border,
                    unsqueeze,
                ],
            )

        submit_button.click(
            inference_table,
            inputs=[
                image_input,
                prompt_input,
                detect_table,
                crop_table_padding,
                _max_tokens,
                _model_name,
                system_prompt_input,
                _device_map,
                repair_latex,
                full_border,
                unsqueeze,
            ],
            outputs=[
                text_output,
                html_output,
                crop_table_results,
                average_token,
                all_complate_token,
                usage_time,
            ],
        )
        return blocks


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    _default_prompt = args_dict["default_prompt"]
    _default_system_prompt = args_dict["default_system_prompt"]
    blocks = test_website(**args_dict)

    import uvicorn

    app = gr.mount_gradio_app(
        app=app,
        blocks=blocks,
        path="/",
    )
    uvicorn.run(
        app=app,
        host=args_dict["host"],
        port=args_dict["port"],
    )
