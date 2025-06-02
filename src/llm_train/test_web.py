# coding: utf-8

import argparse
import base64
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Annotated

import gradio as gr
import pypandoc
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

_default_prompt = os.getenv("DEFAULT_PROMPT", "請使用html解析圖片裡的表格")
_default_system_prompt = os.getenv("DEFAULT_SYSTEM_PROMPT", "")

__model: dict[str, PreTrainedModel | ProcessorMixin | PreTrainedTokenizer | str] = {
    "model": None,
    "tokenizer": None,
    "name": None,
    "adapters": [],
}

app = FastAPI()


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run test website")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Run base model name or path")
    parser.add_argument("--lora-modules", type=str, nargs="+", default=[], help="Lora adapter name or path, format follow vllm")
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
    model_name: str
    tokens: UsageToken
    used_time: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    device_map: str = "cuda:0",
    lora_modules: list[dict[str, str]] = [],
) -> tuple[PreTrainedModel, ProcessorMixin | PreTrainedTokenizer]:
    model: PreTrainedModel = AutoModelForVision2Seq.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    for lora_module in lora_modules:
        model.load_adapter(
            peft_model_id=lora_module["path"],
            adapter_name=lora_module["name"],
        )

    tokenizer = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        padding_side="left",
        device_map=device_map,
    )

    return (model, tokenizer)


@torch.inference_mode()
def generate(
    images: list[Image.Image],
    prompt: str,
    system_prompt: str = _default_system_prompt,
    max_new_tokens: int = 1024,
    **kwds,
) -> list[dict[str, str | int]]:
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
    images = [
        utils.preprocess_image(
            image,
            image_max_pixels=int(os.getenv("IMAGE_MAX_PIXELS", 1631220)),
            image_min_pixels=int(os.getenv("IMAGE_MIN_PIXELS", 0)),
        )
        for image in images
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text=[input_text for _ in range(len(images))],
        images=images,
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
    decode_outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    encode_outputs = tokenizer(text=decode_outputs)
    return [
        {
            "content": decode_outputs[i],
            "tokens": {
                "prompt_tokens": sum(inputs["attention_mask"][i]),
                "completion_tokens": len(encode_outputs["input_ids"][i]) + 1,  # Add EOS token
                "total_tokens": sum(inputs["attention_mask"][i]) + len(encode_outputs["input_ids"][i]) + 1,  # Add EOS token
            },
        }
        for i in range(len(outputs))
    ]


@app.get("/api/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
            }
            for model_id in ["__base__"] + __model["adapters"]
        ],
    }


@app.post("/api/batch_inference_table")
def batch_inference_table_api(
    images: Annotated[list[UploadFile], File()],
    prompt: Annotated[str, Form()] = _default_prompt,
    max_tokens: Annotated[int, Form()] = 4096,
    model_name: Annotated[str, Form()] = None,
    system_prompt: Annotated[str, Form()] = _default_system_prompt,
    img_type: Annotated[str, Form()] = "png",
) -> InferenceTableResponse:
    try:
        _images = [image.file.read() for image in images]
        outputs = inference_table(
            images=_images,
            prompt=prompt,
            max_tokens=max_tokens,
            model_name=model_name,
            system_prompt=system_prompt,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    for i in range(len(outputs)):
        for j in range(len(outputs[i].images)):
            with io.BytesIO(base64.b64decode(outputs[i].images[j].encode("utf-8"))) as origin_img_io:
                img: Image.Image = Image.open(origin_img_io)
                with io.BytesIO() as img_io:
                    img.save(img_io, format=img_type)
                    img_b64_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
                    outputs[i].images[j] = f"data:{img_type};base64,{img_b64_str}"

    return outputs


@app.post("/api/inference_table")
def inference_table_api(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = _default_prompt,
    max_tokens: Annotated[int, Form()] = 4096,
    model_name: Annotated[str, Form()] = None,
    system_prompt: Annotated[str, Form()] = _default_system_prompt,
    img_type: Annotated[str, Form()] = "png",
) -> InferenceTableResponse:
    try:
        output = inference_table(
            image=image.file.read(),
            prompt=prompt,
            max_tokens=max_tokens,
            model_name=model_name,
            system_prompt=system_prompt,
        )[0]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    for i in range(len(output.images)):
        with io.BytesIO(base64.b64decode(output.images[i].encode("utf-8"))) as origin_img_io:
            img: Image.Image = Image.open(origin_img_io)
            with io.BytesIO() as img_io:
                img.save(img_io, format=img_type)
                img_b64_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
                output.images[i] = f"data:{img_type};base64,{img_b64_str}"

    return output


def inference_table_gradio(
    image: Image.Image,
    prompt: str = _default_prompt,
    max_tokens: int = 4096,
    model_name: str = None,
    system_prompt: str = _default_system_prompt,
):
    output = inference_table(
        images=[image],
        prompt=prompt,
        max_tokens=max_tokens,
        model_name=model_name,
        system_prompt=system_prompt,
    )[0]
    return (
        output.origin_content,
        output.html_content,
        [Image.open(io.BytesIO(base64.b64decode(image.encode("utf-8")))) for image in output.images],
        output.tokens.completion_tokens / output.used_time if output.used_time > 0 else 0,
        output.tokens.completion_tokens,
        output.used_time,
    )


def inference_table(
    images: list[str | bytes | Image.Image],
    prompt: str,
    model_name: str,
    max_tokens: int = 4096,
    system_prompt: str = _default_system_prompt,
    **kwds,
) -> list[InferenceTableResponse]:
    for i in range(len(images)):
        if isinstance(images, Image.Image):
            with io.BytesIO() as img_io:
                images[i].save(img_io, format="png")
                images[i] = img_io.getvalue()
        elif isinstance(images[i], str):
            images[i] = images[i].encode("utf-8")

    if model_name and model_name in __model.get("adapters", []):
        __model["model"].enable_adapters()
        __model["model"].set_adapter(model_name)
    elif model_name in ["__base__", __model["name"]]:
        model_name = __model["name"]
        __model["model"].disable_adapters()
    else:
        ValueError(f"Not support this model: '{model_name}")

    generate_responses = list()
    used_time = 0

    try:
        start_time = time.time()
        images_io = [io.BytesIO(image) for image in images]
        converted_images = [Image.open(img_io) for img_io in images_io]
        generate_responses = generate(
            prompt=prompt,
            system_prompt=system_prompt,
            images=converted_images,
            max_new_tokens=max_tokens,
            use_cache=True,
            top_p=1.0,
            top_k=None,
            do_sample=False,
            temperature=None,
        )
        end_time = time.time()
        used_time = end_time - start_time
        for i in range(len(images_io)):
            images_io[i].close()
            converted_images[i].close()

        for generate_response in generate_responses:
            if utils.is_latex_table(generate_response["content"]):
                html_content = pypandoc.convert_text(source=generate_response["content"], to="html", format="latex")
            elif utils.is_html_table(generate_response["content"]):
                html_content = pypandoc.convert_text(source=generate_response["content"], to="html", format="html")
            else:
                html_content = pypandoc.convert_text(source=generate_response["content"], to="html", format="markdown")
            generate_response["html_content"] = html_content

    except Exception as e:
        traceback.print_exception(e)

    return [
        InferenceTableResponse(
            origin_content="\n\n".join(generate_responses[i]["content"]),
            html_content=generate_responses.get("html_content", "推論輸出無法解析"),
            images=[base64.b64encode(images[i]).decode("utf-8")],
            model_name=model_name,
            tokens=UsageToken(
                prompt_tokens=generate_responses[i]["tokens"]["prompt_tokens"],
                completion_tokens=generate_responses[i]["tokens"]["completion_tokens"],
                total_tokens=generate_responses[i]["tokens"]["total_tokens"],
            ),
            used_time=used_time,
        )
        for i in range(len(generate_responses))
    ]


def test_website(
    model_name: str,
    lora_modules: list[str] = [],
    max_tokens: int = 4096,
    device_map: str = "cuda:0",
    dev_mode: bool = False,
    example_folder: str = "examples",
    default_prompt: str = _default_prompt,
    default_system_prompt: str = _default_system_prompt,
    **kwds,
) -> gr.Blocks:
    model_names = list([model_name])

    # Format lora_modules
    formatted_lora_modules: list[dict[str, str]] = list()
    try:
        for lora_module in lora_modules:
            formatted_lora_modules.append(json.loads(lora_module))
    except json.JSONDecodeError:
        formatted_lora_modules.clear()
        for lora_module in lora_modules:
            lora_module_split = lora_module.split("=")
            formatted_lora_modules.append(
                {
                    "name": lora_module_split[0],
                    "path": lora_module_split[1],
                }
            )
    model_names += [lora_module["name"] for lora_module in formatted_lora_modules]

    # Load base model and adapter
    (__model["model"], __model["tokenizer"]) = load_model(
        model_name=model_name,
        device_map=device_map,
        load_in_4bit=kwds.get("load_in_4bit", False),
        lora_modules=formatted_lora_modules,
    )
    __model["name"] = model_name
    __model["adapters"] = model_names[1:]

    # Gradio 接口定義
    with gr.Blocks(
        title="VLM 生成表格測試網站",
        css="#component-6 { max-height: 85vh; }",
    ) as blocks:
        gr.Markdown("## VLM 生成表格測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="上傳圖片", type="pil")

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")

        submit_button = gr.Button("生成表格")

        with gr.Row():
            with gr.Column():
                crop_table_results = gr.Gallery(label="偵測表格結果", format="png")

            with gr.Column():
                _model_name = gr.Dropdown(
                    choices=model_names,
                    label="模型名稱",
                    value=model_names[1] if len(model_names) > 1 else model_names[0],
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
                average_token = gr.Textbox(label="每秒幾個 token")
                all_completion_token = gr.Textbox(label="生成多少 token")
                usage_time = gr.Textbox(label="總花費時間")

        text_output = gr.Textbox(label="生成的文字輸出", visible=dev_mode)

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
                        4096,
                        _model_name,
                        _default_system_prompt,
                    ]
                    for path, name in example_files
                ],
                example_labels=[name for path, name in example_files],
                inputs=[
                    image_input,
                    prompt_input,
                    _max_tokens,
                    _model_name,
                    system_prompt_input,
                ],
            )

        submit_button.click(
            inference_table_gradio,
            inputs=[
                image_input,
                prompt_input,
                _max_tokens,
                _model_name,
                system_prompt_input,
            ],
            outputs=[
                text_output,
                html_output,
                crop_table_results,
                average_token,
                all_completion_token,
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
