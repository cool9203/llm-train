# LLM train

Just a train script with [trl](https://github.com/huggingface/trl)

Support load local data and load [llamafactory](https://github.com/hiyouga/LLaMA-Factory) sharegpt data format

## Convert dataset

### Data folder layout

<details>

<summary>Show data folder layout</summary>

Label file can be `.txt` or `.html`

|- root
|--- A.jpg
|--- A.txt
|--- B.jpg
|--- B.txt
|--- C.jpg
|--- C.html

</details>

### Command

```bash
python scripts/dataset_converter/from_img_and_txt.py \
    --image_path <INPUT_FOLDER_PATH_1> <INPUT_FOLDER_PATH_2> \
    --prompt <PROMPT> \
    --system_prompt <SYSTEM_PROMPT>
    -o <OUTPUT_PATH.json> \
    --output_format <TABLE_FORMAT> \
    --image_path <IMAGE_PATH> \
    --reasoning \
    --code_block \
    --row_comment <ROW_COMMENT_LINE_COUNT>> \
    --tqdm
```

## Train

```bash
bash scripts/qwen25-sft.sh
```

## Eval table data

### Data folder layout

<details>

<summary>Show data folder layout</summary>

Label file can be `.txt` or `.html`

|- root
|--- INFERENCE_DATA
|----- A.txt
|----- B.html
|----- C.txt
|--- A.jpg
|--- A.txt
|--- B.jpg
|--- B.txt
|--- C.jpg
|--- C.html

</details>

### Command

```bash
python scripts/eval/table_correct_rate.py \
    --inference_result_folder <INFERENCE_FOLDER_NAME> \
    --ignore_headers <IGNORE_HEADER_REGEX_1> <IGNORE_HEADER_REGEX_2> \
    --ignore_values <IGNORE_ROW_CONTAIN_VALUE_REGEX_1> <IGNORE_ROW_CONTAIN_VALUE_REGEX_2> \
    --detect_headers <DETECT_HEADER_1> <DETECT_HEADER_2> \
    --datasets <EVAL_DATASET_PATH_1> <EVAL_DATASET_PATH_2>
```

## Run test web

```bash
DEFAULT_PROMPT=<DEFAULT_PROMPT> \
DEFAULT_SYSTEM_PROMPT=<DEFAULT_SYSTEM_PROMPT> \
python src/llm_train/test_web.py \
    --model_name <BASE_MODEL_NAME> \
    --lora-modules <ADAPTER_NAME_1>=<ADAPTER_MODEL_PATH_1> <ADAPTER_NAME_2>=<ADAPTER_MODEL_PATH_2> \
    --max_tokens <MAX_TOKENS> \
    --host <HOST> \
    --port <PORT> \
    --device_map <DEVICE_MAP> \
    --dev
```
