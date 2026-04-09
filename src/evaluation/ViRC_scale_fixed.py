import json
import math
import os
import re
import time

import cv2
from Logger import Logger
from PIL import Image
from QwenVL import QwenVL

IMAGE_PATH = "./src/evaluation/image.png"
QUERY = "Find $m\angle H$ Choices: 97 102 107 122"  # Correct answer is 97
RESPONSE_FOLDER = "./src/evaluation/response"
DETERMINISTIC = True  # Enable deterministic sampling
MAX_RETRY = 4  # Max retries per single dialogue
MAX_TURNS = 16  # Max turns per single round of dialogue

PROMPT_SYSTEM = r"""
You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"crop_image","description":"Crops an image, specified by its index, to a region defined by a bounding box (bbox).","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","description":"The bounding box [x1, y1, x2, y2] defining the region to crop. Coordinates are based on a permille (per-thousand) scale from 0 to 999, relative to the image's width and height.","items":{"type":"integer","minimum":0,"maximum":999},"minItems":4,"maxItems":4},"image_index":{"type":"integer","description":"A 0-based index specifying which image to crop.","minimum":0}},"required":["bbox_2d","image_index"]}}}
{"type":"function","function":{"name":"scale_image","description":"Scales a specific image by a given factor. Values greater than 1.0 zoom in, while values less than 1.0 zoom out.","parameters":{"type":"object","properties":{"scale_factor":{"type":"number","description":"The factor by which to scale the image. E.g., 2.0 for 200% magnification.","minimum":0.25,"maximum":4},"image_index":{"type":"integer","description":"A 0-based index specifying which image to scale.","minimum":0}},"required":["scale_factor","image_index"]}}}
{"type":"function","function":{"name":"display_image","description":"Displays a specific image by a given index for verification or recall.","parameters":{"type":"object","properties":{"image_index":{"type":"integer","description":"A 0-based index specifying which image to display.","minimum":0}},"required":["image_index"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:

**Example 1**:
<tool_call>
{"name": "crop_image", "arguments": {"bbox_2d": [0, 0, 100, 100], "image_index": 0}}
</tool_call>

**Example 2**:
<tool_call>
{"name": "scale_image", "arguments": {"scale_factor": 1.5, "image_index": 3}}
</tool_call>

**Example 3**:
<tool_call>
{"name": "display_image", "arguments": {"image_index": 0}}
</tool_call>
""".strip()
USER_QUERY = r"""
Image {index}: {width} x {height}.
Question: {question}

Let’s think step by step. Call **tool** if needed, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer> (if available).
""".strip()
USER_RESPONSE = r"""
Image {index}: {width} x {height}. Fully utilize the tools and make comprehensive use of the results. Keep thinking.
""".strip()
USER_RESPONSE_WITH_QUESTION = r"""
Image {index}: {width} x {height}.
Remember, after considering all of the above, primary goal is to answer this question: {question}
Fully utilize the tools and make comprehensive use of the results. Keep thinking.
""".strip()

LOGGER = Logger(logger_name="main")


def vllm_sampler_settings():
    result = {
        "session_id": "default_vl_session",
        "cache_dir": None,
        "system_prompt": PROMPT_SYSTEM,
        "model_name": "ViRC",
        "min_pixels": (28 * 2) ** 2,
        "max_pixels": (28 * 256) ** 2,
        "openai_api_key": "EMPTY",
        "openai_api_base": "https://aistudio.alipay.com/proxy/workflow_59640391:8000/v1",
    }
    if DETERMINISTIC:
        result.update({"temperature": 0, "top_p": 1, "top_k": 1})
    else:
        result.update({"temperature": 1.0, "top_p": 0.95, "top_k": 50})
    return result


def save_json_atomically(file_path, data: list[dict]):
    temp_file_path = file_path + ".tmp"
    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file_path, file_path)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        LOGGER.error(f"Error saving data to '{file_path}': {e}")


def smart_resize_optimized(
    width: int, height: int, max_ratio: int = 200, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    def smart_resize(
        width: int, height: int, max_ratio: int = 200, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}")
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, floor_by_factor(height / beta, factor))
            w_bar = max(factor, floor_by_factor(width / beta, factor))
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return w_bar, h_bar

    if width < 28 or height < 28:
        scale_factor = max(28 / width, 28 / height)
        width = math.ceil(width * scale_factor)
        height = math.ceil(height * scale_factor)
    new_width, new_height = smart_resize(width, height, max_ratio, factor, min_pixels, max_pixels)
    return new_width, new_height


def parse_tool_call(response_text: str) -> list | dict | None:
    all_matches = re.findall(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    if not all_matches:
        return None
    tool_code = all_matches[-1].strip().replace("\\", "").replace("”", '"').replace("“", '"')
    try:
        return json.loads(tool_code)
    except:
        return None


def generate_cache_path(source_path: str, data_id: str, operation_suffix: str) -> str:
    base_name = os.path.basename(source_path)
    name_part, suffix = os.path.splitext(base_name)
    if "#" in name_part:
        new_name = f"{name_part}{operation_suffix}{suffix}"
    else:
        new_name = f"{name_part}#{data_id}{operation_suffix}{suffix}"
    return os.path.join(RESPONSE_FOLDER, new_name)


def handle_crop(tool_info: dict, image_history: list, data_id: str) -> tuple[str, int] | None:
    idx = tool_info["arguments"]["image_index"]
    bbox = tool_info["arguments"]["bbox_2d"]
    if not 0 <= idx < len(image_history):
        LOGGER.warning(f"Invalid crop image index: index={idx}, image max index={len(image_history)-1}")
        return None

    source_path = image_history[idx]
    img = cv2.imread(source_path)
    if img is None:
        LOGGER.error(f"Failed to read image for cropping: {source_path}")
        return None

    h, w, _ = img.shape
    x1_rel, y1_rel, x2_rel, y2_rel = [int(c) for c in bbox]
    x1 = int(round(x1_rel / 999 * w))
    y1 = int(round(y1_rel / 999 * h))
    x2 = int(round(x2_rel / 999 * w))
    y2 = int(round(y2_rel / 999 * h))
    if x1 >= x2 or y1 >= y2:
        LOGGER.warning(f"Invalid crop bbox: {[x1, y1, x2, y2]} for image size {(w, h)}")
        return None

    cropped_img = img[y1:y2, x1:x2]
    new_width, new_height = smart_resize_optimized(x2 - x1, y2 - y1)
    cropped_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    op_suffix = f"_[{x1}, {y1}, {x2}, {y2}]_[{new_width}, {new_height}]"
    new_path = generate_cache_path(source_path, data_id, op_suffix)
    cv2.imwrite(new_path, cropped_img)
    image_history.append(new_path)
    return new_path, len(image_history) - 1, "crop"


def handle_scale(tool_info: dict, image_history: list, data_id: str) -> tuple[str, int] | None:
    idx = tool_info["arguments"]["image_index"]
    factor = tool_info["arguments"]["scale_factor"]
    factor = min([0.25, 0.5, 1.5, 2.0, 2.5, 4.0], key=lambda x: abs(x - factor))

    source_path = image_history[idx]
    img = cv2.imread(source_path)
    if img is None:
        LOGGER.error(f"Failed to read image for scaling: {source_path}")
        return None

    h, w, _ = img.shape
    new_w, new_h = smart_resize_optimized(int(w * factor), int(h * factor))
    if new_w == 0 or new_h == 0:
        LOGGER.warning(f"Zero in image size: new_w={new_w}, new_h={new_h}")
        return None

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LANCZOS4)
    op_suffix = f"_[{new_w}, {new_h}]"
    new_path = generate_cache_path(source_path, data_id, op_suffix)
    cv2.imwrite(new_path, resized_img)
    image_history.append(new_path)
    return new_path, len(image_history) - 1, "scale"


def handle_display(tool_info: dict, image_history: list, data_id: str) -> tuple[str, int] | None:
    idx = tool_info["arguments"]["image_index"]
    if not (isinstance(idx, int) and 0 <= idx < len(image_history)):
        LOGGER.warning(f"Invalid display image index: index={idx}, image max index={len(image_history)-1}")
        return None
    image_history.append(image_history[idx])
    return image_history[idx], len(image_history) - 1, "display"


def handle_tool_call(tool_info: dict, image_history: list, data_id: str) -> tuple[str, int] | None:
    # {"name": "crop_image", "arguments": {"bbox_2d": [0, 0, 100, 100], "image_index": 0}}
    # {"name": "scale_image", "arguments": {"scale_factor": 1.5, "image_index": 3}}
    # {"name": "display_image", "arguments": {"image_index": 0}}
    tool_name = tool_info.get("name")
    arguments = tool_info.get("arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return None
    if tool_name == "crop_image":
        bbox = arguments.get("bbox_2d")
        img_idx = arguments.get("image_index")
        if set(arguments.keys()) == {"bbox_2d", "image_index"} and isinstance(bbox, list) and len(bbox) == 4 and isinstance(img_idx, int):
            return handle_crop(tool_info, image_history, data_id)
    elif tool_name == "scale_image":
        scale_factor = arguments.get("scale_factor")
        img_idx = arguments.get("image_index")
        if set(arguments.keys()) == {"scale_factor", "image_index"} and isinstance(scale_factor, float) and isinstance(img_idx, int):
            return handle_scale(tool_info, image_history, data_id)
    elif tool_name == "display_image":
        img_idx = arguments.get("image_index")
        if set(arguments.keys()) == {"image_index"} and isinstance(img_idx, int):
            return handle_display(tool_info, image_history, data_id)
    return None


def find_repetitive_pattern(model_response: str, repetitions: int = 8) -> bool:
    if not model_response or int(repetitions) <= 1:
        return False
    pattern = re.compile(rf"(.+?)\1{{{int(repetitions)-1}}}")
    match = pattern.search(model_response)
    return True if match else False


def check_model_response_format(model_response: str) -> bool:
    matches = re.findall(r"<(/?)(think|tool_call|answer)>", model_response)
    return matches in [[("", "think"), ("/", "think"), ("", "answer"), ("/", "answer")], [("", "think"), ("/", "think"), ("", "tool_call"), ("/", "tool_call")]]


def refine_image(image_path: str, data_id: str) -> str:
    with Image.open(image_path) as img:
        img.load()
        width, height = img.size
        new_width, new_height = smart_resize_optimized(width, height)
        if (new_width, new_height) == (width, height):
            return image_path
        image_to_save = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        op_suffix = "_refined"
        cache_path = generate_cache_path(image_path, data_id, op_suffix)
        image_to_save.save(cache_path)
    return cache_path


def singleton(image_path: str, query: str, sampler: QwenVL):
    sampler.clear_history()
    image_history = []
    conversation_history = []
    data_id = str(int(time.time()))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original image not found: {image_path}")

    initial_processed_path = refine_image(image_path, data_id)
    image_history.append(initial_processed_path)
    initial_image = cv2.imread(initial_processed_path)

    h, w, _ = initial_image.shape
    w, h = smart_resize_optimized(w, h)
    initial_prompt = USER_QUERY.replace("{index}", str(0)).replace("{width}", str(w)).replace("{height}", str(h)).replace("{question}", query)
    current_prompt = initial_prompt
    current_image_path = initial_processed_path

    is_answer_found = False
    start_time = time.time()
    elapsed_time = 0
    for turn in range(MAX_TURNS):
        elapsed_time += time.time() - start_time
        start_time = time.time()
        for retry_attempt in range(MAX_RETRY):
            model_response = sampler.send_message(text_query=current_prompt, image_path=current_image_path)
            print(f"\n------Turn: {turn}------")
            print(model_response)

            if not model_response:
                LOGGER.error(f"API call failed for <{image_path}> <{query}>. Aborting file processing.")
                return

            if find_repetitive_pattern(model_response):
                sampler.rollback_last_turn()
                LOGGER.warning(
                    f"Turn {turn} Attempt {retry_attempt}/{MAX_RETRY} failed: find repetitive pattern. Retrying...\n<|model response|>\n{model_response}"
                )
                continue

            if not check_model_response_format(model_response):
                sampler.rollback_last_turn()
                LOGGER.warning(
                    f"Turn {turn} Attempt {retry_attempt}/{MAX_RETRY} failed: model response format wrong. Retrying...\n<|model response|>\n{model_response}"
                )
                continue

            if re.findall(r"<answer>(.*?)</answer>", model_response, re.DOTALL):
                is_answer_found = True
                break

            tool_call = parse_tool_call(model_response)
            if not tool_call:
                sampler.rollback_last_turn()
                LOGGER.warning(
                    f"Turn {turn} Attempt {retry_attempt}/{MAX_RETRY} failed: No valid tool call found. Retrying...\n<|model response|>\n{model_response}"
                )
                continue

            result = handle_tool_call(tool_call, image_history, data_id)
            if not result:
                sampler.rollback_last_turn()
                LOGGER.warning(f"Turn {turn} Attempt {retry_attempt}/{MAX_RETRY} failed: Tool handling failed. Retrying...\n<|tool_call|>\n{tool_call}")
                continue

            next_image_path, next_image_index, tool_name = result
            next_image = cv2.imread(next_image_path)
            if next_image is None:
                sampler.rollback_last_turn()
                LOGGER.warning(
                    f"Turn {turn} Attempt {retry_attempt}/{MAX_RETRY} failed: Could not load next image. Retrying...\n<|model response|>\n{model_response}"
                )
                continue

            break

        else:
            LOGGER.error(f"Turn {turn} failed after {MAX_RETRY} attempts. Aborting processing for this file.")
            return

        conversation_history.append({"role": "user", "content": current_prompt, "image": os.path.basename(current_image_path)})
        conversation_history.append({"role": "assistant", "content": model_response})

        if is_answer_found:
            break

        h_next, w_next, _ = next_image.shape
        w_next, h_next = smart_resize_optimized(w_next, h_next)
        current_image_path = next_image_path
        if tool_name == "scale":
            current_prompt = (
                USER_RESPONSE_WITH_QUESTION.replace("{index}", str(next_image_index))
                .replace("{width}", str(w_next))
                .replace("{height}", str(h_next))
                .replace("{question}", query)
            )
        else:
            current_prompt = USER_RESPONSE.replace("{index}", str(next_image_index)).replace("{width}", str(w_next)).replace("{height}", str(h_next))
    else:
        if not is_answer_found:
            LOGGER.warning(f"Reached max turns ({MAX_TURNS}) for <{image_path}> <{query}>. Ending conversation.")
            return
    LOGGER.info(f"Inference completed in {elapsed_time}s")
    save_json_atomically(os.path.join(RESPONSE_FOLDER, f"{data_id}.json"), conversation_history)
    return


if __name__ == "__main__":
    sampler = QwenVL(**vllm_sampler_settings())
    os.makedirs(RESPONSE_FOLDER, exist_ok=True)
    singleton(image_path=IMAGE_PATH, query=QUERY, sampler=sampler)
