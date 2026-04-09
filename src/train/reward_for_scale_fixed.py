# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

import base64
import json
import math
import re
from io import BytesIO

from openai import OpenAI
from PIL import Image
from verl.utils.import_utils import deprecated


class QwenVL:

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        model_name: str = "Qwen2.5-VL-72B-Instruct",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        openai_api_key: str = "EMPTY",
        openai_api_base: str = "http://localhost:8005/v1",
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )

    def send_answer_verify_message(self, question: str, answer_gt: str, answer_pred: str) -> str:
        prompt = (
            r"""
Below are two answers to the same question: [Question]. [Standard Answer] is the correct answer, and [Model Answer] is from a model's output. Compare them.
If [Model Answer] has the same meaning as [Standard Answer], even if expressed differently, they are consistent.
The model's output will contain the answer, regardless of its certainty. Just focus on the consistency of the answer, not the solution process. If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judgement as \boxed{0} or \boxed{1}.

[Question]: Who is wearing pants?
[Standard Answer]: A. The boy is wearing pants.
[Model Answer]: C. The girl in the picture is wearing pants.
Judgement: \boxed{0}

[Question]: Is the man phone both blue and closed?
[Standard Answer]: A. Yes, the man phone is both blue and closed.
[Model Answer]: No.
Judgement: \boxed{0}

[Question]: What color is the towel in the center of the picture?
[Standard Answer]: A. The towel in the center of the picture is blue.
[Model Answer]: The towel in the center of the picture is pink.
Judgement: \boxed{0}

[Question]: Is the countertop tan or blue?
[Standard Answer]: A. The countertop is tan.
[Model Answer]: tan
Judgement: \boxed{1}

[Question]: On which side of the picture is the barrier?
[Standard Answer]: A. The barrier is on the left side of the picture.
[Model Answer]: A
Judgement: \boxed{1}

[Question]: Is the kite brown and large?
[Standard Answer]: A. Yes, the kite is brown and large.
[Model Answer]: Yes
Judgement: \boxed{1}

[Question]: Are the spots on a giraffe?
[Standard Answer]: A. No, the spots are on a banana.
[Model Answer]: no
Judgement: \boxed{1}

[Question]: {question}
[Standard Answer]: {answer_gt}
[Model Answer]: {answer_pred}
Judgement:
""".strip()
            .replace("{question}", question)
            .replace("{answer_gt}", answer_gt)
            .replace("{answer_pred}", answer_pred)
        )

        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = f"WRONG! send_answer_verify_message: {e}"
            print(error_message)
            return None

    def send_reason_verify_message(self, image_bytes: str, question: str, answer: str, pre_think: str, latest_step: str) -> str:
        pre_think = "This student is just starting to think" if pre_think is None else pre_think
        prompt = (
            r"""
Question: {question}
The above is a text-and-image problem that a student is working on. The final answer is {answer}.

You are an experienced and fair math professor. Your task is to evaluate the quality of the student's latest problem-solving step **in the context of their previous thinking**. Your goal is to encourage solid reasoning while guiding the student towards a clear and effective solution.

Evaluate the `[latest step]` based on its contribution to solving the problem, considering the `[previous thinking]`.

**Evaluation Aspects:**

1.  **Correctness (Primary)**: Is the logic, reasoning, and computation in the `[latest step]` free of errors?
2.  **Strategic Contribution (Secondary)**: Does this step meaningfully advance the solution, given what was done before?
*   **High Contribution:** Directly moves towards the answer, **corrects a previous error**, or unblocks a dead end.
*   **Positive Contribution:** Involves **strategically verifying previous work, re-evaluating the plan based on past results**, or setting up a solid foundation for the next move.
*   **Low/Negative Contribution:** Is redundant (repeats a previous step without reason), irrelevant, or goes in a wrong direction based on the existing progress.
3.  **Clarity**: Is the step expressed clearly and without ambiguity?

Assign a score based on the following rubric. Put your final score within \boxed{}.

- **\boxed{1.0}**: Perfect. The step is correct, strategically brilliant given the context, and clear. The ideal next step.
- **\boxed{0.8-0.9}**: Excellent. The step is correct and represents a useful, logical progression. It could be a well-justified verification or a solid move forward.
- **\boxed{0.5-0.7}**: Good but flawed. The step has the right general idea but contains a minor logical/computational error. Or, it is correct but strategically weak (e.g., redundant) or poorly explained.
- **\boxed{0.2-0.4}**: Weak. The step is mostly incorrect or unhelpful in the current context.
- **\boxed{0.0-0.1}**: Useless or Harmful. The step is completely wrong, irrelevant, or derails a previously correct line of reasoning.

[previous thinking]: {pre_think}
[latest step]: {latest_step}
""".strip()
            .replace("{question}", question)
            .replace("{answer}", answer)
            .replace("{pre_think}", pre_think)
            .replace("{latest_step}", latest_step)
        )
        mime_mapping = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png"}
        image = Image.open(BytesIO(image_bytes))
        image_format = image.format.lower() if image.format else "jpeg"
        image_in_messages = f"data:{mime_mapping.get(image_format, 'image/jpeg')};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_in_messages}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = f"WRONG! send_reason_verify_message: {e}"
            print(error_message)
            return None

    def send_images_verify_message(self, image_bytes: str, focus_image_bytes: str, question: str, answer: str, latest_step: str) -> str:
        prompt = (
            r"""
Question: {question}
The above is a text-and-image problem that the user is working on. The final answer is {answer}.

You are an expert evaluator for an AI problem-solving agent. Your task is to assess the agent's latest submission and provide a single score from 0.0 to 1.0 based on its reasoning (`[latest step]`) and visual focus (`[latest focus]`).

**Core Principle:** Your evaluation must reward **efficiency and precision**.
*   A **precise crop/zoom** that isolates key details is considered **High Quality**.
*   Using the **full original image** is considered **inefficient** but contains all information, thus it is **Medium Quality**.
*   An irrelevant or misleading focus is **Low Quality**.

Follow this scoring guide to determine the final score.

### **Scoring Guide**

*   **Score 1.0 (Excellent):**
*   **Criteria:** The reasoning is logical and correctly justifies an action, AND the visual focus is a **precise crop/zoom (High Quality)** that effectively supports that reasoning.

*   **Score 0.8 (Good Reasoning, Inefficient Focus):**
*   **Criteria:** The reasoning is logical and correct, BUT the visual focus is the **full image or a suboptimal crop (Medium Quality)**. This is a correct but inefficient step.

*   **Score 0.7 (Weak Reasoning, Good Focus):**
*   **Criteria:** The reasoning is vague or slightly flawed, BUT the agent happens to select a **highly effective crop/zoom (High Quality)**.

*   **Score 0.5 (Mediocre):**
*   **Criteria:** The reasoning is vague, AND the visual focus is the **full image or a suboptimal crop (Medium Quality)**.

*   **Low Scores (0.0 - 0.4):**
*   These scores are for submissions where either the reasoning or the visual focus is fundamentally poor (Low Quality).
*   **0.4:** Logical reasoning leads to a useless/misleading focus.
*   **0.3:** Incorrect reasoning happens to land on a great focus (a "lucky guess").
*   **0.2:** Vague reasoning paired with a useless focus.
*   **0.1:** Incorrect reasoning paired with an inefficient (full image) focus.
*   **0.0:** Both reasoning and visual focus are incorrect, irrelevant, or nonsensical.

Please provide only the final score within \boxed{}.

**Problem Information:**
[latest step]: {latest_step}
[latest focus]:
""".strip()
            .replace("{question}", question)
            .replace("{answer}", answer)
            .replace("{latest_step}", latest_step)
        )
        mime_mapping = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png"}
        image = Image.open(BytesIO(image_bytes))
        image_format = image.format.lower() if image.format else "jpeg"
        image_in_messages = f"data:{mime_mapping.get(image_format, 'image/jpeg')};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        focus_image = Image.open(BytesIO(focus_image_bytes))
        focus_image_format = focus_image.format.lower() if focus_image.format else "jpeg"
        focus_image_in_messages = f"data:{mime_mapping.get(focus_image_format, 'image/jpeg')};base64,{base64.b64encode(focus_image_bytes).decode('utf-8')}"

        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_in_messages}},
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": focus_image_in_messages}},
                ],
            }
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = f"WRONG! send_reason_verify_message: {e}"
            print(error_message)
            return None


def check_tool_format(tool_info: dict) -> bool:
    if tool_info is None or not isinstance(tool_info, dict) or "name" not in tool_info or "arguments" not in tool_info or len(tool_info) != 2:
        return False
    name = tool_info["name"]
    arguments = tool_info["arguments"]
    if not isinstance(arguments, dict) or "image_index" not in arguments or not isinstance(arguments["image_index"], int):
        return False
    if name == "crop_image":
        if len(arguments) != 2 or "bbox_2d" not in arguments:
            return False
        bbox_2d = arguments["bbox_2d"]
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4 or not all(isinstance(x, (int, float)) for x in bbox_2d):
            return False
        return True
    elif name == "scale_image":
        if len(arguments) != 2 or "scale_factor" not in arguments:
            return False
        scale_factor = arguments["scale_factor"]
        if not isinstance(scale_factor, (int, float)):
            return False
        return True
    elif name == "display_image":
        if len(arguments) != 1:
            return False
        return True
    else:
        return False


def parse_think(model_response: str) -> str | None:
    all_matches = re.findall(r"<think>(.*?)</think>", model_response, re.DOTALL)
    if not all_matches:
        return None
    answer = all_matches[-1].strip()
    return answer


def parse_tool_call(response_text: str) -> list | dict | None:
    all_matches = re.findall(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)
    if not all_matches:
        return None
    tool_code = all_matches[-1].strip()
    try:
        return json.loads(tool_code)
    except:
        return None


def parse_answer(model_response: str) -> str | None:
    all_matches = re.findall(r"<answer>(.*?)</answer>", model_response, re.DOTALL)
    if not all_matches:
        return None
    answer = all_matches[-1].strip()
    return answer


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


def parse_boxed(text: str) -> str | None:
    def extract_last_boxed_content(text: str) -> str | None:
        try:
            last_box_start_index = text.rindex(r"\boxed{")
        except ValueError:
            return None

        content_start_index = last_box_start_index + len(r"\boxed{")

        level = 1
        for i in range(content_start_index, len(text)):
            char = text[i]
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1

            if level == 0:
                return text[content_start_index:i]

        return None

    last_box_content = extract_last_boxed_content(text)

    if last_box_content is None:
        return None

    deeper_content = parse_boxed(last_box_content)

    if deeper_content is not None:
        return deeper_content.strip()
    else:
        return last_box_content.strip()


def handle_scale(image_bytes: str, scale_factor: int | float) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    width, height = smart_resize_optimized(width * scale_factor, height * scale_factor)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    output_buffer = BytesIO()
    format = img.format if img.format else "PNG"
    img.save(output_buffer, format=format)
    return output_buffer.getvalue()

# # for Qwen2.5-VL
# def handle_crop(image_bytes: str, bbox_2d: tuple[int, int, int, int]) -> bytes:
#     img = Image.open(BytesIO(image_bytes))
#     x1, y1, x2, y2 = bbox_2d
#     cropped_img = img.crop((x1, y1, x2, y2))
#     output_buffer = BytesIO()
#     format = img.format if img.format else "PNG"
#     cropped_img.save(output_buffer, format=format)

#     return output_buffer.getvalue()

# for Qwen2-VL
def handle_crop(image_bytes: bytes, bbox_2d: tuple[int, int, int, int]) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    x1_rel, y1_rel, x2_rel, y2_rel = bbox_2d
    x1 = int(round(x1_rel / 999 * width))
    y1 = int(round(y1_rel / 999 * height))
    x2 = int(round(x2_rel / 999 * width))
    y2 = int(round(y2_rel / 999 * height))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    # if x2 <= x1 or y2 <= y1:
    #     raise ValueError("Invalid bbox after conversion to absolute coordinates.")
    cropped_img = img.crop((x1, y1, x2, y2))
    output_buffer = BytesIO()
    fmt = img.format if img.format else "PNG"
    cropped_img.save(output_buffer, format=fmt)

    return output_buffer.getvalue()

def find_repetitive_pattern(model_response: str, repetitions: int = 8) -> bool:
    if not model_response or int(repetitions) <= 1:
        return False
    pattern = re.compile(rf"(.+?)\1{{{int(repetitions)-1}}}")
    match = pattern.search(model_response)
    return True if match else False


def cal_bbox_reward(box1, box2, eps=1e-7) -> float:
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + eps)

    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])

    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    diagonal_sq = enclose_w**2 + enclose_h**2

    distance_penalty = center_dist_sq / (diagonal_sq + eps)

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    atan1 = math.atan(w1 / (h1 + eps))
    atan2 = math.atan(w2 / (h2 + eps))
    v = (4 / math.pi**2) * (atan1 - atan2) ** 2

    alpha = v / (1 - iou + v + eps)

    shape_penalty = alpha * v

    ciou = iou - distance_penalty - shape_penalty

    return max(0, ciou)


def default_compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    IS_DEBUG = True

    def _log(*args, **kwargs):
        if IS_DEBUG:
            print(*args, **kwargs)

    # format_reward, if wrong , direct return wrong_format_reward
    wrong_format_reward = -1
    if find_repetitive_pattern(solution_str):
        return wrong_format_reward
    matches = re.findall(r"<(/?)(think|tool_call|answer)>", solution_str)
    if matches not in [
        [("", "think"), ("/", "think"), ("", "answer"), ("/", "answer")],
        [("", "think"), ("/", "think"), ("", "tool_call"), ("/", "tool_call")],
    ]:
        return wrong_format_reward
    latest_step = parse_think(solution_str)
    answer_pred = parse_answer(solution_str)
    tool_pred = parse_tool_call(solution_str)
    # model return answer
    if latest_step is not None and answer_pred is not None and tool_pred is None:
        pass
    # model return tool
    elif latest_step is not None and answer_pred is None and tool_pred is not None:
        if not check_tool_format(tool_pred):
            return wrong_format_reward
        # check image index and "display_image"
        max_image_index = len(extra_info["images"]) - 1
        called_image_index = tool_pred["arguments"]["image_index"]
        if called_image_index > max_image_index:
            return wrong_format_reward
        if tool_pred["name"] == "scale_image":
            scale_factor = tool_pred["arguments"]["scale_factor"]
            if not (0.25 <= scale_factor <= 0.5 or 1.5 <= scale_factor <= 4.0):
                return wrong_format_reward
        if tool_pred["name"] == "crop_image":
            called_image = Image.open(BytesIO(extra_info["images"][called_image_index]["bytes"]))
            width, height = called_image.size
            x1, y1, x2, y2 = tool_pred["arguments"]["bbox_2d"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                return wrong_format_reward
    else:
        return wrong_format_reward

    vlm = QwenVL()
    MAX_RETRY = 4
    correct_reward = 0
    tool_text_reward = 0
    reasoning_reward = 0
    tool_correct_reward = 0

    # correct_reward
    if latest_step is not None and answer_pred is not None and tool_pred is None:
        for _ in range(MAX_RETRY):
            vlm_judge = vlm.send_answer_verify_message(
                question=extra_info["question"],
                answer_gt=ground_truth,
                answer_pred=answer_pred,
            )
            if vlm_judge is None:
                continue
            vlm_judge_result = parse_boxed(vlm_judge)
            if vlm_judge_result == "1":
                correct_reward = 0.8
                _log(f"final_reward: {correct_reward:.2f}")
                return correct_reward
            if vlm_judge_result == "0":
                correct_reward = 0
                _log(f"final_reward: {correct_reward:.2f}")
                return correct_reward
        else:
            _log(f"send_answer_verify_message failed: {vlm_judge}")

    if latest_step is not None and answer_pred is None and tool_pred is not None:
        pre_think = []
        for message in extra_info["prompt"]:
            if message["role"] == "assistant":
                pre_think.append(parse_think(message["content"]))
        pre_think = " ".join(pre_think)
        called_image_index = tool_pred["arguments"]["image_index"]
        image_bytes = handle_scale(extra_info["images"][0]["bytes"], 1)
        focus_image_bytes = extra_info["images"][called_image_index]["bytes"]
        if tool_pred["name"] == "display_image":
            focus_image_bytes = handle_scale(focus_image_bytes, 1)
        elif tool_pred["name"] == "scale_image":
            focus_image_bytes = handle_scale(focus_image_bytes, tool_pred["arguments"]["scale_factor"])
        elif tool_pred["name"] == "crop_image":
            focus_image_bytes = handle_crop(focus_image_bytes, tool_pred["arguments"]["bbox_2d"])
            focus_image_bytes = handle_scale(focus_image_bytes, 1)
        else:
            raise ValueError("tool info wrong!")

        # tool_correct_reward
        if tool_pred["name"] == extra_info["tool_info_gt"]["name"]:
            tool_correct_reward = 0.1

        # reasoning_reward
        for _ in range(MAX_RETRY):
            vlm_judge = vlm.send_reason_verify_message(
                image_bytes=image_bytes,
                question=extra_info["question"],
                answer=ground_truth,
                pre_think=pre_think,
                latest_step=latest_step,
            )
            if vlm_judge is None:
                continue
            vlm_judge_result = parse_boxed(vlm_judge)
            try:
                if "-" in vlm_judge_result:
                    temp = float(vlm_judge_result.split("-")[0].strip())
                    if 0 <= temp <= 1:
                        reasoning_reward = temp
                    break
                if 0 <= float(vlm_judge_result) <= 1:
                    reasoning_reward = float(vlm_judge_result)
                    break
            except:
                _log(f"send_reason_verify_message value parsing failed: {vlm_judge}")
        else:
            _log(f"send_reason_verify_message failed: {vlm_judge}")

        # tool_text_reward
        for _ in range(MAX_RETRY):
            vlm_judge = vlm.send_images_verify_message(
                image_bytes=image_bytes,
                focus_image_bytes=focus_image_bytes,
                question=extra_info["question"],
                answer=ground_truth,
                latest_step=latest_step,
            )
            if vlm_judge is None:
                continue
            vlm_judge_result = parse_boxed(vlm_judge)
            try:
                if 0 <= float(vlm_judge_result) <= 1:
                    tool_text_reward = float(vlm_judge_result)
                    break
            except:
                _log(f"send_images_verify_message value parsing failed: {vlm_judge}")
        else:
            _log(f"send_images_verify_message failed: {vlm_judge}")
    # Model rarely outputs max value == 1
    final_reward = reasoning_reward * 0.5 + tool_text_reward * 0.4 + tool_correct_reward
    _log(
        f"final_reward: {final_reward:.2f}\tonly_text_reward:{reasoning_reward:.2f}\ttool_text_reward:{tool_text_reward:.2f}\ttool_correct_reward:{tool_correct_reward:.2f}"
    )
    return final_reward


# if __name__ == "__main__":
#     solution_str = r"<think></think><tool_call>{'name': 'refer_image', 'arguments': {'image_index': 6}}</tool_call>"
#     print(default_compute_score(data_source=None, solution_str=solution_str, ground_truth=None))
#     pass


def default_compute_score_backup(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)


__all__ = ["default_compute_score"]
