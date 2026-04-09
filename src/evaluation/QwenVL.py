import base64
import json
import os
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, APIError, BadRequestError, OpenAI


class QwenVL:

    def __init__(
        self,
        session_id: str = "default_vl_session",
        cache_dir: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        model_name: str = "default",
        min_pixels: int = (28 * 1) ** 2,
        max_pixels: int = (28 * 256) ** 2,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = 50,
        openai_api_key: str = "EMPTY",
        openai_api_base: str = "http://localhost:8000/v1",
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base

        self.session_id = session_id
        self.cache_dir = cache_dir
        self._cache_path = None

        if self.cache_dir is not None:
            self._cache_path = os.path.join(self.cache_dir, f"{self.session_id}.json")

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )

        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def _save_history(self):
        if self.cache_dir is None or not self._cache_path:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=None, separators=(",", ":"))
        except IOError as e:
            print(f"Failed to save history to '{self._cache_path}': {e}")

    def _encode_image_to_base64(self, image_path: str) -> tuple[str, str]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        file_extension = os.path.splitext(image_path)[1].lower()
        mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime_types.get(file_extension)
        if not mime_type:
            raise ValueError(f"Unsupported image format: {file_extension}")
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        return encoded_image, mime_type

    def _build_user_content(self, text_query: str = None, image_path: str = None) -> list:
        user_content = []
        if image_path:
            base64_image, mime_type = self._encode_image_to_base64(image_path)
            image_in_messages = f"data:{mime_type};base64,{base64_image}"
            user_content.append({"type": "image_url", "image_url": {"url": image_in_messages}})
        if text_query:
            user_content.append({"type": "text", "text": text_query})
        return user_content

    def send_message(self, text_query: str = None, image_path: str = None) -> str:
        try:
            user_content = self._build_user_content(text_query, image_path)
        except (FileNotFoundError, ValueError) as e:
            return f"Error: {e}"

        if not user_content:
            return "Error: Please provide text or an image."

        self.messages.append({"role": "user", "content": user_content})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
            )
            model_response = completion.choices[0].message.content

            self.messages.append({"role": "assistant", "content": model_response})
            self._save_history()
            return model_response

        except (APIError, BadRequestError, APIConnectionError) as e:
            self.messages.pop()
            print(f"API call failed: {e}")
            return None
        except Exception as e:
            self.messages.pop()
            print(f"Unexpected error: {e}")
            return None

    def send_message_without_memory(self, text_query: str = None, image_path: str = None) -> str:
        try:
            user_content = self._build_user_content(text_query, image_path)
        except (FileNotFoundError, ValueError) as e:
            return f"Error: {e}"
        if not user_content:
            return "Error: Please provide text or an image."

        temp_messages = self.messages.copy()
        temp_messages.append({"role": "user", "content": user_content})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=temp_messages,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=self.top_p,
                extra_body={"top_k": self.top_k},
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in stateless message: {e}")
            return None

    def get_history(self) -> list:
        return self.messages

    def rollback_last_turn(self):
        if len(self.messages) >= 2 and self.messages[-1]["role"] == "assistant" and self.messages[-2]["role"] == "user":
            self.messages.pop()
            self.messages.pop()
            return True
        else:
            return False

    def clear_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

        if self.cache_dir is not None and self._cache_path and os.path.exists(self._cache_path):
            try:
                os.remove(self._cache_path)
            except OSError as e:
                print(f"Failed to delete cache file '{self._cache_path}': {e}")

    def complete_message(self):
        if len(self.messages) > 0 and self.messages[-1]["role"] == "assistant":
            try:
                last_input = self.messages[-1]["content"]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=512,
                    top_p=self.top_p,
                    extra_body={"top_k": self.top_k},
                )
                model_response = completion.choices[0].message.content

                model_response = f"{last_input.strip()}{model_response}"
                self.messages[-1] = {"role": "assistant", "content": model_response}
                self._save_history()

                return model_response
            except (APIError, BadRequestError, APIConnectionError) as e:
                print(f"API call failed: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
        else:
            return "Invalid history: last message is not from assistant."
