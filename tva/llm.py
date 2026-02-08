import os
import base64
from abc import ABC, abstractmethod
from typing import List, Optional
from google import genai
from google.genai import types
from mimetypes import guess_type
from openai import OpenAI
from .tui import console


class LLM(ABC):
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_paths: Optional[List[str]] = None,
    ) -> str:
        pass


class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        media_paths: Optional[List[str]] = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]

        for path in media_paths or []:
            if not os.path.isfile(path):
                console.print(f"[yellow]Image file not found, skipping: '{path}'[/]")
                continue

            mime_type, _ = guess_type(path)
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type or 'image/jpeg'};base64,{base64_image}",
                    "detail": "high",
                },
            })

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content


class GeminiLLM(LLM):
    def __init__(
        self,
        model: str = "gemini-flash-lite-latest",
    ):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        video_paths: Optional[List[str]] = None,
    ) -> str:

        parts = []

        for path in video_paths or []:
            if not os.path.isfile(path):
                console.print(f"[yellow]Video file not found, skipping: '{path}'[/]")
                continue
            mime_type, _ = guess_type(path)
            with open(path, "rb") as f:
                video_data = f.read()
            parts.append(
                types.Part.from_bytes(data=video_data, mime_type=mime_type or "video/mp4")
            )

        parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            top_p=0.95,
            max_output_tokens=65535,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        )

        if self.model in ('gemini-3-flash-preview', 'gemini-2.5-flash'):
            thinking_config = types.ThinkingConfig(
                thinking_level="HIGH",
            )
            config.thinking_config = thinking_config

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text
