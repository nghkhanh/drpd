from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from drpd.config import app_config


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make a standard call to the LLM"""
        pass

    @abstractmethod
    def call_structured(
        self, prompt: str, json_schema: dict[str, Any], **kwargs
    ) -> str:
        """Make a call to the LLM and retrieve a structured response."""
        pass

    # @abstractmethod
    # def call_with_resoning(self, prompt: str, **kwarg) -> tuple[str, dict]:
    #     """Call LLM API with reasoning parameters."""
    #     pass


class LiteLLMClient(LLMClient):
    """LiteLLM client implementation"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        # TODO(khanhnh): support local LLM backend  # noqa: TD003
        # Cơ chế gọi lại api nếu nó bị lỗi.
        # self.retry_decorator = retry

    def call(
        self,
        prompt: str,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        """Make a standard call to LiteLLM"""
        completion = self.client.chat.completions.create(
            model=model_name,
            temperature=temperature or app_config["llm"]["temperature"],
            max_tokens=max_tokens or app_config["llm"]["max_tokens_output"],
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return completion.choices[0].messages.content.strip()

    def call_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_message: str = None,
        **kwargs,
    ):
        """Make a structure call to LiteLLM"""
        completion = self.client.chat.completions.create(
            model=model_name,
            temperature=temperature or app_config["llm"]["temperature"],
            max_tokens=max_tokens or app_config["llm"]["max_tokens_output"],
            messages=[
                {
                    "role": "developer",  # or system, depending on modelOpenAI uses developer for o1/o3
                    "content": system_message or app_config["llm"]["system_message"],
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": json_schema,
                },
            },
            **kwargs,
        )
        return completion.choices[0].messages.content.strip()
