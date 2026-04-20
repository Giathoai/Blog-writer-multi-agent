"""
LLM Configuration Module
Supports: HuggingFace (default with Qwen3-1.7B) and Google AI Studio (Gemini)
Switch providers via LLM_PROVIDER env variable.
"""

from __future__ import annotations

import os
import ssl
import warnings
import requests as req_lib

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

load_dotenv()

# Disable SSL warnings for networks with proxy/firewall issues
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
req_lib.packages.urllib3.disable_warnings()


class HuggingFaceDirectChat(BaseChatModel):
    """
    Custom ChatModel that calls HuggingFace Inference API via direct HTTP.
    Bypasses SSL issues by using verify=False when needed.
    """
    model_id: str = "Qwen/Qwen3-1.7B"
    api_token: str = ""
    temperature: float = 0.7
    max_new_tokens: int = 4096

    @property
    def _llm_type(self) -> str:
        return "huggingface-direct"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages to API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
                api_messages.append({"role": "user", "content": msg.content})

        # Build request payload
        payload = {
            "model": self.model_id,
            "messages": api_messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        # HuggingFace Inference Providers - try multiple providers
        # Qwen3-1.7B is hosted by featherless-ai and nebius, NOT hf-inference
        endpoints = [
            "https://router.huggingface.co/featherless-ai/v1/chat/completions",
            "https://router.huggingface.co/nebius/v1/chat/completions",
            f"https://api-inference.huggingface.co/models/{self.model_id}/v1/chat/completions",
        ]

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        last_error = None
        for url in endpoints:
            try:
                try:
                    response = req_lib.post(url, json=payload, headers=headers, timeout=180, verify=True)
                except (req_lib.exceptions.SSLError, req_lib.exceptions.ConnectionError):
                    response = req_lib.post(url, json=payload, headers=headers, timeout=180, verify=False)

                if response.status_code == 200:
                    break
                # If 400/404, try next endpoint
                if response.status_code in (400, 404, 422):
                    last_error = response.text
                    continue
                # Other errors, raise immediately
                last_error = response.text
            except Exception as e:
                last_error = str(e)
                continue
        else:
            raise RuntimeError(f"All HuggingFace endpoints failed. Last error: {last_error}")

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                pass
            raise RuntimeError(
                f"HuggingFace API error ({response.status_code}): {error_detail}"
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def get_llm():
    """
    Factory function to create the appropriate LLM based on configuration.

    LLM_PROVIDER options:
        - "huggingface" (default): Uses Qwen/Qwen3-1.7B via HuggingFace Inference API
        - "google": Uses Gemini via Google AI Studio API
    """
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower()

    if provider == "google":
        return _build_google_llm()
    else:
        return _build_huggingface_llm()


def _build_huggingface_llm():
    """Build HuggingFace LLM using direct HTTP calls (default: Qwen3-1.7B)."""
    hf_token = os.getenv("HF_TOKEN", "")

    if not hf_token or hf_token == "your_huggingface_token_here":
        raise RuntimeError(
            "HF_TOKEN environment variable is required!\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
            "Then set it in server/.env file."
        )

    model_id = os.getenv("HF_MODEL", "Qwen/Qwen3-1.7B")

    return HuggingFaceDirectChat(
        model_id=model_id,
        api_token=hf_token,
        temperature=0.7,
        max_new_tokens=4096,
    )


def _build_google_llm():
    """Build Google Gemini LLM via AI Studio API (ready for expansion)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key or api_key == "your_google_ai_studio_api_key_here":
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required!\n"
            "Get your key at: https://aistudio.google.com/apikey\n"
            "Then set it in server/.env file."
        )

    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=4096,
        convert_system_message_to_human=True,
    )

    return llm


def get_llm_info() -> dict:
    """Return current LLM configuration info."""
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower()

    if provider == "google":
        return {
            "provider": "Google AI Studio",
            "model": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        }
    else:
        return {
            "provider": "HuggingFace",
            "model": os.getenv("HF_MODEL", "Qwen/Qwen3-1.7B"),
        }
