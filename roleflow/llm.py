from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import APIError, OpenAI

from .config import ModelConfig

EXECUTOR_TEMPERATURE = 0.0
TIMEOUT = 1800.0
RETRIES = 1


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})


def with_scheme(address: str, scheme: str = "") -> str:
    if not address or "://" in address:
        return address
    scheme = scheme or ("http" if address.startswith(("localhost", "127.")) else "https")
    return f"{scheme}://{address}"


class LanguageModel:
    def __init__(self, config: ModelConfig, role: str = "executor"):
        self.config = config
        self.role = role
        self.client = OpenAI(
            base_url=with_scheme(config.api_base or "api-base-not-set"),
            api_key=config.api_key or "EMPTY",
            timeout=TIMEOUT,
            max_retries=RETRIES,
        )

    def reply(self, messages: list, temperature: float = EXECUTOR_TEMPERATURE, max_tokens=None) -> Reply:
        request = {"model": self.config.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**request)
        usage = response.usage
        return Reply(
            (response.choices[0].message.content or "").strip(),
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        )

    def address(self) -> str:
        return self.config.api_base or "(api_base not set)"


def preflight(model: LanguageModel) -> None:
    if not model.config.api_base:
        raise SystemExit(f"the {model.role} endpoint {model.address()} (model {model.config.model}): set {model.role}.api_base in the config")
    try:
        model.reply([{"role": "user", "content": "ping"}], max_tokens=16)
    except APIError as error:
        raise SystemExit(f"the {model.role} endpoint {model.address()} (model {model.config.model}) failed a test request: {error}") from None


def parse_json(text: str):
    decoder = json.JSONDecoder(strict=False)
    for start, character in enumerate(text or ""):
        if character == "{":
            try:
                return decoder.raw_decode(text, start)[0]
            except json.JSONDecodeError:
                continue
    return None


def structured_reply(model: LanguageModel, caller: str, messages: list, keys: tuple) -> dict:
    reply = model.reply(messages)
    data = parse_json(reply.text)
    if not isinstance(data, dict) or any(key not in data for key in keys):
        raise RuntimeError(
            f"the {caller} reply from {model.address()} (model {model.config.model}) is not a JSON object with the keys {list(keys)}: {reply.text[:300]!r}"
        )
    return data


def flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False
    raise RuntimeError(f"expected a true or false value, got {value!r}")
