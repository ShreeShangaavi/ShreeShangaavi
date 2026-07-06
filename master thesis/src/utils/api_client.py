from __future__ import annotations

import logging
import time

import openai

from src.utils.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


def make_client(cfg: PipelineConfig) -> openai.OpenAI:
    """Build an OpenAI client pointed at the KISSKI endpoint."""
    return openai.OpenAI(base_url=cfg.api.base_url, api_key=cfg.api_key)


def chat_completion(
    client: openai.OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Retries on rate limits (429) and server errors (5xx) with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(f"Empty response content from model '{model}'")
            return content
        except openai.RateLimitError:
            wait = 2**attempt
            logger.warning(
                "Rate limit hit (attempt %d/%d), retrying in %ds",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                wait = 2**attempt
                logger.warning(
                    "Server error %d (attempt %d/%d), retrying in %ds",
                    e.status_code, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise  # 4xx client errors are not retried
    raise RuntimeError(f"KISSKI API call failed after {max_retries} retries")
