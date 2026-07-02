"""
agents/base_agent.py
====================
Abstract base class for all Phoenix agents.

All agents share:
  - LLM backend instantiation (OpenAI / Anthropic / Ollama).
  - Prompt template loading from prompts/*.md.
  - Token usage tracking forwarded to MetricsCollector.
  - Dry-run support (returns stub responses without LLM calls).
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metrics.collector import MetricsCollector

# Root of the phoenix project (agents/ is one level down)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BaseAgent(ABC):
    """Abstract base for all Phoenix agents.

    Parameters
    ----------
    name:
        Human-readable agent name (e.g. ``"planner"``).
    llm_cfg:
        The ``llm`` sub-dict from the loaded YAML config.
    metrics:
        A ``MetricsCollector`` instance for token/cost tracking.
        Pass ``None`` to disable tracking (e.g., in tests).
    dry_run:
        If ``True``, skip real LLM calls and return placeholder text.
    """

    def __init__(
        self,
        name: str,
        llm_cfg: dict,
        metrics: "MetricsCollector | None" = None,
        dry_run: bool = False,
    ) -> None:
        self.name = name
        self.llm_cfg = llm_cfg
        self.metrics = metrics
        self.dry_run = dry_run

        self._client = self._build_client()
        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, **kwargs: Any) -> Any:
        """Execute the agent's main action and return a structured result."""
        ...

    def reset(self) -> None:
        """Reset any per-episode state (override if needed)."""
        pass

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        """Instantiate the appropriate LLM client from config."""
        backend = self.llm_cfg.get("backend", "openai")
        if self.dry_run:
            return None

        if backend == "openai":
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        elif backend == "anthropic":
            from anthropic import Anthropic  # type: ignore

            return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        elif backend == "ollama":
            from openai import OpenAI  # type: ignore

            base_url = self.llm_cfg.get("base_url", "http://localhost:11434/v1")
            return OpenAI(base_url=base_url, api_key="ollama")

        else:
            raise ValueError(f"Unknown LLM backend: {backend!r}")

    def _call_llm(self, system: str, user: str) -> tuple[str, dict]:
        """Call the LLM and return (response_text, usage_dict).

        usage_dict keys: prompt_tokens, completion_tokens, total_tokens.
        """
        if self.dry_run:
            return f"[DRY RUN — {self.name} stub response]", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        backend = self.llm_cfg.get("backend", "openai")
        model = self.llm_cfg.get("model", "gpt-4o")
        temperature = self.llm_cfg.get("temperature", 0.0)
        max_tokens = self.llm_cfg.get("max_tokens", 1024)

        t0 = time.perf_counter()

        if backend in ("openai", "ollama"):
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        elif backend == "anthropic":
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        elapsed = time.perf_counter() - t0
        usage["latency_s"] = round(elapsed, 3)

        # Forward to metrics collector
        if self.metrics is not None:
            self.metrics.record_llm_call(
                agent=self.name,
                model=model,
                usage=usage,
            )

        return text, usage

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    def _load_prompt_template(self) -> str:
        """Load the prompt template from prompts/<name>_prompt.md."""
        template_path = _PROJECT_ROOT / "prompts" / f"{self.name}_prompt.md"
        if template_path.exists():
            return template_path.read_text()
        return ""

    def _fill_template(self, template: str, **kwargs: str) -> str:
        """Replace ``{placeholder}`` tokens in the template."""
        result = template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, backend={self.llm_cfg.get('backend')})"
