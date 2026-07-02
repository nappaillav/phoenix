"""
evaluation/baselines.py
=======================
Stub interfaces for baseline methods from the proposal.

Baselines:
  - VoxPoser  (language-guided value map manipulation)
  - RVT       (Robotic View Transformer)
  - MA-11     (multi-agent baseline)
  - DirectLLM (direct LLM planning without structured framework)

Each baseline implements the same ``act(obs, task) -> action`` interface
so they can be swapped in via config for fair comparison.

To add a real baseline implementation, subclass ``BaselineAgent`` and
override ``act()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaselineAgent(ABC):
    """Abstract baseline agent interface."""

    name: str = "baseline"

    @abstractmethod
    def act(self, obs: Any, task: str) -> np.ndarray:
        """Return an action given an observation and task string."""
        ...

    def reset(self) -> None:
        pass


class VoxPoserBaseline(BaselineAgent):
    """Stub for VoxPoser (Huang et al., 2023).

    VoxPoser composes language model output with 3D voxel value maps for
    zero-shot robotic manipulation. Replace the ``act()`` body with a
    real VoxPoser call when integrating the full system.
    """

    name = "VoxPoser"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def act(self, obs: Any, task: str) -> np.ndarray:
        # TODO: integrate real VoxPoser model
        return self.action_space.sample()


class RVTBaseline(BaselineAgent):
    """Stub for RVT (Goyal et al., 2023).

    RVT uses a multi-view transformer for robotic manipulation.
    Replace ``act()`` with actual RVT inference when integrating.
    """

    name = "RVT"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def act(self, obs: Any, task: str) -> np.ndarray:
        # TODO: integrate real RVT model
        return self.action_space.sample()


class MA11Baseline(BaselineAgent):
    """Stub for MA-11 multi-agent baseline.

    Replace ``act()`` with actual MA-11 inference when integrating.
    """

    name = "MA-11"

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def act(self, obs: Any, task: str) -> np.ndarray:
        # TODO: integrate real MA-11 model
        return self.action_space.sample()


class DirectLLMBaseline(BaselineAgent):
    """Direct LLM planning baseline — no structured Planner/Verifier.

    Asks the LLM for a direct action given the observation. Useful as
    the ablation control against the full Phoenix framework.
    """

    name = "DirectLLM"

    def __init__(self, action_space: Any, llm_cfg: dict, dry_run: bool = False) -> None:
        self.action_space = action_space
        self.llm_cfg = llm_cfg
        self.dry_run = dry_run
        self._action_dim = int(np.prod(action_space.shape))

    def act(self, obs: Any, task: str) -> np.ndarray:
        if self.dry_run:
            return np.zeros(self._action_dim, dtype=np.float32)

        # Build a minimal LLM call
        import os
        import re
        import json

        backend = self.llm_cfg.get("backend", "openai")
        model = self.llm_cfg.get("model", "gpt-4o")
        prompt = (
            f"Task: {task}\nObservation: {obs}\n"
            f"Output a {self._action_dim}-dim continuous action as JSON array. "
            "Example: ACTION: [0.1, -0.2, 0.5, 0.0]"
        )

        if backend in ("openai", "ollama"):
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self.llm_cfg.get("base_url"),
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
        else:
            text = ""

        match = re.search(r"ACTION:\s*(\[.*?\])", text, re.DOTALL)
        if match:
            try:
                vals = json.loads(match.group(1))
                action = np.array(vals, dtype=np.float32)
                return np.clip(action, self.action_space.low, self.action_space.high)
            except Exception:
                pass
        return np.zeros(self._action_dim, dtype=np.float32)


# Registry for config-driven baseline selection
BASELINE_REGISTRY: dict[str, type[BaselineAgent]] = {
    "voxposer": VoxPoserBaseline,
    "rvt": RVTBaseline,
    "ma11": MA11Baseline,
    "direct_llm": DirectLLMBaseline,
}


def build_baseline(name: str, **kwargs: Any) -> BaselineAgent:
    """Instantiate a baseline by name.

    Parameters
    ----------
    name:
        One of: ``"voxposer"``, ``"rvt"``, ``"ma11"``, ``"direct_llm"``.
    kwargs:
        Forwarded to the baseline's ``__init__``.
    """
    cls = BASELINE_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown baseline: {name!r}. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return cls(**kwargs)
