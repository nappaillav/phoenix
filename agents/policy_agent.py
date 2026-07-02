"""
agents/policy_agent.py
======================
Policy Agent — translates high-level sub-goals into environment actions.

The Policy Agent receives:
  - A sub-goal string from the Planner Agent.
  - A text description of the current observation.
  - The action space description.

It returns a numpy action vector by parsing the LLM response.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from .base_agent import BaseAgent


class PolicyAgent(BaseAgent):
    """Translates Planner sub-goals into low-level environment actions.

    Parameters
    ----------
    action_space:
        The environment's ``gymnasium.Space`` action space, used to
        describe the action format in the prompt and to validate/clip outputs.
    """

    def __init__(self, action_space: Any, **kwargs: Any) -> None:
        super().__init__(name="policy", **kwargs)
        self.action_space = action_space
        self._action_dim = int(np.prod(action_space.shape))

    def act(
        self,
        subgoal: str,
        obs_text: str,
        history: list[str] | None = None,
    ) -> tuple[np.ndarray, str]:
        """Generate a low-level action for the given sub-goal.

        Parameters
        ----------
        subgoal:
            Current sub-goal string from the Planner.
        obs_text:
            Text-serialized observation from ``PhoenixEnvWrapper``.
        history:
            Optional list of recent (subgoal, action) strings for context.

        Returns
        -------
        action : np.ndarray
            Clipped action vector within the action space bounds.
        reasoning : str
            The LLM's step-by-step reasoning text.
        """
        system = (
            "You are a low-level robotic manipulation controller. "
            "Given the current observation and a sub-goal, output a continuous "
            "action vector as a JSON array of floats. "
            f"The action must have exactly {self._action_dim} dimensions. "
            f"Each value should be in range [{self.action_space.low.min():.2f}, "
            f"{self.action_space.high.max():.2f}]. "
            "First reason step-by-step, then output the action on a line starting with 'ACTION:'."
        )

        history_text = ""
        if history:
            history_text = "\n\nRecent history:\n" + "\n".join(history[-5:])

        user = self._fill_template(
            self._prompt_template or _DEFAULT_POLICY_PROMPT,
            subgoal=subgoal,
            observation=obs_text,
            action_dim=str(self._action_dim),
            action_low=str(round(float(self.action_space.low.min()), 3)),
            action_high=str(round(float(self.action_space.high.max()), 3)),
            history=history_text,
        )

        response, _ = self._call_llm(system=system, user=user)
        action = self._parse_action(response)
        return action, response

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_action(self, response: str) -> np.ndarray:
        """Extract a JSON float array from the LLM response."""
        # Try to find ACTION: [...] pattern
        match = re.search(r"ACTION:\s*(\[.*?\])", response, re.DOTALL)
        if match:
            try:
                values = json.loads(match.group(1))
                action = np.array(values, dtype=np.float32)
                return np.clip(action, self.action_space.low, self.action_space.high)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: zero action
        return np.zeros(self._action_dim, dtype=np.float32)


_DEFAULT_POLICY_PROMPT = """\
Current sub-goal: {subgoal}

{observation}

{history}

Action space: {action_dim} continuous dimensions in [{action_low}, {action_high}].

Think step-by-step about what motion will make progress toward the sub-goal.
Then output the action vector on a line starting with 'ACTION:'.
Example: ACTION: [0.1, -0.2, 0.5, 0.0]
"""
