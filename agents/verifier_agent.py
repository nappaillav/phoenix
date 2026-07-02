"""
agents/verifier_agent.py
========================
Verifier Agent — validates proposed plans before execution.

Given a candidate Plan and the current environment state, the Verifier:
  1. Checks feasibility (can the robot physically execute each sub-goal?).
  2. Checks goal alignment (will the plan achieve the desired goal?).
  3. Checks consistency (are the sub-goals ordered correctly?).

Returns a ``VerificationResult`` with a pass/fail flag, reason, and an
optional corrected plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base_agent import BaseAgent
from .planner_agent import Plan


@dataclass
class VerificationResult:
    """Result from the Verifier Agent."""

    feasible: bool
    """True if the plan is deemed executable and goal-aligned."""

    reason: str
    """Explanation of the verdict."""

    corrected_plan: Plan | None = None
    """A corrected Plan suggested by the verifier, if ``feasible=False``."""

    confidence: float = 1.0
    """Confidence score in [0, 1]."""

    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓ PASS" if self.feasible else "✗ FAIL"
        return f"VerificationResult({status}, confidence={self.confidence:.2f}, reason={self.reason[:60]!r})"


class VerifierAgent(BaseAgent):
    """Validates plans for feasibility, consistency, and goal alignment.

    Parameters
    ----------
    env_wrapper:
        Optional ``PhoenixEnvWrapper`` instance. When provided, the Verifier
        can perform lightweight symbolic rollouts by saving/restoring state.
    """

    def __init__(self, env_wrapper: Any | None = None, **kwargs: Any) -> None:
        super().__init__(name="verifier", **kwargs)
        self.env_wrapper = env_wrapper

    def act(
        self,
        plan: Plan,
        obs_text: str,
        goal_text: str,
        task: str = "",
    ) -> VerificationResult:
        """Verify a proposed plan.

        Parameters
        ----------
        plan:
            The ``Plan`` to verify.
        obs_text:
            Current observation text.
        goal_text:
            Desired goal description.
        task:
            High-level task string for context.

        Returns
        -------
        VerificationResult
        """
        if plan.is_empty():
            return VerificationResult(
                feasible=False,
                reason="Plan is empty — no sub-goals provided.",
                confidence=1.0,
            )

        plan_text = "\n".join(
            f"  Step {i+1}: {sg}" for i, sg in enumerate(plan.subgoals)
        )

        system = (
            "You are a robotic plan verifier. "
            "Evaluate the proposed plan for feasibility, goal alignment, and consistency. "
            "Respond with a JSON object containing:\n"
            "  - 'feasible': boolean\n"
            "  - 'confidence': float in [0.0, 1.0]\n"
            "  - 'reason': string explaining the verdict\n"
            "  - 'corrected_steps': list of strings (corrected plan) if not feasible, else null\n"
            "Output only valid JSON."
        )

        user = self._fill_template(
            self._prompt_template or _DEFAULT_VERIFIER_PROMPT,
            task=task,
            plan=plan_text,
            observation=obs_text,
            goal=goal_text,
        )

        response, _ = self._call_llm(system=system, user=user)
        return self._parse_result(response, plan)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_result(self, response: str, original_plan: Plan) -> VerificationResult:
        """Parse the LLM JSON response into a VerificationResult."""
        # Extract JSON block (may be wrapped in markdown code fences)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                feasible = bool(data.get("feasible", True))
                confidence = float(data.get("confidence", 1.0))
                reason = str(data.get("reason", ""))
                corrected_steps = data.get("corrected_steps")

                corrected_plan = None
                if not feasible and corrected_steps:
                    corrected_plan = Plan(
                        subgoals=corrected_steps,
                        reasoning=f"Corrected by Verifier: {reason}",
                    )

                return VerificationResult(
                    feasible=feasible,
                    reason=reason,
                    corrected_plan=corrected_plan,
                    confidence=confidence,
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback: trust the plan if parsing fails
        return VerificationResult(
            feasible=True,
            reason="Verification response could not be parsed; defaulting to PASS.",
            confidence=0.5,
            metadata={"raw_response": response},
        )


_DEFAULT_VERIFIER_PROMPT = """\
Task: {task}

Desired goal: {goal}

Current state:
{observation}

Proposed plan:
{plan}

Evaluate whether this plan is feasible, consistent, and will achieve the goal.
Respond with a JSON object as specified.
"""
