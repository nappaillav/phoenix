"""
planners/llm_planner.py
=======================
LLM chain-of-thought planner backend (default).

Uses a structured prompt to elicit step-by-step task decomposition.
Returns a ``Plan`` with an ordered list of sub-goals.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.planner_agent import Plan, PlannerAgent


class LLMPlanner:
    """Chain-of-thought LLM-based task decomposer.

    This is the default planner backend.  It asks the LLM to reason
    about the task and output a numbered list of sub-goals.
    """

    def plan(
        self,
        task: str,
        obs_text: str,
        goal_text: str,
        knowledge_context: str,
        history: list[str],
        agent: "PlannerAgent",
    ) -> "Plan":
        """Generate a plan via LLM chain-of-thought.

        Parameters
        ----------
        task:
            High-level task description.
        obs_text:
            Current observation as text.
        goal_text:
            Desired end goal description.
        knowledge_context:
            Answer from the Knowledge Agent.
        history:
            Previous plan attempts (for iterative refinement).
        agent:
            The parent ``PlannerAgent`` (provides ``_call_llm``, ``_fill_template``).

        Returns
        -------
        Plan
        """
        from agents.planner_agent import Plan  # local import to avoid circular

        system = (
            "You are a hierarchical robotic manipulation planner. "
            "Decompose the given task into a short, ordered list of concrete sub-goals "
            "that a low-level controller can execute sequentially. "
            "Each sub-goal should be a single, specific action phrase. "
            "Think step-by-step, then output the sub-goals as a numbered list "
            "under the header 'PLAN:'."
        )

        history_text = ""
        if history:
            history_text = "\n\nPrevious planning attempts:\n" + "\n".join(history[-3:])

        user = agent._fill_template(
            agent._prompt_template or _DEFAULT_PLANNER_PROMPT,
            task=task,
            observation=obs_text,
            goal=goal_text,
            knowledge=knowledge_context,
            history=history_text,
        )

        response, _ = agent._call_llm(system=system, user=user)
        subgoals = self._parse_plan(response)

        return Plan(
            subgoals=subgoals,
            reasoning=response,
            metadata={"backend": "llm"},
        )

    # ------------------------------------------------------------------

    def _parse_plan(self, response: str) -> list[str]:
        """Extract numbered sub-goals from the LLM response."""
        # Find the PLAN: section
        plan_match = re.search(r"PLAN:(.*?)(?:\n\n|$)", response, re.DOTALL | re.IGNORECASE)
        section = plan_match.group(1) if plan_match else response

        # Extract numbered items: "1. ...", "1) ...", "- ..."
        items = re.findall(
            r"(?:^\s*(?:\d+[\.\)]\s*|-\s*))(.+)$",
            section,
            re.MULTILINE,
        )
        # Clean up
        subgoals = [item.strip() for item in items if item.strip()]
        return subgoals if subgoals else [response.strip()]


_DEFAULT_PLANNER_PROMPT = """\
Task: {task}

Desired goal: {goal}

Current observation:
{observation}

Knowledge context:
{knowledge}

{history}

Decompose the task into an ordered list of sub-goals.
Think step-by-step, then output sub-goals under 'PLAN:'.
"""
