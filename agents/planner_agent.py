"""
agents/planner_agent.py
=======================
Planner Agent — decomposes tasks into ordered sub-goal sequences.

Delegates to one of three planner backends (controlled by config):
  - ``llm``   : LLM chain-of-thought decomposition (default).
  - ``mcts``  : Monte Carlo Tree Search over symbolic states.
  - ``graph`` : Knowledge-graph-based search (HP-KG inspired).

Returns a ``Plan`` dataclass with a list of sub-goal strings and metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base_agent import BaseAgent


@dataclass
class Plan:
    """A structured plan produced by the Planner Agent."""

    subgoals: list[str]
    """Ordered list of sub-goal strings."""

    reasoning: str = ""
    """LLM chain-of-thought or search trace."""

    metadata: dict = field(default_factory=dict)
    """Backend-specific metadata (e.g., MCTS statistics, graph path)."""

    def is_empty(self) -> bool:
        return len(self.subgoals) == 0

    def __len__(self) -> int:
        return len(self.subgoals)

    def __repr__(self) -> str:
        goals = " → ".join(self.subgoals[:3])
        if len(self.subgoals) > 3:
            goals += " → ..."
        return f"Plan([{goals}], n={len(self.subgoals)})"


class PlannerAgent(BaseAgent):
    """Decomposes manipulation tasks into sub-goal sequences.

    Parameters
    ----------
    planner_cfg:
        The ``planner`` sub-dict from the loaded YAML config.
    knowledge_agent:
        Optional ``KnowledgeAgent`` instance for context-enriched planning.
    """

    def __init__(
        self,
        planner_cfg: dict,
        knowledge_agent: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="planner", **kwargs)
        self.planner_cfg = planner_cfg
        self.knowledge_agent = knowledge_agent
        self._backend = self._build_backend()

    def act(
        self,
        task: str,
        obs_text: str,
        goal_text: str = "",
        history: list[str] | None = None,
    ) -> Plan:
        """Decompose the task into a sub-goal plan.

        Parameters
        ----------
        task:
            High-level task description string.
        obs_text:
            Current observation text.
        goal_text:
            Desired goal description (from env info dict if available).
        history:
            Previous planning attempts for context.

        Returns
        -------
        Plan
            Structured plan with ordered sub-goals.
        """
        # Fetch knowledge context if available
        knowledge_context = ""
        if self.knowledge_agent is not None:
            knowledge_context = self.knowledge_agent.act(
                query=f"What are the steps to accomplish: {task}",
                obs_text=obs_text,
            )

        backend_name = self.planner_cfg.get("backend", "llm")

        if backend_name == "llm":
            return self._backend.plan(
                task=task,
                obs_text=obs_text,
                goal_text=goal_text,
                knowledge_context=knowledge_context,
                history=history or [],
                agent=self,
            )
        elif backend_name in ("mcts", "graph"):
            return self._backend.plan(
                task=task,
                obs_text=obs_text,
                goal_text=goal_text,
                knowledge_context=knowledge_context,
            )
        else:
            raise ValueError(f"Unknown planner backend: {backend_name!r}")

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    def _build_backend(self) -> Any:
        backend_name = self.planner_cfg.get("backend", "llm")
        if backend_name == "llm":
            from planners.llm_planner import LLMPlanner

            return LLMPlanner()
        elif backend_name == "mcts":
            from planners.mcts_planner import MCTSPlanner

            return MCTSPlanner(cfg=self.planner_cfg.get("mcts", {}))
        elif backend_name == "graph":
            from planners.graph_planner import GraphPlanner

            return GraphPlanner(cfg=self.planner_cfg.get("graph", {}))
        else:
            raise ValueError(f"Unknown planner backend: {backend_name!r}")
