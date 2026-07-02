"""
planners/mcts_planner.py
========================
Monte Carlo Tree Search (MCTS) planner backend (ablation).

Implements UCB1-based MCTS over a symbolic state-action space.
The LLM is used as a policy/value network: it proposes candidate
sub-goals and scores them at leaf nodes.

This backend is selected via ``config.planner.backend: mcts``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.planner_agent import Plan


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: str
    """Text description of the current symbolic state."""

    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: str = ""
    """The sub-goal action that led to this node."""

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, exploration_c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.value / self.visits) + exploration_c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )

    def best_child(self, exploration_c: float) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration_c))

    def best_action_child(self) -> "MCTSNode":
        """Most-visited child (used for final action selection)."""
        return max(self.children, key=lambda c: c.visits)


class MCTSPlanner:
    """MCTS-based task planner.

    The LLM serves as both the expansion policy (proposes candidate
    sub-goals from a state) and the evaluation function (scores a
    terminal state's alignment with the goal).

    Parameters
    ----------
    cfg:
        The ``planner.mcts`` sub-dict from config:
        ``n_simulations``, ``exploration_constant``, ``max_depth``.
    """

    def __init__(self, cfg: dict) -> None:
        self.n_simulations = cfg.get("n_simulations", 50)
        self.exploration_c = cfg.get("exploration_constant", 1.41)
        self.max_depth = cfg.get("max_depth", 10)

    def plan(
        self,
        task: str,
        obs_text: str,
        goal_text: str,
        knowledge_context: str,
        **kwargs: Any,
    ) -> "Plan":
        """Run MCTS and return the best plan found.

        Note: In the current implementation the LLM expansion and scoring
        calls are routed through the parent ``PlannerAgent`` (passed via
        kwargs if available). If no agent is available, falls back to a
        random expansion policy for testing purposes.
        """
        from agents.planner_agent import Plan

        agent = kwargs.get("agent", None)
        root = MCTSNode(state=obs_text)

        for _ in range(self.n_simulations):
            node = self._select(root)
            child = self._expand(node, task, goal_text, knowledge_context, agent)
            reward = self._simulate(child, task, goal_text, agent)
            self._backpropagate(child, reward)

        # Extract best path from root
        subgoals = self._extract_plan(root)

        return Plan(
            subgoals=subgoals,
            reasoning=f"MCTS with {self.n_simulations} simulations, depth {self.max_depth}.",
            metadata={
                "backend": "mcts",
                "n_simulations": self.n_simulations,
                "root_visits": root.visits,
            },
        )

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node for expansion via UCB1."""
        depth = 0
        while not node.is_leaf() and depth < self.max_depth:
            node = node.best_child(self.exploration_c)
            depth += 1
        return node

    def _expand(
        self,
        node: MCTSNode,
        task: str,
        goal_text: str,
        knowledge_context: str,
        agent: Any,
    ) -> MCTSNode:
        """Expand a node by generating candidate sub-goals via LLM."""
        candidates = self._propose_subgoals(
            node.state, task, goal_text, knowledge_context, agent
        )
        for candidate in candidates[:3]:  # limit branching factor
            child = MCTSNode(state=candidate, parent=node, action=candidate)
            node.children.append(child)

        return random.choice(node.children) if node.children else node

    def _simulate(
        self, node: MCTSNode, task: str, goal_text: str, agent: Any
    ) -> float:
        """Rollout from node and return a reward in [0, 1]."""
        return self._score_state(node.state, goal_text, agent)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _propose_subgoals(
        self,
        state: str,
        task: str,
        goal_text: str,
        knowledge_context: str,
        agent: Any,
    ) -> list[str]:
        """Ask the LLM to propose candidate next sub-goals."""
        if agent is None:
            # Stub for testing
            return [f"stub_subgoal_{random.randint(1, 100)}"]

        system = (
            "You are a robotic planning expansion function. "
            "Given the current state and task, propose 3 concrete candidate "
            "next sub-goals as a numbered list. Be specific and concise."
        )
        user = (
            f"Task: {task}\nGoal: {goal_text}\n"
            f"Current state: {state[:400]}\n"
            f"Knowledge: {knowledge_context[:200]}\n\n"
            "List 3 candidate next sub-goals:"
        )
        response, _ = agent._call_llm(system=system, user=user)
        import re

        items = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", response, re.MULTILINE)
        return [i.strip() for i in items if i.strip()] or [response.strip()]

    def _score_state(self, state: str, goal_text: str, agent: Any) -> float:
        """Ask the LLM to score how close state is to the goal (0–1)."""
        if agent is None:
            return random.random()

        system = (
            "You are a robotic state evaluator. "
            "Rate how close the current state is to achieving the goal. "
            "Respond with only a float between 0.0 (far) and 1.0 (achieved)."
        )
        user = f"Goal: {goal_text}\nCurrent state: {state[:400]}\nScore (0.0–1.0):"
        response, _ = agent._call_llm(system=system, user=user)
        import re

        match = re.search(r"\d+\.?\d*", response)
        if match:
            val = float(match.group())
            return min(max(val, 0.0), 1.0)
        return 0.5

    # ------------------------------------------------------------------

    def _extract_plan(self, root: MCTSNode) -> list[str]:
        """Walk the most-visited path from root to extract the plan."""
        subgoals = []
        node = root
        depth = 0
        while node.children and depth < self.max_depth:
            node = node.best_action_child()
            if node.action:
                subgoals.append(node.action)
            depth += 1
        return subgoals
