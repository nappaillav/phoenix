"""
planners/graph_planner.py
=========================
Knowledge-graph-based planner backend (ablation).

Inspired by HP-KG (Hierarchical Procedural Knowledge Graphs).
Builds a directed NetworkX graph of (state → sub-goal → state) triples
and uses shortest-path / beam search to find a plan.

When a pre-built graph is provided (via ``config.planner.graph.knowledge_graph_path``),
it is loaded from a pickle file.  Otherwise, the graph is grown on-the-fly
using LLM-generated (state, action, next-state) triples.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from agents.planner_agent import Plan


class GraphPlanner:
    """Knowledge-graph search planner.

    Parameters
    ----------
    cfg:
        The ``planner.graph`` sub-dict from config.
        ``knowledge_graph_path``: path to a pre-built NetworkX DiGraph pickle.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._graph: nx.DiGraph = self._load_graph(cfg.get("knowledge_graph_path"))

    def plan(
        self,
        task: str,
        obs_text: str,
        goal_text: str,
        knowledge_context: str,
        **kwargs: Any,
    ) -> "Plan":
        """Find a plan via graph search.

        If the graph already contains a path from the current state to the
        goal, returns it directly.  Otherwise, expands the graph via LLM
        and re-searches.
        """
        from agents.planner_agent import Plan

        agent = kwargs.get("agent", None)

        # Normalize node labels
        src = self._normalize(obs_text)
        tgt = self._normalize(goal_text)

        # Ensure source and target exist
        self._graph.add_node(src, type="state")
        self._graph.add_node(tgt, type="goal")

        # Try shortest path
        if nx.has_path(self._graph, src, tgt):
            path = nx.shortest_path(self._graph, src, tgt)
            subgoals = self._path_to_subgoals(path)
            reasoning = f"Graph search found path of length {len(path)} via {len(self._graph.nodes)} nodes."
        else:
            # Expand graph with LLM-generated triples, then search again
            subgoals = self._llm_expand_and_search(
                task, obs_text, goal_text, knowledge_context, agent, src, tgt
            )
            reasoning = "Graph expanded via LLM; beam-searched for plan."

        return Plan(
            subgoals=subgoals,
            reasoning=reasoning,
            metadata={
                "backend": "graph",
                "graph_nodes": self._graph.number_of_nodes(),
                "graph_edges": self._graph.number_of_edges(),
            },
        )

    def save_graph(self, path: str) -> None:
        """Serialize the current graph to a pickle file for reuse."""
        with open(path, "wb") as f:
            pickle.dump(self._graph, f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_graph(self, path: str | None) -> nx.DiGraph:
        if path and Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def _normalize(self, text: str) -> str:
        """Truncate and clean text for use as a node label."""
        return text.strip()[:80].replace("\n", " ")

    def _path_to_subgoals(self, path: list[str]) -> list[str]:
        """Convert a graph path to sub-goal strings via edge labels."""
        subgoals = []
        for u, v in zip(path[:-1], path[1:]):
            edge_data = self._graph.get_edge_data(u, v, default={})
            action = edge_data.get("action", v)
            subgoals.append(action)
        return subgoals

    def _llm_expand_and_search(
        self,
        task: str,
        obs_text: str,
        goal_text: str,
        knowledge_context: str,
        agent: Any,
        src: str,
        tgt: str,
    ) -> list[str]:
        """Use the LLM to generate graph triples and find a path."""
        if agent is None:
            # Stub: direct connection
            self._graph.add_edge(src, tgt, action="achieve goal")
            return ["achieve goal"]

        system = (
            "You are a robotic knowledge graph builder. "
            "Given a task, current state, and goal, generate up to 5 "
            "(state, action, next_state) triples that form a path from "
            "current state to goal. "
            "Output each triple on a new line as: STATE | ACTION | NEXT_STATE"
        )
        user = (
            f"Task: {task}\nCurrent state: {obs_text[:300]}\n"
            f"Goal: {goal_text}\nKnowledge: {knowledge_context[:200]}\n\n"
            "Generate triples:"
        )
        response, _ = agent._call_llm(system=system, user=user)
        self._parse_triples(response, src)

        # Search again after expansion
        if nx.has_path(self._graph, src, tgt):
            path = nx.shortest_path(self._graph, src, tgt)
            return self._path_to_subgoals(path)

        # Last resort: flatten LLM output as sub-goals
        import re

        actions = re.findall(r"\|\s*(.+?)\s*\|", response)
        return [a.strip() for a in actions] if actions else [response.strip()]

    def _parse_triples(self, response: str, src: str) -> None:
        """Parse LLM-generated triples into the graph."""
        import re

        prev_state = src
        for line in response.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                state, action, next_state = parts
                state = self._normalize(state or prev_state)
                next_state = self._normalize(next_state)
                self._graph.add_node(state, type="state")
                self._graph.add_node(next_state, type="state")
                self._graph.add_edge(state, next_state, action=action)
                prev_state = next_state
