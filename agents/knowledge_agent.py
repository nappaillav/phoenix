"""
agents/knowledge_agent.py
=========================
Knowledge Agent — maintains and retrieves procedural/semantic knowledge.

Stores task-procedure templates, object affordances, and relationship facts.
The Planner Agent queries this agent before generating sub-goals.

Knowledge is stored as a simple in-memory dict augmented by an optional
NetworkX graph for structured retrieval.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Built-in seed knowledge for Gymnasium-Robotics tasks
# ---------------------------------------------------------------------------
_SEED_KNOWLEDGE: dict[str, dict] = {
    "FetchPickAndPlace": {
        "procedure": [
            "Move gripper above the object.",
            "Lower gripper to grasp height.",
            "Close gripper to grasp the object.",
            "Lift the object to transport height.",
            "Move the object above the goal position.",
            "Lower the object to the goal position.",
            "Open gripper to release.",
        ],
        "affordances": {
            "block": ["graspable", "liftable", "placeable"],
            "gripper": ["open", "close", "translate"],
        },
    },
    "FetchPush": {
        "procedure": [
            "Align gripper behind the object relative to goal.",
            "Move gripper into contact with the object.",
            "Push the object toward the goal.",
        ],
        "affordances": {
            "block": ["pushable"],
            "gripper": ["translate"],
        },
    },
    "FetchSlide": {
        "procedure": [
            "Align gripper behind the puck.",
            "Strike the puck with appropriate velocity toward the goal.",
        ],
        "affordances": {
            "puck": ["slideable"],
        },
    },
    "FetchReach": {
        "procedure": [
            "Move the gripper end-effector to the target position.",
        ],
        "affordances": {
            "gripper": ["translate"],
        },
    },
    "PointMaze": {
        "procedure": [
            "Navigate from current position to goal while avoiding walls.",
        ],
        "affordances": {
            "agent": ["move_forward", "move_backward", "turn"],
        },
    },
}


class KnowledgeAgent(BaseAgent):
    """Maintains task knowledge and responds to queries from the Planner.

    Parameters
    ----------
    env_id:
        The Gymnasium-Robotics environment ID.  Used to look up seed knowledge.
    """

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(name="knowledge", **kwargs)
        self.env_id = env_id

        # Load seed knowledge
        self._store: dict = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._init_seed_knowledge(env_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, query: str, obs_text: str = "") -> str:
        """Answer a natural-language knowledge query.

        Retrieves relevant facts from the store and optionally calls the LLM
        to synthesize a structured answer.

        Parameters
        ----------
        query:
            Natural-language question from the Planner Agent.
        obs_text:
            Current observation text for context-sensitive answers.

        Returns
        -------
        str
            Knowledge answer string.
        """
        local_facts = self._retrieve_local(query)

        system = (
            "You are a robotic task knowledge base. "
            "Answer questions about object affordances, task procedures, and "
            "action feasibility concisely and factually. "
            "Only state what you know; do not hallucinate properties."
        )

        user = self._fill_template(
            self._prompt_template or _DEFAULT_KNOWLEDGE_PROMPT,
            query=query,
            local_facts=local_facts,
            observation=obs_text,
        )

        response, _ = self._call_llm(system=system, user=user)
        return response

    def add_fact(self, subject: str, relation: str, obj: str) -> None:
        """Add a (subject, relation, object) triple to the knowledge graph."""
        self._graph.add_edge(subject, obj, relation=relation)
        self._store.setdefault(subject, []).append(f"{subject} --[{relation}]--> {obj}")

    def get_procedure(self, task_key: str) -> list[str]:
        """Return the ordered procedure steps for a known task."""
        return _SEED_KNOWLEDGE.get(task_key, {}).get("procedure", [])

    def get_affordances(self, object_name: str) -> list[str]:
        """Return affordances for a given object type."""
        for task_data in _SEED_KNOWLEDGE.values():
            affs = task_data.get("affordances", {})
            if object_name in affs:
                return affs[object_name]
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_seed_knowledge(self, env_id: str) -> None:
        """Populate the store from seed knowledge matching the env_id."""
        for key, data in _SEED_KNOWLEDGE.items():
            if key.lower() in env_id.lower():
                self._store["procedure"] = data.get("procedure", [])
                for obj, affs in data.get("affordances", {}).items():
                    for aff in affs:
                        self.add_fact(obj, "has_affordance", aff)
                break

    def _retrieve_local(self, query: str) -> str:
        """Return a text summary of locally stored knowledge relevant to query."""
        lines = []
        q_lower = query.lower()
        for key, facts in self._store.items():
            if isinstance(facts, list):
                for fact in facts:
                    if any(word in fact.lower() for word in q_lower.split()):
                        lines.append(f"- {fact}")
            else:
                if any(word in str(facts).lower() for word in q_lower.split()):
                    lines.append(f"- {key}: {facts}")
        return "\n".join(lines) if lines else "No local facts found."


_DEFAULT_KNOWLEDGE_PROMPT = """\
Query: {query}

Current observation context:
{observation}

Local knowledge base:
{local_facts}

Based on the above, provide a concise, factual answer to the query.
"""
