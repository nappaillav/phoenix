"""
metrics/collector.py
====================
MetricsCollector — tracks all research metrics from the proposal.

Tracked metrics (per episode and aggregate):
  - Task success rate          (RQ1, RQ3)
  - Plan accuracy              (RQ1)
  - Verifier accuracy          (RQ4)
  - Number of planning steps   (RQ3)
  - Sample efficiency          (RQ2)
  - Token efficiency           (RQ2)
  - Inference cost             (RQ3)
  - Generalization performance (RQ1)

Saves results to:
  - results/{run_id}.json         (full episode traces)
  - results/{run_id}_summary.csv  (aggregate summary)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Per-LLM-call pricing (USD per 1K tokens) — update as needed
# ---------------------------------------------------------------------------
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
    "default": {"prompt": 0.001, "completion": 0.002},
}


@dataclass
class EpisodeRecord:
    """Metrics for a single episode."""

    episode_id: int
    task: str
    env_id: str
    success: bool = False

    # Planning
    n_planning_iterations: int = 0   # Planner→Verifier cycles
    n_subgoals_planned: int = 0
    n_subgoals_executed: int = 0
    plan_was_verified: bool = False
    verifier_accepted: bool = False
    verifier_corrected: bool = False

    # Steps & samples
    n_env_steps: int = 0             # total env.step() calls
    n_env_steps_to_success: int = 0

    # Tokens & cost
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Timing
    wall_time_s: float = 0.0

    # Generalization flag
    is_ood: bool = False

    # Per-agent token breakdown
    agent_tokens: dict = field(default_factory=dict)

    # Full trace (optional)
    trace: list[dict] = field(default_factory=list)


@dataclass
class RunSummary:
    """Aggregate metrics across all episodes in a run."""

    run_id: str
    experiment_name: str
    env_id: str
    n_episodes: int = 0

    # Success
    n_success: int = 0
    task_success_rate: float = 0.0

    # Planning
    avg_planning_iterations: float = 0.0
    avg_subgoals: float = 0.0
    plan_accuracy: float = 0.0        # % episodes where verifier accepted first plan
    verifier_accuracy: float = 0.0    # % episodes where verifier decision was correct

    # Efficiency
    avg_env_steps: float = 0.0
    sample_efficiency: float = 0.0    # success_rate / avg_env_steps
    avg_total_tokens: float = 0.0
    token_efficiency: float = 0.0     # success_rate / avg_tokens * 1000
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0

    # Timing
    avg_wall_time_s: float = 0.0
    total_wall_time_s: float = 0.0

    # Generalization
    ood_success_rate: float = 0.0
    iid_success_rate: float = 0.0


class MetricsCollector:
    """Collects, aggregates, and persists experiment metrics.

    Parameters
    ----------
    run_id:
        Unique identifier for this experiment run (used in filenames).
    experiment_name:
        Human-readable experiment name from config.
    env_id:
        Gymnasium-Robotics environment ID.
    results_dir:
        Directory where JSON + CSV files will be saved.
    track_tokens:
        Whether to track LLM token usage.
    track_cost:
        Whether to estimate USD cost from token counts.
    track_timing:
        Whether to record wall-clock times.
    save_traces:
        Whether to save full per-step traces in the JSON output.
    """

    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        env_id: str,
        results_dir: str = "results",
        track_tokens: bool = True,
        track_cost: bool = True,
        track_timing: bool = True,
        save_traces: bool = True,
    ) -> None:
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.env_id = env_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.track_tokens = track_tokens
        self.track_cost = track_cost
        self.track_timing = track_timing
        self.save_traces = save_traces

        self._episodes: list[EpisodeRecord] = []
        self._current: EpisodeRecord | None = None
        self._episode_start: float = 0.0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self, episode_id: int, task: str, is_ood: bool = False) -> None:
        """Start tracking a new episode."""
        self._current = EpisodeRecord(
            episode_id=episode_id,
            task=task,
            env_id=self.env_id,
            is_ood=is_ood,
        )
        self._episode_start = time.perf_counter()

    def end_episode(self, success: bool) -> EpisodeRecord:
        """Finalize the current episode and record it."""
        assert self._current is not None, "Call begin_episode() first."
        self._current.success = success
        if self.track_timing:
            self._current.wall_time_s = round(
                time.perf_counter() - self._episode_start, 3
            )
        self._episodes.append(self._current)
        record = self._current
        self._current = None
        return record

    # ------------------------------------------------------------------
    # Recording methods (called by agents/evaluator)
    # ------------------------------------------------------------------

    def record_plan(
        self,
        n_subgoals: int,
        n_planning_iterations: int,
        verified: bool,
        verifier_accepted: bool,
        verifier_corrected: bool = False,
    ) -> None:
        """Record a planning + verification result."""
        if self._current is None:
            return
        self._current.n_subgoals_planned = n_subgoals
        self._current.n_planning_iterations = n_planning_iterations
        self._current.plan_was_verified = verified
        self._current.verifier_accepted = verifier_accepted
        self._current.verifier_corrected = verifier_corrected

    def record_step(self, subgoal_completed: bool = False) -> None:
        """Record one environment step."""
        if self._current is None:
            return
        self._current.n_env_steps += 1
        if subgoal_completed:
            self._current.n_subgoals_executed += 1

    def record_llm_call(
        self, agent: str, model: str, usage: dict
    ) -> None:
        """Record token usage from one LLM call."""
        if self._current is None or not self.track_tokens:
            return

        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        tt = usage.get("total_tokens", pt + ct)

        self._current.total_prompt_tokens += pt
        self._current.total_completion_tokens += ct
        self._current.total_tokens += tt

        # Per-agent breakdown
        if agent not in self._current.agent_tokens:
            self._current.agent_tokens[agent] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "n_calls": 0,
            }
        rec = self._current.agent_tokens[agent]
        rec["prompt_tokens"] += pt
        rec["completion_tokens"] += ct
        rec["total_tokens"] += tt
        rec["n_calls"] += 1

        # Cost estimation
        if self.track_cost:
            pricing = _MODEL_PRICING.get(model, _MODEL_PRICING["default"])
            cost = (pt / 1000) * pricing["prompt"] + (ct / 1000) * pricing["completion"]
            self._current.estimated_cost_usd += cost

    def record_trace_step(self, step_data: dict) -> None:
        """Append one step's trace data (plan/verify/act details)."""
        if self._current is not None and self.save_traces:
            self._current.trace.append(step_data)

    def record_success_step(self) -> None:
        """Mark the current env step count as the success step."""
        if self._current is not None:
            self._current.n_env_steps_to_success = self._current.n_env_steps

    # ------------------------------------------------------------------
    # Aggregation & saving
    # ------------------------------------------------------------------

    def compute_summary(self) -> RunSummary:
        """Compute aggregate metrics across all recorded episodes."""
        n = len(self._episodes)
        if n == 0:
            return RunSummary(
                run_id=self.run_id,
                experiment_name=self.experiment_name,
                env_id=self.env_id,
            )

        n_success = sum(1 for e in self._episodes if e.success)
        success_rate = n_success / n

        ood_eps = [e for e in self._episodes if e.is_ood]
        iid_eps = [e for e in self._episodes if not e.is_ood]
        ood_sr = (sum(1 for e in ood_eps if e.success) / len(ood_eps)) if ood_eps else 0.0
        iid_sr = (sum(1 for e in iid_eps if e.success) / len(iid_eps)) if iid_eps else 0.0

        avg_steps = sum(e.n_env_steps for e in self._episodes) / n
        avg_tokens = sum(e.total_tokens for e in self._episodes) / n
        avg_pi = sum(e.n_planning_iterations for e in self._episodes) / n
        avg_sg = sum(e.n_subgoals_planned for e in self._episodes) / n

        verified_eps = [e for e in self._episodes if e.plan_was_verified]
        plan_acc = (
            sum(1 for e in verified_eps if e.verifier_accepted) / len(verified_eps)
            if verified_eps else 0.0
        )
        # Verifier accuracy: correct if accepted→success or rejected→failure
        verifier_correct = sum(
            1 for e in self._episodes
            if e.plan_was_verified and (
                (e.verifier_accepted and e.success)
                or (not e.verifier_accepted and not e.success)
            )
        )
        ver_acc = verifier_correct / max(len(verified_eps), 1)

        total_cost = sum(e.estimated_cost_usd for e in self._episodes)
        total_time = sum(e.wall_time_s for e in self._episodes)

        return RunSummary(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            env_id=self.env_id,
            n_episodes=n,
            n_success=n_success,
            task_success_rate=round(success_rate, 4),
            avg_planning_iterations=round(avg_pi, 2),
            avg_subgoals=round(avg_sg, 2),
            plan_accuracy=round(plan_acc, 4),
            verifier_accuracy=round(ver_acc, 4),
            avg_env_steps=round(avg_steps, 2),
            sample_efficiency=round(success_rate / max(avg_steps, 1), 6),
            avg_total_tokens=round(avg_tokens, 1),
            token_efficiency=round(success_rate / max(avg_tokens, 1) * 1000, 6),
            total_cost_usd=round(total_cost, 6),
            avg_cost_usd=round(total_cost / n, 6),
            avg_wall_time_s=round(total_time / n, 3),
            total_wall_time_s=round(total_time, 3),
            ood_success_rate=round(ood_sr, 4),
            iid_success_rate=round(iid_sr, 4),
        )

    def save(self) -> tuple[Path, Path]:
        """Save full episode records (JSON) and summary (CSV).

        Returns
        -------
        (json_path, csv_path)
        """
        summary = self.compute_summary()

        # JSON — full episode data
        json_path = self.results_dir / f"{self.run_id}.json"
        payload = {
            "summary": asdict(summary),
            "episodes": [asdict(e) for e in self._episodes],
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        # CSV — summary row
        csv_path = self.results_dir / f"{self.run_id}_summary.csv"
        summary_dict = asdict(summary)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_dict.keys()))
            writer.writeheader()
            writer.writerow(summary_dict)

        return json_path, csv_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a rich summary table to stdout."""
        from rich.console import Console
        from rich.table import Table

        s = self.compute_summary()
        console = Console()
        table = Table(title=f"[bold]Phoenix Results — {s.experiment_name}[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        rows = [
            ("Experiment", s.experiment_name),
            ("Environment", s.env_id),
            ("Episodes", str(s.n_episodes)),
            ("Task Success Rate", f"{s.task_success_rate:.1%}"),
            ("Plan Accuracy", f"{s.plan_accuracy:.1%}"),
            ("Verifier Accuracy", f"{s.verifier_accuracy:.1%}"),
            ("Avg Planning Iterations", str(s.avg_planning_iterations)),
            ("Avg Sub-goals", str(s.avg_subgoals)),
            ("Avg Env Steps", str(s.avg_env_steps)),
            ("Sample Efficiency", f"{s.sample_efficiency:.4f}"),
            ("Avg Total Tokens", str(s.avg_total_tokens)),
            ("Token Efficiency", f"{s.token_efficiency:.4f}"),
            ("Est. Cost (total)", f"${s.total_cost_usd:.4f}"),
            ("OOD Success Rate", f"{s.ood_success_rate:.1%}"),
            ("IID Success Rate", f"{s.iid_success_rate:.1%}"),
            ("Total Wall Time", f"{s.total_wall_time_s:.1f}s"),
        ]
        for name, val in rows:
            table.add_row(name, val)

        console.print(table)
