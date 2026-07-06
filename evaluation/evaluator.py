"""
evaluation/evaluator.py
=======================
Evaluator — orchestrates the Planner→Verifier→Policy loop.

The Evaluator:
  1. Reads the config to instantiate only the configured agents.
  2. Runs N episodes, each following the planning-verification loop.
  3. Feeds metrics to MetricsCollector after every step and episode.
  4. Supports dry-run mode (no LLM calls) for structural validation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from agents.planner_agent import Plan
from agents.verifier_agent import VerificationResult
from envs.env_wrapper import PhoenixEnvWrapper
from metrics.collector import MetricsCollector

console = Console()


class Evaluator:
    """Runs the Phoenix Planner→Verifier→Policy evaluation loop.

    Parameters
    ----------
    cfg:
        Full loaded YAML config dict.
    metrics:
        ``MetricsCollector`` instance (created externally so it can be
        shared across multiple ``Evaluator`` instances in sweeps).
    """

    def __init__(self, cfg: dict, metrics: MetricsCollector) -> None:
        self.cfg = cfg
        self.metrics = metrics
        self.dry_run: bool = cfg["experiment"].get("dry_run", False)

        # Build environment
        env_cfg = cfg["env"]
        self.env = PhoenixEnvWrapper(
            env_id=env_cfg["name"],
            seed=cfg["experiment"].get("seed", 42),
            render_mode=env_cfg.get("render_mode"),
            obs_as_text=env_cfg.get("obs_as_text", True),
            save_video=env_cfg.get("save_video", False),
            video_dir=env_cfg.get("video_dir"),
        )

        # Build active agents
        self.agents = self._build_agents()

        # Experiment params
        self.n_episodes: int = cfg["experiment"].get("n_episodes", 10)
        self.max_steps: int = cfg["experiment"].get("max_steps_per_episode", 50)
        self.planning_budget: int = cfg["experiment"].get("planning_budget", 5)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> MetricsCollector:
        """Run all episodes and return the MetricsCollector."""
        console.rule(f"[bold cyan]Phoenix Evaluation — {self.cfg['experiment']['name']}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Running {self.n_episodes} episodes ...", total=self.n_episodes
            )

            for ep_id in range(self.n_episodes):
                self._run_episode(ep_id)
                progress.advance(task)

        self.env.close()
        return self.metrics

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(self, ep_id: int) -> None:
        """Execute one complete episode."""
        obs, info, obs_text = self.env.reset()
        task_str = self._extract_task(info, obs)
        goal_text = self._extract_goal(info, obs)

        self.metrics.begin_episode(episode_id=ep_id, task=task_str)

        # Reset per-episode agent state
        for agent in self.agents.values():
            agent.reset()

        plan: Plan | None = None
        history: list[str] = []
        success = False

        for step in range(self.max_steps):
            # ── 1. Planning + Verification ────────────────────────────
            if plan is None or self._needs_replan(step, plan):
                plan, n_iter = self._planning_loop(
                    task=task_str,
                    obs_text=obs_text,
                    goal_text=goal_text,
                    history=history,
                    ep_id=ep_id,
                    step=step,
                )

            # ── 2. Select current sub-goal ───────────────────────────
            if plan.is_empty():
                console.print(f"  [yellow]Ep {ep_id} step {step}: empty plan, skipping.")
                break

            subgoal_idx = min(
                step // max(self.max_steps // max(len(plan), 1), 1), len(plan) - 1
            )
            subgoal = plan.subgoals[subgoal_idx]

            # ── 3. Policy: sub-goal → action ─────────────────────────
            action = self._get_action(subgoal=subgoal, obs_text=obs_text, history=history)

            # ── 4. Environment step ──────────────────────────────────
            obs, reward, terminated, truncated, info, obs_text = self.env.step(action)
            self.metrics.record_step()

            # ── 5. Check success ─────────────────────────────────────
            success = bool(info.get("is_success", False)) or terminated
            if success:
                self.metrics.record_success_step()

            # ── 6. Trace ─────────────────────────────────────────────
            self.metrics.record_trace_step({
                "step": step,
                "subgoal": subgoal,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "success": success,
            })

            # Record plan/history for context
            history.append(f"[step {step}] subgoal={subgoal!r} reward={reward:.3f}")

            if success or truncated:
                break

        ep_record = self.metrics.end_episode(success=success)
        status = "[green]✓ SUCCESS" if success else "[red]✗ FAIL"
        console.print(
            f"  Ep {ep_id:3d} | {status} | steps={ep_record.n_env_steps} "
            f"| tokens={ep_record.total_tokens} | cost=${ep_record.estimated_cost_usd:.4f}"
        )

    # ------------------------------------------------------------------
    # Planning loop (Planner → Verifier → repeat up to budget)
    # ------------------------------------------------------------------

    def _planning_loop(
        self,
        task: str,
        obs_text: str,
        goal_text: str,
        history: list[str],
        ep_id: int,
        step: int,
    ) -> tuple[Plan, int]:
        """Run the Planner→Verifier loop, returning the accepted plan."""
        planner = self.agents.get("planner")
        verifier = self.agents.get("verifier")

        plan: Plan | None = None
        verified = verifier is not None
        accepted = True  # default if no verifier
        corrected = False
        n_iter = 0

        for iteration in range(self.planning_budget):
            n_iter += 1

            # ── Plan ──────────────────────────────────────────────────
            if planner is not None:
                plan = planner.act(
                    task=task,
                    obs_text=obs_text,
                    goal_text=goal_text,
                    history=history,
                )
            else:
                # No planner — use a single stub sub-goal
                from agents.planner_agent import Plan
                plan = Plan(subgoals=[task], reasoning="No planner configured.")

            # ── Verify ────────────────────────────────────────────────
            if verifier is not None and not plan.is_empty():
                result: VerificationResult = verifier.act(
                    plan=plan,
                    obs_text=obs_text,
                    goal_text=goal_text,
                    task=task,
                )
                accepted = result.feasible
                if not accepted and result.corrected_plan is not None:
                    plan = result.corrected_plan
                    corrected = True
                    console.print(
                        f"    [dim]Verifier rejected plan (ep={ep_id}, iter={iteration}); "
                        f"using corrected plan.[/dim]"
                    )
                elif accepted:
                    break
            else:
                accepted = True
                break

        self.metrics.record_plan(
            n_subgoals=len(plan) if plan else 0,
            n_planning_iterations=n_iter,
            verified=verified,
            verifier_accepted=accepted,
            verifier_corrected=corrected,
        )

        from agents.planner_agent import Plan as PlanCls
        return (plan if plan is not None else PlanCls(subgoals=[task])), n_iter

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _get_action(
        self, subgoal: str, obs_text: str, history: list[str]
    ) -> "import numpy as np; np.ndarray":
        """Get a low-level action from the Policy Agent (or random fallback)."""
        import numpy as np

        policy = self.agents.get("policy")
        if policy is not None:
            action, _ = policy.act(
                subgoal=subgoal,
                obs_text=obs_text,
                history=history,
            )
            return action
        # Fallback: random action
        return self.env.action_space.sample()

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _build_agents(self) -> dict:
        """Instantiate agents as specified by config.agents."""
        agents_cfg = self.cfg.get("agents", {})
        llm_cfg = self.cfg.get("llm", {})
        planner_cfg = self.cfg.get("planner", {})
        env_cfg = self.cfg.get("env", {})

        common = dict(llm_cfg=llm_cfg, metrics=self.metrics, dry_run=self.dry_run)
        active: dict = {}

        # Knowledge Agent
        if agents_cfg.get("knowledge", True):
            from agents.knowledge_agent import KnowledgeAgent
            active["knowledge"] = KnowledgeAgent(env_id=env_cfg["name"], **common)

        # Planner Agent
        if agents_cfg.get("planner", True):
            from agents.planner_agent import PlannerAgent
            active["planner"] = PlannerAgent(
                planner_cfg=planner_cfg,
                knowledge_agent=active.get("knowledge"),
                **common,
            )

        # Verifier Agent
        if agents_cfg.get("verifier", True):
            from agents.verifier_agent import VerifierAgent
            active["verifier"] = VerifierAgent(env_wrapper=self.env, **common)

        # Policy Agent
        if agents_cfg.get("policy", True):
            from agents.policy_agent import PolicyAgent
            active["policy"] = PolicyAgent(
                action_space=self.env.action_space, **common
            )

        console.print(f"[bold]Active agents:[/bold] {list(active.keys())}")
        return active

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_task(self, info: dict, obs: Any) -> str:
        """Return the environment's task description for agent prompts.

        Delegates to ``env.task_description`` (which does a prefix lookup in
        ``ENV_TASK_DESCRIPTIONS``) so we get a rich, env-specific sentence
        rather than a raw goal array or a generic fallback.
        """
        return self.env.task_description

    def _extract_goal(self, info: dict, obs: Any) -> str:
        """Extract goal description for prompts."""
        if isinstance(obs, dict) and "desired_goal" in obs:
            import numpy as np
            return np.array2string(obs["desired_goal"], precision=4)
        return str(info.get("goal", "achieve the task objective"))

    def _needs_replan(self, step: int, plan: Plan) -> bool:
        """Decide whether to re-plan mid-episode.

        Currently replans every time all sub-goals have been consumed.
        """
        return step > 0 and (step % max(len(plan), 1)) == 0
