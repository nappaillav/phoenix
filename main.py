"""
main.py
=======
Phoenix — entry point for hierarchical LLM-guided robotic manipulation.

Usage:
    # Full framework (default config)
    python main.py

    # Override config
    python main.py --config config/ablation_no_verifier.yaml

    # Structural dry-run (no LLM calls, no env stepping)
    python main.py --dry-run

    # Override specific config fields
    python main.py --config config/default.yaml --env FetchReach-v2 --n-episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Add project root to path so subpackages import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()  # load OPENAI_API_KEY / ANTHROPIC_API_KEY from .env if present

console = Console()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML config, applying env-var overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Fill defaults for any missing top-level keys
    cfg.setdefault("experiment", {})
    cfg.setdefault("env", {"name": "FetchPickAndPlace-v2"})
    cfg.setdefault("agents", {})
    cfg.setdefault("llm", {"backend": "openai", "model": "gpt-4o"})
    cfg.setdefault("planner", {"backend": "llm"})
    cfg.setdefault("metrics", {})
    cfg.setdefault("output", {"results_dir": "results", "format": ["json", "csv"]})

    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply command-line argument overrides to the config."""
    if args.dry_run:
        cfg["experiment"]["dry_run"] = True
    if args.env:
        cfg["env"]["name"] = args.env
    if args.n_episodes is not None:
        cfg["experiment"]["n_episodes"] = args.n_episodes
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    if args.planner_backend:
        cfg["planner"]["backend"] = args.planner_backend
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phoenix: Hierarchical LLM-Guided Planning for Robotic Manipulation"
    )
    parser.add_argument(
        "--config",
        # default="/usr/local/data/vchida/wm_research/phoenix/config/default.yaml",
        default="/usr/local/data/vchida/wm_research/phoenix/config/ablation_no_verifier.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM calls (validates structure only)",
    )
    parser.add_argument("--env", help="Override environment ID")
    parser.add_argument("--n-episodes", type=int, help="Override n_episodes")
    parser.add_argument("--model", help="Override LLM model name")
    parser.add_argument("--backend", choices=["openai", "anthropic", "ollama"],
                        help="Override LLM backend")
    parser.add_argument("--planner-backend", choices=["llm", "mcts", "graph"],
                        help="Override planner backend")
    parser.add_argument("--run-id", help="Custom run ID (default: auto-generated)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))
    cfg = apply_cli_overrides(cfg, args)

    # ── Run ID ───────────────────────────────────────────────────────────
    run_id = args.run_id or (
        cfg["experiment"].get("name", "run")
        + "_"
        + time.strftime("%Y%m%d_%H%M%S")
    )

    # ── Print header ─────────────────────────────────────────────────────
    console.rule("[bold magenta]Phoenix: Hierarchical LLM-Guided Planning")
    console.print(f"  Config      : [cyan]{config_path}")
    console.print(f"  Run ID      : [cyan]{run_id}")
    console.print(f"  Environment : [cyan]{cfg['env']['name']}")
    console.print(f"  LLM         : [cyan]{cfg['llm']['backend']} / {cfg['llm']['model']}")
    console.print(f"  Planner     : [cyan]{cfg['planner']['backend']}")
    console.print(f"  Dry run     : [cyan]{cfg['experiment'].get('dry_run', False)}")
    console.print(f"  Episodes    : [cyan]{cfg['experiment'].get('n_episodes', 10)}")
    console.print()

    # ── Metrics collector ─────────────────────────────────────────────────
    from metrics.collector import MetricsCollector

    metrics_cfg = cfg.get("metrics", {})
    results_dir = cfg["output"].get("results_dir", "results")

    metrics = MetricsCollector(
        run_id=run_id,
        experiment_name=cfg["experiment"].get("name", run_id),
        env_id=cfg["env"]["name"],
        results_dir=results_dir,
        track_tokens=metrics_cfg.get("track_tokens", True),
        track_cost=metrics_cfg.get("track_cost", True),
        track_timing=metrics_cfg.get("track_timing", True),
        save_traces=metrics_cfg.get("save_episode_traces", True),
    )

    # ── Evaluator ─────────────────────────────────────────────────────────
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(cfg=cfg, metrics=metrics)

    try:
        evaluator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user — saving partial results.")

    # ── Save & print results ──────────────────────────────────────────────
    json_path, csv_path = metrics.save()
    metrics.print_summary()

    console.print()
    console.print(f"[green]Results saved to:")
    console.print(f"  JSON : [cyan]{json_path}")
    console.print(f"  CSV  : [cyan]{csv_path}")


if __name__ == "__main__":
    main()
