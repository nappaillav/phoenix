# Phoenix 🔥

**Hierarchical LLM-Guided Planning for Robotic Manipulation**

Phoenix is a research framework that uses a multi-agent LLM pipeline to solve goal-conditioned robotic manipulation tasks in [Gymnasium-Robotics](https://robotics.farama.org/) environments (Fetch, Hand, Maze, Adroit). A high-level **Planner** decomposes tasks into sub-goals, an optional **Verifier** validates plans before execution, and a low-level **Policy** agent translates each sub-goal into continuous actions. A **Knowledge** agent provides procedural and affordance context to the planner.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Evaluator loop                       │
│                                                             │
│  ┌──────────┐   query   ┌───────────┐                       │
│  │Knowledge │◄──────────│  Planner  │                       │
│  │  Agent   │──context─►│  Agent    │──Plan──►┌──────────┐  │
│  └──────────┘           └───────────┘         │ Verifier │  │
│                                               │  Agent   │  │
│                         ┌───────────┐◄─Plan───└──────────┘  │
│  Environment ◄──action──│  Policy   │                       │
│  (Gymnasium- ──obs_text─►  Agent    │                       │
│   Robotics)             └───────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role |
|---|---|
| **KnowledgeAgent** | Maintains a seed knowledge graph of task procedures and object affordances. Answers natural-language queries from the Planner. |
| **PlannerAgent** | Decomposes the task into an ordered list of sub-goals (`Plan`). Delegates to one of three backends: `llm`, `mcts`, or `graph`. |
| **VerifierAgent** | Validates a proposed `Plan` for feasibility, goal alignment, and step consistency. Returns a `VerificationResult` that can include a corrected plan. |
| **PolicyAgent** | Maps each sub-goal + current observation to a continuous action vector. Parses an `ACTION: [...]` JSON array from the LLM response. |

### Planner Backends

| Backend | Description |
|---|---|
| `llm` | Chain-of-thought decomposition via a single LLM call (default). |
| `mcts` | UCB1 Monte Carlo Tree Search. The LLM acts as both the expansion policy (proposes candidate sub-goals) and the value function (scores leaf states). |
| `graph` | Knowledge-graph-based search inspired by HP-KG. Operates on a pre-built NetworkX graph. |

---

## Supported Environments

Phoenix ships with task descriptions and seed knowledge for the following Gymnasium-Robotics families. Any env ID prefixed with these names is recognized automatically:

| Family prefix | Task |
|---|---|
| `FetchReach` | Move end-effector to a target position |
| `FetchPush` | Push a puck to a target position |
| `FetchSlide` | Slide a puck to a target position |
| `FetchPickAndPlace` | Pick up a block and place it at a target |
| `HandManipulateBlock` | Reorient a block with a dexterous hand |
| `HandManipulatePen` | Reorient a pen with a dexterous hand |
| `HandManipulateEgg` | Reorient an egg without dropping it |
| `PointMaze` | Navigate a point agent through a maze |
| `AntMaze` | Drive an Ant robot through a maze |
| `AdroitHandDoor` | Open a door with a dexterous hand |
| `AdroitHandHammer` | Drive a nail with a hammer |
| `AdroitHandPen` | Reposition a pen in a dexterous hand |
| `AdroitHandRelocate` | Pick up and relocate a ball |

Any other env ID falls back to a generic task description.

---

## Installation

**Prerequisites:** Python ≥ 3.10, MuJoCo installed.

```bash
git clone https://github.com/nappaillav/phoenix.git
cd phoenix

pip install -r requirements.txt

# Also install simulation dependencies (not pinned in requirements.txt):
pip install gymnasium>=0.29.0 gymnasium-robotics>=1.3.0
```

**API keys** — create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # optional
```

---

## Quick Start

```bash
# Run with default config (FetchPickAndPlaceDense-v4, GPT-4o-mini)
python main.py

# Override config file
python main.py --config config/ablation_no_verifier.yaml

# Override individual fields
python main.py --env FetchReach-v2 --n-episodes 5 --model gpt-4o

# Structural dry-run (no LLM calls, validates pipeline only)
python main.py --dry-run

# Use a local Ollama model
python main.py --backend ollama --model llama3
```

---

## Configuration

All settings live in YAML files under `config/`. Fields are documented inline.

```yaml
# config/default.yaml (excerpt)
experiment:
  name: "phoenix_default"
  seed: 42
  n_episodes: 10
  max_steps_per_episode: 100
  planning_budget: 5        # max Planner→Verifier iterations per step

env:
  name: "FetchPickAndPlaceDense-v4"
  render_mode: null         # null | "human" | "rgb_array"
  obs_as_text: true         # serialize obs dict to text for LLM
  save_video: false         # record MP4 episodes (forces rgb_array)
  video_dir: null           # null → videos/<env_id>/

agents:
  policy: true
  knowledge: true
  planner: true
  verifier: true            # set false for ablation

planner:
  backend: "llm"            # llm | mcts | graph
  mcts:
    n_simulations: 50
    exploration_constant: 1.41
    max_depth: 10

llm:
  backend: "openai"         # openai | anthropic | ollama
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 1024
```

### Bundled configs

| File | Purpose |
|---|---|
| `config/default.yaml` | Full pipeline — all 4 agents active |
| `config/ablation_no_verifier.yaml` | Planner-only (no Verifier) |
| `config/ablation_mcts.yaml` | MCTS planner backend |
| `config/ablation_graph.yaml` | Graph planner backend |

---

## Video Recording

Set `save_video: true` in the config (or add the env key at runtime). The wrapper automatically switches `render_mode` to `rgb_array` if needed.

```yaml
env:
  save_video: true
  video_dir: "results/videos"   # optional; default: videos/<env_id>/
```

Episodes are saved as MP4 files named `<env_id>-episode-<N>.mp4`.

---

## Metrics & Output

After each run, two files are written to `results/`:

| File | Contents |
|---|---|
| `<run_id>.json` | Full episode traces — per-step sub-goals, rewards, token usage |
| `<run_id>_summary.csv` | Aggregate summary row |

### Tracked metrics

| Metric | Research question |
|---|---|
| Task success rate | RQ1, RQ3 |
| Plan accuracy (verifier first-pass acceptance rate) | RQ1 |
| Verifier accuracy | RQ4 |
| Avg planning iterations | RQ3 |
| Sample efficiency (`success_rate / avg_env_steps`) | RQ2 |
| Token efficiency (`success_rate / avg_tokens × 1000`) | RQ2 |
| Estimated USD cost | RQ3 |
| OOD / IID success rate split | RQ1 |
| Per-agent token breakdown | RQ2 |

A rich summary table is printed to the terminal at the end of every run.

---

## Prompt Templates

Each agent loads its prompt from `prompts/<agent_name>_prompt.md`. Edit these files to tune agent behavior without touching Python code. Available templates:

| File | Agent |
|---|---|
| `prompts/policy_prompt.md` | PolicyAgent |
| `prompts/planner_prompt.md` | PlannerAgent |
| `prompts/verifier_prompt.md` | VerifierAgent |
| `prompts/knowledge_prompt.md` | KnowledgeAgent |

Placeholders like `{subgoal}`, `{observation}`, `{action_dim}` are filled at runtime.

---

## Adding a New Environment

1. **Task description** — add an entry to `ENV_TASK_DESCRIPTIONS` in `envs/env_wrapper.py`:
   ```python
   "MyEnvPrefix": "Describe the goal in one sentence.",
   ```
   The key is matched with `str.startswith()`, so `"MyEnvPrefix-v2"` resolves automatically.

2. **Seed knowledge** (optional) — add a `"MyEnvPrefix": { "procedure": [...], "affordances": {...} }` entry to `_SEED_KNOWLEDGE` in `agents/knowledge_agent.py`.

3. **Config** — set `env.name` to your env ID.

---

## Project Structure

```
phoenix/
├── main.py                  # Entry point, CLI argument parsing
├── config/                  # YAML experiment configs
│   ├── default.yaml
│   ├── ablation_no_verifier.yaml
│   ├── ablation_mcts.yaml
│   └── ablation_graph.yaml
├── agents/                  # LLM agents
│   ├── base_agent.py        # Abstract base: LLM client, prompt loading, token tracking
│   ├── knowledge_agent.py   # Seed knowledge store + NL query answering
│   ├── planner_agent.py     # Task decomposition → Plan dataclass
│   ├── verifier_agent.py    # Plan validation → VerificationResult dataclass
│   └── policy_agent.py      # Sub-goal → continuous action vector
├── planners/                # Planner backends
│   ├── llm_planner.py       # Chain-of-thought (default)
│   ├── mcts_planner.py      # UCB1 Monte Carlo Tree Search
│   └── graph_planner.py     # NetworkX knowledge-graph search
├── envs/
│   └── env_wrapper.py       # PhoenixEnvWrapper (step/reset, obs→text, video, state save/restore)
├── evaluation/
│   └── evaluator.py         # Orchestrates Planner→Verifier→Policy loop
├── metrics/
│   └── collector.py         # EpisodeRecord, RunSummary, JSON/CSV output
├── prompts/                 # Markdown prompt templates (editable without code changes)
│   ├── policy_prompt.md
│   ├── planner_prompt.md
│   ├── verifier_prompt.md
│   └── knowledge_prompt.md
└── results/                 # Auto-created; stores JSON + CSV outputs
```

---

## LLM Backends

| Backend | Key env var | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | Default; any OpenAI-compatible model |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude models |
| `ollama` | — | Local models via `http://localhost:11434/v1`; set `base_url` in config |

---

## Ablation Experiments

Phoenix is designed around ablations. Toggle agents in the config:

```yaml
agents:
  policy: true
  knowledge: false   # ← remove knowledge context
  planner: false     # ← use stub single-step plan
  verifier: false    # ← skip verification loop
```

Use the `--config` flag to switch between bundled ablation configs, or pass `--dry-run` to validate the full pipeline structure without making any LLM calls.

---

## Citation

If you use Phoenix in your research, please cite the accompanying paper (link TBD).

---

## License

This project is for research purposes. See `LICENSE` for details.
