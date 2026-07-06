"""
envs/env_wrapper.py
===================
Unified wrapper around Gymnasium-Robotics environments.

Key responsibilities:
  - Expose a clean step/reset interface.
  - Serialize observations to a text description consumable by LLMs.
  - Support hypothetical state save/restore for Verifier Agent rollouts.
  - Track per-step sample counts for sample-efficiency metrics.
  - Optionally record episode videos via gymnasium RecordVideo wrapper.
  - Provide per-environment natural-language task descriptions for LLM prompts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np


# ---------------------------------------------------------------------------
# Per-environment task descriptions
# ---------------------------------------------------------------------------
# Add an entry here whenever you introduce a new env so that agents/prompts
# get a crisp, environment-specific goal statement instead of a generic one.
# Keys are matched with str.startswith() so a family prefix (e.g. "FetchReach")
# covers all density variants (Dense/Sparse, v3/v4, …).
ENV_TASK_DESCRIPTIONS: dict[str, str] = {
    # ── Fetch family ──────────────────────────────────────────────────────
    "FetchReach": (
        "Move the robot end-effector to the target position (red sphere) "
        "as quickly and precisely as possible."
    ),
    "FetchPush": (
        "Push the puck on the table to the target position using the "
        "robot's end-effector. Lifting the puck is not allowed."
    ),
    "FetchSlide": (
        "Hit the puck so that it slides across the frictionless table "
        "and comes to rest on the target position."
    ),
    "FetchPickAndPlace": (
        "Grasp the block on the table and place it at the "
        "target position, which may be in the air."
    ),
    # ── HandManipulate family ─────────────────────────────────────────────
    "HandManipulateBlock": (
        "Manipulate the block with the Shadow Dexterous Hand to achieve "
        "the target orientation shown by the goal pose."
    ),
    "HandManipulatePen": (
        "Rotate the pen in the Shadow Dexterous Hand to match "
        "the target orientation."
    ),
    "HandManipulateEgg": (
        "Reorient the egg in the Shadow Dexterous Hand to match "
        "the goal orientation without dropping it."
    ),
    # ── Maze family ───────────────────────────────────────────────────────
    "PointMaze": (
        "Navigate the point agent through the maze to reach the goal "
        "position marked in the observation."
    ),
    "AntMaze": (
        "Drive the Ant robot through the maze to reach the goal position."
    ),
    # ── Adroit family ─────────────────────────────────────────────────────
    "AdroitHandDoor": (
        "Open the door by grasping the handle and pulling/pushing it "
        "to the fully-open position."
    ),
    "AdroitHandHammer": (
        "Use the hammer to drive the nail into the board by striking it "
        "with the hammerhead."
    ),
    "AdroitHandPen": (
        "Reposition the pen in the hand to match the target configuration."
    ),
    "AdroitHandRelocate": (
        "Pick up the ball and move it to the target location."
    ),
}

_FALLBACK_TASK_DESCRIPTION = (
    "Complete the task defined by the environment's goal as indicated "
    "in the observation."
)


class PhoenixEnvWrapper:
    """Thin wrapper around a gymnasium-robotics environment.

    Parameters
    ----------
    env_id:
        Any valid ``gymnasium_robotics`` environment ID,
        e.g. ``"FetchPickAndPlace-v2"`` or ``"PointMaze_UMaze-v3"``.
    seed:
        Random seed forwarded to ``env.reset()``.
    render_mode:
        Passed to ``gym.make()``.  Use ``None`` for headless runs.
    obs_as_text:
        If ``True``, ``step()`` and ``reset()`` also return a
        human-readable text description of the observation for LLM use.
    save_video:
        If ``True``, episodes are recorded as MP4 files using
        ``gymnasium.wrappers.RecordVideo``.  Requires ``render_mode="rgb_array"``
        (the wrapper enforces this automatically if not set).
    video_dir:
        Directory to save recorded videos.  Defaults to ``"videos/<env_id>"``.
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        render_mode: str | None = None,
        obs_as_text: bool = True,
        save_video: bool = False,
        video_dir: str | Path | None = None,
    ) -> None:
        self.env_id = env_id
        self.seed = seed
        self.obs_as_text = obs_as_text
        self.save_video = save_video

        # gymnasium robotics
        gym.register_envs(gymnasium_robotics)

        # Video recording requires rgb_array render mode
        if save_video and render_mode is None:
            render_mode = "rgb_array"

        base_env = gym.make(env_id, render_mode=render_mode)

        if save_video:
            _video_dir = Path(video_dir) if video_dir else Path("videos") / env_id
            _video_dir.mkdir(parents=True, exist_ok=True)
            self._env = gym.wrappers.RecordVideo(
                base_env,
                video_folder=str(_video_dir),
                episode_trigger=lambda ep: True,  # record every episode
                name_prefix=env_id,
            )
            self.video_dir: Path | None = _video_dir
        else:
            self._env = base_env
            self.video_dir = None

        self._step_count: int = 0
        self._episode_count: int = 0

        # State snapshot stack for hypothetical rollouts
        self._state_stack: list[dict] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict, dict, str]:
        """Reset the environment.

        Returns
        -------
        obs : dict
            Raw observation from the environment.
        info : dict
            Auxiliary info dict.
        obs_text : str
            Human-readable text description of the observation.
        """
        obs, info = self._env.reset(seed=self.seed if self._episode_count == 0 else None)
        self._step_count = 0
        self._episode_count += 1
        return obs, info, self._obs_to_text(obs)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict, str]:
        """Take one environment step.

        Returns
        -------
        obs, reward, terminated, truncated, info, obs_text
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        return obs, reward, terminated, truncated, info, self._obs_to_text(obs)

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Hypothetical rollout support (for Verifier Agent)
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """Snapshot the current environment state.

        Note: Gymnasium-Robotics MuJoCo envs expose ``env.sim.get_state()``
        (MjSimState). We fall back to a deep copy of the unwrapped env's
        ``data`` if available, otherwise raise NotImplementedError so callers
        can handle gracefully.
        """
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, "sim"):
            # MuJoCo (mujoco-py) environments
            state = {
                "mj_state": copy.deepcopy(unwrapped.sim.get_state()),
                "goal": copy.deepcopy(getattr(unwrapped, "goal", None)),
                "step_count": self._step_count,
            }
        elif hasattr(unwrapped, "data"):
            # MuJoCo (mujoco bindings) environments
            import mujoco

            state = {
                "qpos": unwrapped.data.qpos.copy(),
                "qvel": unwrapped.data.qvel.copy(),
                "goal": copy.deepcopy(getattr(unwrapped, "goal", None)),
                "step_count": self._step_count,
            }
        else:
            raise NotImplementedError(
                f"State save not supported for {self.env_id}. "
                "Override save_state() for custom envs."
            )
        return state

    def restore_state(self, state: dict) -> None:
        """Restore a previously saved environment state."""
        unwrapped = self._env.unwrapped
        if "mj_state" in state:
            unwrapped.sim.set_state(state["mj_state"])
            unwrapped.sim.forward()
        elif "qpos" in state:
            unwrapped.data.qpos[:] = state["qpos"]
            unwrapped.data.qvel[:] = state["qvel"]
            import mujoco

            mujoco.mj_forward(unwrapped.model, unwrapped.data)
        else:
            raise NotImplementedError("Unknown state format.")

        if state.get("goal") is not None and hasattr(unwrapped, "goal"):
            unwrapped.goal = copy.deepcopy(state["goal"])

        self._step_count = state.get("step_count", self._step_count)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def task_description(self) -> str:
        """Return a natural-language description of the task for this environment.

        Looks up ``ENV_TASK_DESCRIPTIONS`` using a prefix match so that
        e.g. ``"FetchReachDense-v4"`` resolves to the ``"FetchReach"`` entry.
        Falls back to a generic description if no match is found.
        """
        for prefix, desc in ENV_TASK_DESCRIPTIONS.items():
            if self.env_id.startswith(prefix):
                return desc
        return _FALLBACK_TASK_DESCRIPTION

    # ------------------------------------------------------------------
    # Observation → text
    # ------------------------------------------------------------------

    def _obs_to_text(self, obs: Any) -> str:
        """Convert a raw observation to a compact text description.

        Prepends the environment-specific task description so that LLM agents
        always have goal context without needing a separate prompt field.
        Handles dict observations (standard in Gymnasium-Robotics goal-conditioned
        envs) as well as flat numpy arrays.
        """
        header = f"Task: {self.task_description}\n"
        if isinstance(obs, dict):
            lines = []
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    formatted = np.array2string(val, precision=4, suppress_small=True)
                else:
                    formatted = str(val)
                lines.append(f"  {key}: {formatted}")
            return header + "Observation:\n" + "\n".join(lines)
        elif isinstance(obs, np.ndarray):
            return header + f"Observation (array): {np.array2string(obs, precision=4, suppress_small=True)}"
        else:
            return header + f"Observation: {obs}"
