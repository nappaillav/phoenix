"""
envs/env_wrapper.py
===================
Unified wrapper around Gymnasium-Robotics environments.

Key responsibilities:
  - Expose a clean step/reset interface.
  - Serialize observations to a text description consumable by LLMs.
  - Support hypothetical state save/restore for Verifier Agent rollouts.
  - Track per-step sample counts for sample-efficiency metrics.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np


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
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 42,
        render_mode: str | None = None,
        obs_as_text: bool = True,
    ) -> None:
        self.env_id = env_id
        self.seed = seed
        self.obs_as_text = obs_as_text

        self._env = gym.make(env_id, render_mode=render_mode)
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

    # ------------------------------------------------------------------
    # Observation → text
    # ------------------------------------------------------------------

    def _obs_to_text(self, obs: Any) -> str:
        """Convert a raw observation to a compact text description.

        Handles dict observations (standard in Gymnasium-Robotics goal-conditioned
        envs) as well as flat numpy arrays.
        """
        if isinstance(obs, dict):
            lines = []
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    formatted = np.array2string(val, precision=4, suppress_small=True)
                else:
                    formatted = str(val)
                lines.append(f"  {key}: {formatted}")
            return "Observation:\n" + "\n".join(lines)
        elif isinstance(obs, np.ndarray):
            return f"Observation (array): {np.array2string(obs, precision=4, suppress_small=True)}"
        else:
            return f"Observation: {obs}"
