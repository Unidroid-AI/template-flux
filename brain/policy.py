# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FLUX 3 Action DROID brain for the Franka tabletop task.

``weights/policy`` is the released ``black-forest-labs/flux-3-action-droid`` package.
``weights/video_vae.safetensors`` and ``weights/text_encoder`` are the shared frozen
encoders it references, stored locally because the brain runs offline. The flux-action
runtime lives in the brain image. The policy reads the left exterior, right exterior, and wrist cameras, the
seven arm joints, and the gripper, then returns this task's 8-D absolute joint command.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from adapt import command_chunk, droid_views, hold_chunk
from brain_comm.brain import Brain


class Flux3DroidBrain(Brain):
    """Load a FLUX 3 Action DROID package and return absolute joint actions."""

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        # Eager inference is the default: compiled runs differ across processes by about 0.015 rad.
        self._offset = int(config.get("action_offset", 0))
        self._second_exterior = str(config.get("second_exterior", "camera"))
        self._adapter = _load_adapter(weights_dir, config, device)
        self._instructions: list[str] = []
        self._last_joints: np.ndarray | None = None

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        del episode_ids, seeds
        self._instructions = list(instructions)

    def end_episode(self, episode_ids: list[str]) -> None:
        """Drop the instruction batch and return unused CUDA blocks to the driver."""
        del episode_ids
        self._instructions = []
        self._last_joints = None
        _release_cached_gpu_memory()

    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        joints = np.asarray(obs["joint_pos"], dtype=np.float32)
        grip = np.asarray(obs["gripper_pos"], dtype=np.float32)
        exterior = np.asarray(obs["cam_exterior_left"])
        wrist = np.asarray(obs["cam_wrist"])
        right = np.asarray(obs["cam_exterior_right"]) if "cam_exterior_right" in obs else None
        count = joints.shape[0]
        if len(self._instructions) != count:
            raise ValueError(f"{len(self._instructions)} instructions for {count} envs")
        # The lockstep loop prefetches every chunk after the first from an observation
        # that is already half a chunk old, so the first actions of that chunk are stale.
        offset = self._offset if step > 0 else 0
        chunks = []
        for index in range(count):
            joint_row = np.asarray(joints[index], dtype=np.float32).reshape(-1)
            grip_row = np.asarray(grip[index], dtype=np.float32).reshape(-1)
            # A non-finite arm state is what the policy rejects as "invalid state".
            # Hold the last legal pose instead of ending the worker.
            if joint_row.shape != (7,) or not np.isfinite(joint_row).all() or not np.isfinite(grip_row).all():
                chunks.append(hold_chunk(self._last_joints, self.horizon))
                continue
            self._last_joints = joint_row
            request = _request(
                droid_views(
                    exterior[index],
                    wrist[index],
                    None if right is None else right[index],
                    second_exterior=self._second_exterior,
                ),
                joints[index],
                grip[index],
                self._instructions[index],
            )
            # The server reseeds numpy for each act, so this seed repeats for a repeated run.
            seed = int(np.random.randint(0, 2**31 - 1))
            actions = self._adapter.infer(request, seed=seed)["action"]
            chunks.append(command_chunk(actions, offset, self.horizon, joint_row))
        stacked = np.stack(chunks, axis=0).astype(np.float32)
        if stacked.shape[-1] != self.action_dim:
            raise ValueError(f"action width {stacked.shape[-1]} != {self.action_dim}")
        return stacked


def _load_adapter(weights_dir: Path, config: dict, device: str):
    package = weights_dir / "policy"
    if not (package / "manifest.json").is_file():
        raise FileNotFoundError(f"FLUX policy package is missing manifest.json in {package}")
    from flux_action.models.text_encoder import load_text_encoder
    from flux_action.models.video_vae import load_video_vae
    from flux_action.policy import FluxActionPolicy
    from flux_action.serving.robolab import RoboLabPolicy, prepare_serving_policy

    # Passing the encoders in keeps the package config, which names Hub repos, unchanged.
    video_vae = load_video_vae(str(weights_dir / "video_vae.safetensors"))
    text_encoder = load_text_encoder(str(weights_dir / "text_encoder"))
    policy = FluxActionPolicy.from_pretrained(
        package,
        video_vae=video_vae,
        text_encoder=text_encoder,
        device="cpu",
        subfolder=str(config.get("flux_subfolder") or "") or None,
    )
    # The policy is still on the CPU here. Offloading the text encoder before the move
    # keeps the GPU peak within max_gpu_mem_gb.
    prepared = prepare_serving_policy(
        policy,
        device=device,
        compile_dit=bool(config.get("compile", False)),
        offload_text_encoder=bool(config.get("offload_text_encoder", True)),
        warmup=1,
    )
    return RoboLabPolicy(prepared)


def _request(
    views: tuple[np.ndarray, np.ndarray, np.ndarray], joints: np.ndarray, grip: np.ndarray, prompt: str
) -> dict:
    wrist, left, right = views
    return {
        "observation/wrist_image_left": wrist,
        "observation/exterior_image_1_left": left,
        "observation/exterior_image_2_left": right,
        "observation/joint_position": np.asarray(joints, dtype=np.float32),
        "observation/gripper_position": np.asarray(grip, dtype=np.float32).reshape(-1),
        "prompt": prompt,
    }


def _release_cached_gpu_memory() -> None:
    """Return the caching allocator's unused blocks. Live weights are kept."""
    import gc

    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
