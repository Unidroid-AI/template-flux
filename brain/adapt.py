# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map observations to FLUX 3 Action DROID views and back to a joint command chunk."""

from __future__ import annotations

import numpy as np

VIEW_HW = (360, 640)
SECOND_EXTERIOR_MODES = ("camera", "duplicate", "black")
# The model and the eval both run at 15 Hz. One row of a chunk is one control step.
CONTROL_HZ = 15.0
# Revolute limits [rad] authored on the Franka USD, and the drive velocity limits [rad/s].
# Joints 1-4 are the shoulder drive; joints 5-7 are the forearm drive.
JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)
JOINT_VELOCITY = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
# Ready pose [rad], used when the simulator has already lost the arm state.
READY_JOINTS = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0], dtype=np.float32)


def to_droid_view(image: np.ndarray, height: int = VIEW_HW[0], width: int = VIEW_HW[1]) -> np.ndarray:
    """Resize one HWC image to the DROID view size with an antialiased filter.

    Args:
        image: RGB image, shape [H, W, 3]. Non-uint8 values are clipped to 0..255.
        height: Target height [pixels].
        width: Target width [pixels].

    Returns:
        uint8 image, shape [height, width, 3]. An image of that size is returned unchanged.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[:2] == (height, width):
        return array
    from PIL import Image

    resized = Image.fromarray(array).resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized)


def droid_views(
    exterior: np.ndarray,
    wrist: np.ndarray,
    right: np.ndarray | None = None,
    *,
    second_exterior: str = "camera",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the wrist, left exterior, and right exterior views FLUX 3 Action expects.

    ``camera`` uses the rendered right exterior. ``duplicate`` copies the left exterior,
    and ``black`` sends a blank frame. Those two ignore ``right``.

    Args:
        exterior: Left exterior camera image, shape [H, W, 3].
        wrist: Wrist camera image, shape [H, W, 3].
        right: Right exterior camera image, shape [H, W, 3]. Required when
            ``second_exterior`` is ``camera``.
        second_exterior: ``camera``, ``duplicate``, or ``black``.

    Returns:
        Three uint8 views, each shape [360, 640, 3], in the order wrist, left, right.
    """
    if second_exterior not in SECOND_EXTERIOR_MODES:
        raise ValueError(f"second_exterior must be one of {SECOND_EXTERIOR_MODES}, got {second_exterior!r}")
    left = to_droid_view(exterior)
    if second_exterior == "duplicate":
        right_view = left.copy()
    elif second_exterior == "black":
        right_view = np.zeros_like(left)
    elif right is None:
        raise ValueError("cam_exterior_right is required when second_exterior is camera")
    else:
        right_view = to_droid_view(right)
    return to_droid_view(wrist), left, right_view


def command_chunk(actions: np.ndarray, offset: int, horizon: int, joints: np.ndarray) -> np.ndarray:
    """Slice a policy chunk into joint commands the Franka drives can track.

    A raw FLUX target can sit radians away from the measured pose. The position drive then
    applies full effort while the hand is still in contact, and the joint state becomes
    non-finite. Each kept row stays inside the joint limits and within one drive step of
    the previous command, starting from the measured joints.

    Args:
        actions: Policy actions, shape [K, 8]. Seven absolute joint targets [rad] and the
            gripper closed fraction.
        offset: First action to keep. Skips actions that are already out of date when the
            chunk starts to run.
        horizon: Number of actions to keep.
        joints: Measured arm joints [rad], shape [7].

    Returns:
        float32 actions, shape [horizon, 8]. The gripper column is clipped to 0..1; the
        environment closes the gripper above 0.5.
    """
    raw = np.asarray(actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 8 or raw.shape[0] < offset + horizon:
        raise ValueError(f"expected at least {(offset + horizon, 8)} actions, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("policy returned non-finite actions")
    measured = np.asarray(joints, dtype=np.float32).reshape(-1)
    if measured.shape != (7,) or not np.isfinite(measured).all():
        raise ValueError(f"expected finite joints of shape (7,), got {measured.shape}")
    chunk = raw[offset : offset + horizon].copy()
    chunk[:, :7] = _track_joints(chunk[:, :7], measured)
    chunk[:, -1] = np.clip(chunk[:, -1], 0.0, 1.0)
    return chunk


def hold_chunk(joints: np.ndarray | None, horizon: int) -> np.ndarray:
    """Repeat one legal pose for a whole chunk.

    Used when the simulator state is already non-finite, so the policy is not called.
    A finite measurement is held. A missing or non-finite measurement holds the ready pose.

    Args:
        joints: Arm joints [rad], shape [7], or None.
        horizon: Number of repeated rows.

    Returns:
        float32 actions, shape [horizon, 8], with the gripper open.
    """
    pose = np.zeros(8, dtype=np.float32)
    arm = None if joints is None else np.asarray(joints, dtype=np.float32).reshape(-1)
    if arm is not None and arm.shape[0] >= 7 and np.isfinite(arm[:7]).all():
        pose[:7] = np.clip(arm[:7], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    else:
        pose[:7] = READY_JOINTS
    return np.repeat(pose[None], horizon, axis=0)


def _track_joints(targets: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Clip targets to the joint limits and approach them at the drive velocity limit."""
    step = JOINT_VELOCITY / CONTROL_HZ
    current = np.clip(measured, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    tracked = np.empty_like(targets)
    for index, row in enumerate(targets):
        desired = np.clip(row, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        current = current + np.clip(desired - current, -step, step)
        tracked[index] = current
    return tracked
