# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "brain" / "adapt.py"
_SPEC = importlib.util.spec_from_file_location("flux3_adapt", _PATH)
adapt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapt)


def test_distant_targets_stay_within_one_drive_step():
    measured = adapt.READY_JOINTS.copy()
    raw = np.zeros((16, 8), dtype=np.float32)
    raw[:, :7] = 10.0
    raw[:, 7] = 2.0
    chunk = adapt.command_chunk(raw, 0, 16, measured)
    step = adapt.JOINT_VELOCITY / adapt.CONTROL_HZ
    assert chunk.shape == (16, 8)
    assert np.isfinite(chunk).all()
    np.testing.assert_allclose(chunk[0, :7], measured + step, atol=1e-6)
    assert np.all(chunk[:, :7] >= adapt.JOINT_LIMITS[:, 0] - 1e-6)
    assert np.all(chunk[:, :7] <= adapt.JOINT_LIMITS[:, 1] + 1e-6)
    assert np.all(chunk[:, 7] == 1.0)
    delta = np.diff(chunk[:, :7], axis=0)
    assert np.all(np.abs(delta) <= step + 1e-5)


def test_non_finite_state_holds_the_ready_pose():
    chunk = adapt.hold_chunk(np.array([np.nan] * 7), 16)
    assert chunk.shape == (16, 8)
    np.testing.assert_allclose(chunk[:, :7], np.broadcast_to(adapt.READY_JOINTS, (16, 7)))
    assert np.all(chunk[:, 7] == 0.0)
