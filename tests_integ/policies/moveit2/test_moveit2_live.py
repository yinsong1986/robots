"""Integration test for :class:`MoveIt2Policy` against a live ZMQ sidecar.

Gated behind the ``moveit2`` pytest marker. Enable with:

.. code-block:: bash

    MOVEIT2_LIVE_SERVER=1 \
    MOVEIT2_SERVER_HOST=127.0.0.1 \
    MOVEIT2_SERVER_PORT=5556 \
    hatch run test-integ tests_integ/policies/moveit2/ -m moveit2 -v

A live sidecar can be started either via the docker-compose recipe at
``strands_robots/policies/moveit2/server/docker-compose.yml`` or via
``ros2 run`` (see the README under that directory). The test exercises
the same wire protocol the unit tests pin in-process — the difference is
end-to-end ZMQ + msgpack round-trip plus a real ``moveit_py`` planner on
the other side.

Acceptance criterion from issue #302:

  > At least one ``tests_integ/`` test (gated on the ``moveit2`` marker)
  > plans + executes a collision-free reach via the ZMQ sidecar.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("MOVEIT2_LIVE_SERVER", "").lower() in ("1", "true", "yes")
HOST = os.environ.get("MOVEIT2_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOVEIT2_SERVER_PORT", "5556"))
PLANNING_GROUP = os.environ.get("MOVEIT2_PLANNING_GROUP", "arm")

# Both the marker AND the env var gate are required so a CI run that
# happens to set MOVEIT2_LIVE_SERVER doesn't accidentally exercise the
# test without the explicit ``-m moveit2`` opt-in.
pytestmark = [
    pytest.mark.moveit2,
    pytest.mark.skipif(
        not LIVE,
        reason=(
            "Requires a pre-running MoveIt2 sidecar. "
            "Set MOVEIT2_LIVE_SERVER=1 plus MOVEIT2_SERVER_HOST / MOVEIT2_SERVER_PORT to enable."
        ),
    ),
]

# Skip cleanly if client extras aren't installed.
msgpack = pytest.importorskip("msgpack")
zmq = pytest.importorskip("zmq")

from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies.moveit2 import MoveIt2InferenceClient, MoveIt2Policy  # noqa: E402


@pytest.fixture(scope="module")
def policy() -> MoveIt2Policy:
    """Build a SERVICE-mode policy targeting the live sidecar."""
    p = create_policy(
        "moveit2",
        host=HOST,
        port=PORT,
        planning_group=PLANNING_GROUP,
        timeout_ms=30000,
    )
    assert isinstance(p, MoveIt2Policy)
    return p


def test_ping_reaches_live_sidecar() -> None:
    """Pre-flight: the sidecar's ``ping`` endpoint must respond.

    Run before the planning tests so a network / process-down failure
    surfaces as a clear ``ping`` error instead of an opaque planner
    timeout."""
    client = MoveIt2InferenceClient(host=HOST, port=PORT, timeout_ms=5000)
    assert client.ping(), f"MoveIt2 sidecar did not respond at {HOST}:{PORT}"


def test_plan_reach_target_pose_returns_trajectory(policy: MoveIt2Policy) -> None:
    """A typical Cartesian reach plan returns a non-empty trajectory.

    The exact pose is deliberately conservative — close to a typical
    UR5 / Panda home configuration in front of the base — so the test
    works against a wide range of real-world ``moveit_py`` configs
    without per-sidecar tuning. Sidecars where this pose is
    unreachable should override via ``MOVEIT2_TARGET_POSE`` (JSON list).
    """
    import json

    pose_str = os.environ.get("MOVEIT2_TARGET_POSE")
    if pose_str:
        target_pose = json.loads(pose_str)
    else:
        target_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

    actions = policy.get_actions_sync(
        observation_dict={},
        instruction="reach for the goal",
        target_pose=target_pose,
    )
    assert isinstance(actions, list)
    assert len(actions) > 0, "Sidecar must return a non-empty trajectory"
    for step in actions:
        assert isinstance(step, dict)
        assert step, "Per-step action dict must be non-empty"


def test_plan_target_joints_returns_trajectory(policy: MoveIt2Policy) -> None:
    """Joint-space goal goes through the same wire path."""
    import json

    joints_str = os.environ.get("MOVEIT2_TARGET_JOINTS")
    if joints_str:
        target_joints = json.loads(joints_str)
    else:
        # Tiny offsets from any reasonable home configuration.
        target_joints = {"joint_1": 0.1, "joint_2": -0.1}

    actions = policy.get_actions_sync(
        observation_dict={},
        instruction="",
        target_joints=target_joints,
    )
    assert isinstance(actions, list)
    assert len(actions) > 0


def test_failed_plan_raises_runtime_error(policy: MoveIt2Policy) -> None:
    """An obviously unreachable pose surfaces as a ``RuntimeError`` with
    the sidecar's status code, not as silent zero actions.

    Pose 99 m above the base is unreachable for any ground-mounted arm,
    so the sidecar returns ``success=False``."""
    import pytest as _pytest

    with _pytest.raises(RuntimeError):
        policy.get_actions_sync(
            observation_dict={},
            instruction="",
            target_pose=[0.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0],
        )
