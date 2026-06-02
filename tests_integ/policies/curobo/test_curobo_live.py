"""Integration test for :class:`CuroboPolicy` against the live cuRobo planner.

Gated behind both the ``curobo`` pytest marker AND a CUDA-capable GPU
detection. Enable with:

.. code-block:: bash

    pip install 'strands-robots[curobo]'
    hatch run test-integ tests_integ/policies/curobo/ -m curobo -v

A pre-installed cuRobo + a CUDA-capable GPU are required. The test
plans a UR5e reach to a target pose, asserts the trajectory is non-empty
and collision-free (``result.success``), and checks the chunked-action
yield matches the ``action_horizon``.

Acceptance criteria from issue #301:

  > Integration test in ``tests_integ/policies/curobo/`` that plans a
  > UR5e reach to a target pose, asserts collision-free, executes in sim.
"""

from __future__ import annotations

import os

import pytest

# Skip cleanly if cuRobo isn't installed (the ``[curobo]`` extra wasn't
# enabled). ``importorskip`` produces a SKIPPED line with a clear reason
# rather than an opaque ImportError trace at collection time.
curobo = pytest.importorskip(
    "curobo",
    reason="curobo not installed - pip install 'strands-robots[curobo]'",
)

# Both the marker AND a GPU-detected env are required so a CI run that
# happens to have cuRobo installed without a GPU doesn't try to compile
# CUDA kernels and time out.
LIVE = os.environ.get("CUROBO_LIVE", "").lower() in ("1", "true", "yes")

pytestmark = [
    pytest.mark.curobo,
    pytest.mark.skipif(
        not LIVE,
        reason=(
            "Requires CUDA-capable GPU + cuRobo installed. "
            "Set CUROBO_LIVE=1 to enable; the policy will pick the "
            "default cuda:0 device."
        ),
    ),
]

# E402: importorskip must execute before these imports to skip cleanly.
from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies.curobo import CuroboPolicy  # noqa: E402

# Default UR5e robot config that ships with cuRobo. Override via
# ``CUROBO_ROBOT_CONFIG`` for sites that prefer a custom YAML.
ROBOT_CONFIG = os.environ.get("CUROBO_ROBOT_CONFIG", "ur5e.yml")
ACTION_HORIZON = int(os.environ.get("CUROBO_ACTION_HORIZON", "16"))


@pytest.fixture(scope="module")
def policy() -> CuroboPolicy:
    """Build a UR5e :class:`CuroboPolicy` and warm it up once."""
    p = create_policy(
        "curobo",
        robot_config=ROBOT_CONFIG,
        action_horizon=ACTION_HORIZON,
        # Keep warmup ON in the live test — the first ``plan_single``
        # would otherwise pay a multi-second JIT cost that the chunked
        # action assertions below would mistake for a timeout.
        warmup=True,
    )
    assert isinstance(p, CuroboPolicy)
    return p


def test_plan_reach_target_pose_returns_collision_free_trajectory(
    policy: CuroboPolicy,
) -> None:
    """A typical Cartesian reach plan returns a non-empty, collision-free
    trajectory.

    The pose is a conservative reachable target in front of a UR5e
    home configuration. Sites with a different robot can override via
    ``CUROBO_TARGET_POSE`` (JSON list) and ``CUROBO_HOME_STATE`` (JSON list).
    """
    import json

    pose_str = os.environ.get("CUROBO_TARGET_POSE")
    target_pose = json.loads(pose_str) if pose_str else [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

    home_str = os.environ.get("CUROBO_HOME_STATE")
    home_state = json.loads(home_str) if home_str else [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

    actions = policy.get_actions_sync(
        observation_dict={"observation.state": home_state},
        instruction="reach for the target",
        target_pose=target_pose,
    )

    # The first call yields the first ``action_horizon`` chunk of the
    # cached trajectory. Empty -> planner failed silently, which is a
    # bug the integration test exists to catch.
    assert isinstance(actions, list)
    assert len(actions) > 0, "cuRobo must return a non-empty trajectory chunk"
    assert len(actions) <= ACTION_HORIZON
    for step in actions:
        assert isinstance(step, dict)
        assert step, "Per-step action dict must be non-empty"
        # The default ``_resolve_joint_keys`` falls back to ``joint_<i>``
        # when ``set_robot_state_keys`` was not called.
        assert all(k.startswith("joint_") for k in step.keys())
        assert all(isinstance(v, float) for v in step.values())


def test_chunked_yield_drains_full_trajectory(policy: CuroboPolicy) -> None:
    """Repeated calls drain the cached trajectory ``action_horizon`` rows
    at a time without re-planning until the cache empties.

    This pins the chunked-action contract from the acceptance criteria:

      > Cache the full trajectory; yield ``action_horizon``-sized chunks
      > per ``get_actions`` call to match the existing 50Hz execution loop.
    """
    home_state = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    target_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

    # Force a fresh plan via reset() so the cache is empty.
    policy.reset()

    yielded = 0
    chunks = 0
    for _ in range(50):  # generous upper bound — typical traj is ~100 waypoints.
        actions = policy.get_actions_sync(
            observation_dict={"observation.state": home_state},
            instruction="",
            target_pose=target_pose,
        )
        chunks += 1
        yielded += len(actions)
        # On the final chunk the trajectory may be shorter than horizon.
        if len(actions) < ACTION_HORIZON:
            break

    assert chunks >= 1
    assert yielded >= ACTION_HORIZON


def test_failed_plan_raises_runtime_error(policy: CuroboPolicy) -> None:
    """An obviously unreachable pose surfaces as a ``RuntimeError`` with
    cuRobo's status code, not as silent zero actions.

    Pose 99 m above the base is unreachable for any ground-mounted arm,
    so cuRobo returns ``success=False``."""
    policy.reset()
    home_state = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    with pytest.raises(RuntimeError):
        policy.get_actions_sync(
            observation_dict={"observation.state": home_state},
            instruction="",
            target_pose=[0.0, 0.0, 99.0, 1.0, 0.0, 0.0, 0.0],
        )
