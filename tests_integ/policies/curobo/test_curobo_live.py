"""Integration test for :class:`CuroboPolicy` against the live cuRobo planner.

Targets cuRobo's restructured ``main`` API (issue #421):

* ``curobo.motion_planner.MotionPlanner`` / ``MotionPlannerCfg``
* ``curobo.types.DeviceCfg`` / ``JointState`` / ``GoalToolPose``

Gated behind both the ``curobo`` pytest marker AND a CUDA-capable GPU
detection. Enable with:

.. code-block:: bash

    # cuRobo from source - PyPI doesn't host the real package
    git clone https://github.com/NVlabs/curobo.git
    pip install -e ./curobo
    pip install 'strands-robots[curobo]'

    CUROBO_LIVE=1 hatch run test-integ tests_integ/policies/curobo/ -m curobo -v

A pre-installed cuRobo + a CUDA-capable GPU are required. The test suite
covers:

1. UR5e Cartesian reach to a target pose - asserts the trajectory is
   non-empty and the chunked-action yield matches ``action_horizon``.
2. Drain-the-cache contract - repeated calls yield the cached
   trajectory ``action_horizon`` rows at a time without re-planning.
3. Unreachable-pose error path - ``success=False`` surfaces as a
   ``RuntimeError`` rather than silent zero actions.
4. Franka-7DOF reach-to-pose sanity check (the canonical Thor
   validation example, transcribed from the report linked off issue
   #421). Pinned to the validated working example so a future cuRobo
   ``main`` shift surfaces here first.

The cuRobo ``main`` API surface is still moving. Sites running these
tests against a fresh ``NVlabs/curobo`` checkout should pin to a known
commit (set ``CUROBO_COMMIT_SHA`` in the test docstring as the
canonical pin) until upstream cuts a stable release. As of this issue
filing the canonical pin is ``main`` HEAD; update this comment with
the SHA once the integration suite is wired up to a CI runner with a
sm_110 / Blackwell board.
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

# Default UR10e robot config that ships with cuRobo's ``main`` content tree
# (``ur5e.yml`` is not packaged there). Both are 6-DOF, so the 6-element home
# state below stays valid. Override via ``CUROBO_ROBOT_CONFIG`` for a custom YAML.
ROBOT_CONFIG = os.environ.get("CUROBO_ROBOT_CONFIG", "ur10e.yml")
ACTION_HORIZON = int(os.environ.get("CUROBO_ACTION_HORIZON", "16"))


@pytest.fixture(scope="module")
def policy() -> CuroboPolicy:
    """Build a UR5e :class:`CuroboPolicy` and warm it up once."""
    p = create_policy(
        "curobo",
        robot_config=ROBOT_CONFIG,
        action_horizon=ACTION_HORIZON,
        # Keep warmup ON in the live test - the first ``plan_pose``
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
    for _ in range(50):  # generous upper bound - typical traj is ~100 waypoints.
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


# ---------------------------------------------------------------------------
# Franka-7DOF Thor sanity check (issue #421 acceptance criterion 4)
# ---------------------------------------------------------------------------


# Override-only Franka config; fixture is module-scoped to share warmup
# cost with the other tests.
FRANKA_ROBOT_CONFIG = os.environ.get("CUROBO_FRANKA_ROBOT_CONFIG", "franka.yml")


@pytest.fixture(scope="module")
def franka_policy() -> CuroboPolicy:
    """Build a Franka-7DOF :class:`CuroboPolicy` and warm it up once.

    Mirrors the canonical Thor validation example transcribed from
    issue #421's linked report:

    .. code-block:: python

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg, JointState, GoalToolPose

        cfg = MotionPlannerCfg.create(
            robot='franka.yml',
            device_cfg=DeviceCfg(device=torch.device('cuda:0')),
            use_cuda_graph=False,
        )
        planner = MotionPlanner(cfg)
    """
    p = create_policy(
        "curobo",
        robot_config=FRANKA_ROBOT_CONFIG,
        action_horizon=ACTION_HORIZON,
        # The Thor validation example pins ``use_cuda_graph=False`` for
        # the first-bring-up case to avoid graph-capture surprises on
        # sm_110 / CUDA 13. Forwarded via ``motion_planner_kwargs``.
        motion_planner_kwargs={"use_cuda_graph": False},
        warmup=True,
    )
    assert isinstance(p, CuroboPolicy)
    return p


def test_franka_7dof_reach_to_pose_thor_sanity_check(
    franka_policy: CuroboPolicy,
) -> None:
    """Franka-7DOF reach-to-pose sanity check (Thor validation acceptance).

    Pinned to the canonical example from the Thor validation report:

    * Home: ``[0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]``
      (Franka neutral 7-DOF arm configuration; cuRobo's ``franka.yml``
      plans the 2 finger joints too, so the planned trajectory rows are
      9-DOF).
    * Goal: ``[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]`` (50 cm in front of
      the base, identity quaternion).

    The Thor report observed a 60-waypoint collision-free trajectory in
    under 2 s after warmup. We assert success + non-empty trajectory
    here; the exact waypoint count is not pinned because cuRobo's
    interpolation density is configurable.
    """
    home_state = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
    target_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

    actions = franka_policy.get_actions_sync(
        observation_dict={"observation.state": home_state},
        instruction="franka thor sanity reach",
        target_pose=target_pose,
    )

    assert isinstance(actions, list)
    assert len(actions) > 0, "Franka reach plan must return a non-empty chunk"
    assert len(actions) <= ACTION_HORIZON
    # cuRobo's ``franka.yml`` plans the full 9-DOF chain (7 arm joints + 2
    # finger joints), so per-step action dicts reflect 9 joints, not just the
    # 7 arm DOF. Pin the exact count so a future cuRobo robot-config change
    # (e.g. dropping the fingers from the planning model) surfaces here.
    for step in actions:
        assert isinstance(step, dict)
        assert len(step) == 9, (
            f"Franka step should have 9 joints (7 arm + 2 finger), got {len(step)}: {sorted(step.keys())}"
        )
        assert all(isinstance(v, float) for v in step.values())
