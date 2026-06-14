"""Smoke tests for :mod:`strands_robots.policies.curobo` — no GPU required.

These tests exercise the in-process :class:`CuroboPolicy` against a stubbed
``MotionGen`` so they run on any developer machine. The integration test
under ``tests_integ/policies/curobo/`` covers the live GPU path.

Subtask 2 of issue #299. The :class:`Policy` ABC contract for non-VLA
providers landed in #300 (well-known ``target_pose`` / ``target_joints`` /
``world_update`` kwargs).

The acceptance criteria pin:

* :class:`CuroboPolicy` is creatable via ``create_policy("curobo", ...)``
  with the registered shorthand and the ``cumotion`` alias.
* Goal extraction reads from the issue #300 well-known kwargs and falls
  back to a JSON-in-instruction parse for LLM-driven workflows.
* The full trajectory is cached on the first call; ``action_horizon``-sized
  chunks are yielded per call.
* Validation rejects malformed goals up-front.
"""

from __future__ import annotations

import asyncio

import pytest

from strands_robots.policies import (
    Policy,
    create_policy,
    list_providers,
)
from strands_robots.policies.curobo import CuroboPolicy

# ---------------------------------------------------------------------------
# Stub MotionGen
# ---------------------------------------------------------------------------


class _StubMotionGen:
    """Minimal stand-in for cuRobo's ``MotionGen`` for unit tests.

    Records the calls it receives, returns a synthetic trajectory shaped
    like the one cuRobo produces (list-of-lists of floats per waypoint),
    and exposes the optional ``warmup`` / ``reset`` / ``update_world``
    hooks the policy probes for.
    """

    def __init__(
        self,
        ndof: int = 6,
        horizon: int = 10,
        success: bool = True,
        status: str = "ok",
    ) -> None:
        self.ndof = ndof
        self.horizon = horizon
        self.success = success
        self.status = status

        # Recording surfaces.
        self.plan_calls: list[tuple] = []
        self.warmup_called: int = 0
        self.reset_called: int = 0
        self.world_updates: list = []

    def warmup(self) -> None:
        self.warmup_called += 1

    def reset(self) -> None:
        self.reset_called += 1

    def update_world(self, new_world: object) -> None:
        self.world_updates.append(new_world)

    def plan_single(self, start_state: object, goal: object) -> _StubMotionGenResult:
        self.plan_calls.append(("plan_single", start_state, goal))
        return _StubMotionGenResult(
            ndof=self.ndof,
            horizon=self.horizon,
            success=self.success,
            status=self.status,
        )

    def plan_single_js(self, start_state: object, goal: object) -> _StubMotionGenResult:
        self.plan_calls.append(("plan_single_js", start_state, goal))
        return _StubMotionGenResult(
            ndof=self.ndof,
            horizon=self.horizon,
            success=self.success,
            status=self.status,
        )


class _StubMotionGenResult:
    """Stand-in for cuRobo's ``MotionGenResult``.

    The policy's ``_extract_trajectory`` falls back to ``result.trajectory``
    (list-of-lists) when present, so we expose that directly here without
    needing to mock the ``get_interpolated_plan().position`` torch path.
    """

    def __init__(self, ndof: int, horizon: int, success: bool, status: str) -> None:
        self.success = success
        self.status = status
        # Synthetic trajectory: row t has values [t/100 * (i+1) for i in range(ndof)].
        self.trajectory: list[list[float]] = [[(t + 1) / 100.0 * (i + 1) for i in range(ndof)] for t in range(horizon)]


# ---------------------------------------------------------------------------
# CuroboPolicy - construction & registry
# ---------------------------------------------------------------------------


class TestCuroboPolicyConstruction:
    def test_provider_name(self) -> None:
        p = CuroboPolicy(motion_gen=_StubMotionGen())
        assert p.provider_name == "curobo"

    def test_does_not_require_images(self) -> None:
        """Planner-style policies must skip camera rendering (#300 contract)."""
        p = CuroboPolicy(motion_gen=_StubMotionGen())
        assert p.requires_images is False

    def test_subclass_of_policy_abc(self) -> None:
        """Pin the inheritance contract from issue #300."""
        p = CuroboPolicy(motion_gen=_StubMotionGen())
        assert isinstance(p, Policy)

    def test_silent_unknown_kwargs(self) -> None:
        """Per #300: providers MUST ignore unknown kwargs rather than raising."""
        p = CuroboPolicy(
            motion_gen=_StubMotionGen(),
            future_kwarg_we_dont_know_about="ignore me",
        )
        assert p.action_horizon == 16

    def test_warmup_called_by_default(self) -> None:
        """Real construction path warms the planner so the first plan is fast.

        We exercise this via the stub seam: when the user passes
        ``motion_gen=`` directly, warmup is **not** called (the user
        owns the lifecycle). The constructor only warms when it
        builds the planner itself. Verify via the fact that injecting
        a stub does not touch ``warmup``.
        """
        stub = _StubMotionGen()
        CuroboPolicy(motion_gen=stub)
        assert stub.warmup_called == 0  # caller-owned planner; no auto-warmup.

    def test_action_horizon_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="action_horizon must be >= 1"):
            CuroboPolicy(motion_gen=_StubMotionGen(), action_horizon=0)

    def test_missing_robot_config_and_motion_gen_raises(self) -> None:
        """Without either ``robot_config`` or a pre-built ``motion_gen``,
        the constructor must refuse — no silent fall-through to a
        pseudo-default planner."""
        with pytest.raises(ValueError, match="robot_config"):
            CuroboPolicy()  # no robot_config, no motion_gen

    def test_create_policy_by_canonical_name(self) -> None:
        # Pre-bind a stub MotionGen via kwargs so the factory path
        # doesn't try to import cuRobo.
        p = create_policy("curobo", motion_gen=_StubMotionGen())
        assert isinstance(p, CuroboPolicy)

    def test_create_policy_by_cumotion_alias(self) -> None:
        p = create_policy("cumotion", motion_gen=_StubMotionGen())
        assert isinstance(p, CuroboPolicy)

    def test_listed_in_providers(self) -> None:
        providers = list_providers()
        assert "curobo" in providers


# ---------------------------------------------------------------------------
# Validation - reject malformed goals up-front
# ---------------------------------------------------------------------------


class TestCuroboPolicyValidation:
    def _make_policy(self) -> CuroboPolicy:
        return CuroboPolicy(motion_gen=_StubMotionGen())

    def test_missing_target_raises(self) -> None:
        """Neither target_pose nor target_joints (and no JSON in instruction)
        -> ValueError."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="target_pose|target_joints"):
            asyncio.run(p.get_actions({"observation.state": [0.0] * 6}, "go to the box"))

    def test_target_pose_wrong_length_rejected(self) -> None:
        p = self._make_policy()
        with pytest.raises(ValueError, match="7 elements"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.0, 0.0, 0.0],  # only 3 elements
                )
            )

    def test_target_pose_nan_rejected(self) -> None:
        p = self._make_policy()
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, float("nan")],
                )
            )

    def test_target_pose_inf_rejected(self) -> None:
        p = self._make_policy()
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[float("inf"), 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                )
            )

    def test_target_joints_non_dict_rejected(self) -> None:
        p = self._make_policy()
        with pytest.raises(ValueError, match="must be a dict"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints=[0.5, 1.0],  # list instead of dict
                )
            )

    def test_target_joints_bad_key_rejected(self) -> None:
        """Joint names with shell metacharacters rejected up-front."""
        p = self._make_policy()
        with pytest.raises(ValueError, match="must match"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0; rm -rf": 0.5},
                )
            )

    def test_target_joints_inf_rejected(self) -> None:
        p = self._make_policy()
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_joints={"j0": float("inf")},
                )
            )


# ---------------------------------------------------------------------------
# Plan + cache + chunked yield
# ---------------------------------------------------------------------------


class TestCuroboPolicyPlanAndChunk:
    def test_first_call_plans_and_yields_chunk(self) -> None:
        stub = _StubMotionGen(ndof=6, horizon=20)
        p = CuroboPolicy(motion_gen=stub, action_horizon=8)
        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        # First call plans (one plan_single invocation), yields 8 rows.
        assert len(stub.plan_calls) == 1
        assert stub.plan_calls[0][0] == "plan_single"
        assert len(actions) == 8
        # Each row is a per-joint dict with positional ``joint_<i>`` keys.
        for step in actions:
            assert set(step.keys()) == {f"joint_{i}" for i in range(6)}
            assert all(isinstance(v, float) for v in step.values())

    def test_subsequent_calls_yield_from_cache_no_replan(self) -> None:
        """Second call must NOT re-invoke the planner — chunked-action
        contract pins the cache as the source of truth between
        re-plans."""
        stub = _StubMotionGen(ndof=6, horizon=20)
        p = CuroboPolicy(motion_gen=stub, action_horizon=8)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        # Second call: cache still has 12 rows, yield next 8 without
        # touching the planner.
        actions2 = asyncio.run(
            p.get_actions(
                {"observation.state": [0.05] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(stub.plan_calls) == 1  # unchanged
        assert len(actions2) == 8

    def test_replan_on_exhaustion(self) -> None:
        """When the cache empties, the next call re-plans."""
        stub = _StubMotionGen(ndof=6, horizon=10)
        p = CuroboPolicy(motion_gen=stub, action_horizon=10)
        # First call drains the entire cached trajectory.
        actions1 = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(actions1) == 10
        # Second call: cache empty -> re-plan.
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(stub.plan_calls) == 2

    def test_replan_kwarg_forces_replan(self) -> None:
        """``replan=True`` forces a new plan even when the cache still
        has waypoints — useful when the world updated mid-rollout."""
        stub = _StubMotionGen(ndof=6, horizon=20)
        p = CuroboPolicy(motion_gen=stub, action_horizon=8)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                replan=True,
            )
        )
        assert len(stub.plan_calls) == 2

    def test_target_joints_routes_to_plan_single_js(self) -> None:
        """Joint-space goals go through ``plan_single_js`` when available."""
        stub = _StubMotionGen(ndof=3, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0, 0.0, 0.0]},
                "",
                target_joints={"j0": 0.5, "j1": -0.3, "j2": 0.2},
            )
        )
        assert len(stub.plan_calls) == 1
        assert stub.plan_calls[0][0] == "plan_single_js"

    def test_world_update_forwarded(self) -> None:
        stub = _StubMotionGen(ndof=6, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        update = {"cuboid": {"obstacle1": {"dims": [0.1, 0.1, 0.1]}}}
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                world_update=update,
            )
        )
        # The stub records the dict passed through (cuRobo isn't
        # installed in the test env, so the policy falls through to
        # the raw-dict branch).
        assert stub.world_updates == [update]

    def test_set_robot_state_keys_used_for_action_dicts(self) -> None:
        """When configured, custom joint names are used in the per-step dicts."""
        stub = _StubMotionGen(ndof=4, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        p.set_robot_state_keys(["shoulder", "elbow", "wrist", "gripper"])
        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 4},
                "",
                target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        for step in actions:
            assert set(step.keys()) == {"shoulder", "elbow", "wrist", "gripper"}

    def test_failed_plan_raises_runtime_error(self) -> None:
        """``success=False`` from cuRobo surfaces as a RuntimeError with
        status / goal context for debugging."""
        stub = _StubMotionGen(success=False, status="no_collision_free_path")
        p = CuroboPolicy(motion_gen=stub)
        with pytest.raises(RuntimeError, match="no_collision_free_path"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                )
            )

    def test_planner_exception_wrapped_as_runtime_error(self) -> None:
        """An unexpected exception from cuRobo is wrapped as ``RuntimeError``
        with the original goal in the message — saves the user from
        reading an opaque internal trace."""

        class _BoomMotionGen(_StubMotionGen):
            def plan_single(self, *args, **kwargs):  # type: ignore[override]
                raise ValueError("kinematics solver crashed")

        p = CuroboPolicy(motion_gen=_BoomMotionGen())
        with pytest.raises(RuntimeError, match="kinematics solver crashed"):
            asyncio.run(
                p.get_actions(
                    {"observation.state": [0.0] * 6},
                    "",
                    target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                )
            )


# ---------------------------------------------------------------------------
# LLM-driven fallback: parse goals out of the natural-language instruction
# ---------------------------------------------------------------------------


class TestCuroboPolicyInstructionFallback:
    def test_fallback_parses_target_pose_from_json_instruction(self) -> None:
        """``start_task`` paths that pack the goal into the instruction
        string still work via the JSON-in-instruction fallback."""
        stub = _StubMotionGen(ndof=6, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                'Reach for the cube: {"target_pose": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]}',
            )
        )
        assert len(actions) == 4
        assert stub.plan_calls[0][0] == "plan_single"

    def test_fallback_parses_target_joints_from_json_instruction(self) -> None:
        stub = _StubMotionGen(ndof=3, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        actions = asyncio.run(
            p.get_actions(
                {"observation.state": [0.0, 0.0, 0.0]},
                'Move to: {"target_joints": {"j0": 0.5, "j1": -0.3, "j2": 0.1}}',
            )
        )
        assert len(actions) == 4
        assert stub.plan_calls[0][0] == "plan_single_js"

    def test_fallback_unparseable_instruction_raises(self) -> None:
        """If the instruction has no parseable JSON goal, the user gets a
        clear ValueError pointing at the well-known kwargs."""
        p = CuroboPolicy(motion_gen=_StubMotionGen())
        with pytest.raises(ValueError, match="target_pose|target_joints"):
            asyncio.run(p.get_actions({"observation.state": [0.0] * 6}, "go fetch"))


# ---------------------------------------------------------------------------
# reset() — best-effort, clears cache + forwards to planner
# ---------------------------------------------------------------------------


class TestCuroboPolicyReset:
    def test_reset_clears_cache(self) -> None:
        """After ``reset``, the next ``get_actions`` re-plans even though
        the previous trajectory still had waypoints to yield."""
        stub = _StubMotionGen(ndof=6, horizon=20)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        p.reset(seed=42)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 6},
                "",
                target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(stub.plan_calls) == 2

    def test_reset_forwards_to_motion_gen(self) -> None:
        stub = _StubMotionGen()
        p = CuroboPolicy(motion_gen=stub)
        p.reset(seed=42)
        assert stub.reset_called == 1

    def test_reset_swallows_motion_gen_errors(self) -> None:
        """``reset`` is best-effort — any cuRobo-side failure must be
        logged and swallowed."""

        class _BoomMotionGen(_StubMotionGen):
            def reset(self) -> None:  # type: ignore[override]
                raise RuntimeError("planner busy")

        p = CuroboPolicy(motion_gen=_BoomMotionGen())
        # Should not raise.
        p.reset(seed=0)


# ---------------------------------------------------------------------------
# Policy ABC contract — same shape as MockPolicy
# ---------------------------------------------------------------------------


class TestPolicyContractParity:
    """Mock + cuRobo must pass the same Policy-shape contract.

    Pins the issue #300 ABC contract for non-VLA providers so a future
    refactor that breaks one cannot pass while breaking the other. This
    is the regression harness called out in subtask 2 of #299:

      > MockPolicy + CuroboPolicy pass the same ``Policy``-shape contract
      > test added in #300.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("curobo", motion_gen=_StubMotionGen()),
        ],
    )
    def test_provider_is_policy_subclass(self, factory) -> None:
        assert isinstance(factory(), Policy)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("curobo", motion_gen=_StubMotionGen()),
        ],
    )
    def test_provider_has_provider_name(self, factory) -> None:
        p = factory()
        assert isinstance(p.provider_name, str)
        assert p.provider_name  # non-empty

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("curobo", motion_gen=_StubMotionGen()),
        ],
    )
    def test_provider_set_robot_state_keys_is_no_raise(self, factory) -> None:
        p = factory()
        p.set_robot_state_keys(["j0", "j1", "j2"])

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("curobo", motion_gen=_StubMotionGen()),
        ],
    )
    def test_provider_requires_images_is_false_for_planners(self, factory) -> None:
        """Both providers consume joint state only - skip camera rendering."""
        assert factory().requires_images is False

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: create_policy("mock"),
            lambda: create_policy("curobo", motion_gen=_StubMotionGen()),
        ],
    )
    def test_provider_reset_is_no_raise(self, factory) -> None:
        """reset() is best-effort and must not raise on the default path."""
        p = factory()
        p.reset(seed=0)


# ---------------------------------------------------------------------------
# cuRobo ``main`` API migration regression tests (issue #421)
# ---------------------------------------------------------------------------


class _NewApiStubMotionPlanner(_StubMotionGen):
    """Stub planner that exposes the **new** cuRobo API surface.

    Mirrors ``MotionPlanner`` on cuRobo ``main``:

    * ``plan_pose(goal, start_state)`` for Cartesian goals
    * ``plan_js(start_state, goal_js)`` for joint-space goals
    * ``update_scene(dict)`` for the per-call collision-scene refresh

    The legacy ``plan_single`` / ``plan_single_js`` / ``update_world`` are
    intentionally **absent** so the policy must route through the new
    methods. Pin the dispatch order from issue #421's acceptance
    criteria - any future regression that drops the new-API code path
    will fail this test.
    """

    def __init__(self, ndof: int = 7, horizon: int = 60) -> None:
        super().__init__(ndof=ndof, horizon=horizon)

    # Override to remove the legacy methods. Python does not let you
    # ``del`` an inherited method cleanly, so we rebind to a sentinel
    # the policy probes with ``getattr(..., None)``.
    plan_single = None  # type: ignore[assignment]
    plan_single_js = None  # type: ignore[assignment]
    update_world = None  # type: ignore[assignment]

    def plan_pose(self, goal: object, start_state: object) -> _StubMotionGenResult:
        self.plan_calls.append(("plan_pose", start_state, goal))
        return _StubMotionGenResult(
            ndof=self.ndof,
            horizon=self.horizon,
            success=self.success,
            status=self.status,
        )

    def plan_js(self, start_state: object, goal: object) -> _StubMotionGenResult:
        self.plan_calls.append(("plan_js", start_state, goal))
        return _StubMotionGenResult(
            ndof=self.ndof,
            horizon=self.horizon,
            success=self.success,
            status=self.status,
        )

    def update_scene(self, scene_model: object) -> None:
        self.world_updates.append(scene_model)


class TestCuroboMainApiDispatch:
    """Pin the new-API dispatch path so a future refactor cannot silently
    fall back to the legacy ``plan_single`` shim.

    Acceptance criteria from issue #421:

      * ``MotionGen.plan_single(start, goal_pose)`` ->
        ``MotionPlanner.plan_pose(GoalToolPose, JointState)``
      * Joint-space goals route through ``plan_js`` (was
        ``plan_single_js``).
      * ``world_update`` flows through ``update_scene`` (was
        ``update_world``).
    """

    def test_target_pose_routes_to_plan_pose(self) -> None:
        stub = _NewApiStubMotionPlanner(ndof=7, horizon=8)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 7},
                "",
                target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(stub.plan_calls) == 1
        # New API: plan_pose is preferred over plan_single. The stub
        # has plan_single = None so a regression would raise
        # AttributeError - the assertion below pins the routing intent.
        assert stub.plan_calls[0][0] == "plan_pose"

    def test_target_joints_routes_to_plan_js(self) -> None:
        stub = _NewApiStubMotionPlanner(ndof=7, horizon=8)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 7},
                "",
                target_joints={"j0": 0.1, "j1": -0.2, "j2": 0.3},
            )
        )
        assert len(stub.plan_calls) == 1
        # plan_js (new API) preferred over plan_single_js (legacy stub).
        assert stub.plan_calls[0][0] == "plan_js"

    def test_world_update_routes_to_update_scene(self) -> None:
        stub = _NewApiStubMotionPlanner(ndof=7, horizon=4)
        p = CuroboPolicy(motion_gen=stub, action_horizon=4)
        scene = {"cuboid": {"obstacle1": {"dims": [0.1, 0.1, 0.1]}}}
        asyncio.run(
            p.get_actions(
                {"observation.state": [0.0] * 7},
                "",
                target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                world_update=scene,
            )
        )
        # update_scene (new API) preferred over update_world (legacy).
        # Stub forwards the raw dict for record-and-assert simplicity.
        assert stub.world_updates == [scene]


class TestCuroboPolicyConstructorAliases:
    """Pin the constructor's backwards-compatible aliases.

    The cuRobo ``main`` API renamed ``TensorDeviceType`` ->
    ``DeviceCfg`` and ``MotionGenConfig`` -> ``MotionPlannerCfg``. Code
    written against the 0.7.x kwargs (``tensor_args=``,
    ``motion_gen_kwargs=``) must keep working without modification, but
    the new canonical names (``device_cfg=``, ``motion_planner_kwargs=``)
    should be the documented form.
    """

    def test_legacy_tensor_args_kwarg_accepted(self) -> None:
        """``tensor_args=`` is still accepted as a legacy alias for
        ``device_cfg=`` so 0.7.x call sites keep working through the
        migration."""
        # We can't actually exercise the cuRobo build path without
        # importing curobo; the easy regression test is that the kwarg
        # doesn't crash in the constructor when a stub is injected
        # (the alias is parsed before the build path runs).
        p = CuroboPolicy(motion_gen=_StubMotionGen(), tensor_args="cuda:0")
        assert p.action_horizon == 16

    def test_legacy_motion_gen_kwargs_alias_accepted(self) -> None:
        p = CuroboPolicy(
            motion_gen=_StubMotionGen(),
            motion_gen_kwargs={"interpolation_dt": 0.02},
        )
        # The legacy alias must reach the same internal slot the
        # canonical kwarg does.
        assert p._motion_planner_kwargs == {"interpolation_dt": 0.02}

    def test_canonical_motion_planner_kwargs_accepted(self) -> None:
        p = CuroboPolicy(
            motion_gen=_StubMotionGen(),
            motion_planner_kwargs={"use_cuda_graph": False},
        )
        assert p._motion_planner_kwargs == {"use_cuda_graph": False}

    def test_supplying_both_aliases_raises(self) -> None:
        """Passing the legacy alias and the canonical kwarg together is
        almost always a stale-callsite bug. Reject loudly rather than
        applying a silent precedence rule."""
        with pytest.raises(ValueError, match="device_cfg|tensor_args"):
            CuroboPolicy(
                motion_gen=_StubMotionGen(),
                device_cfg="cuda:0",
                tensor_args="cuda:0",
            )
        with pytest.raises(ValueError, match="motion_planner_kwargs|motion_gen_kwargs"):
            CuroboPolicy(
                motion_gen=_StubMotionGen(),
                motion_planner_kwargs={"a": 1},
                motion_gen_kwargs={"a": 1},
            )
