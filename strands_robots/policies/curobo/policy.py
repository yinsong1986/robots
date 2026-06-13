"""CuroboPolicy - in-process collision-aware motion planning via NVIDIA cuRobo.

The policy reads a goal from the well-known ``**kwargs`` keys defined in
issue #300 (``target_pose``, ``target_joints``, ``world_update``), forwards
the request to a :class:`MotionGen` instance running in the same process,
caches the resulting collision-free trajectory, and yields
``action_horizon``-sized chunks per ``get_actions`` call so the 50Hz
execution loop in :class:`~strands_robots.robot.Robot` can stream per-step
joint targets without re-planning.

Construction mirrors the other non-VLA providers — no service mode, since
cuRobo is a CUDA library rather than a sidecar:

.. code-block:: python

    from strands_robots.policies import create_policy

    policy = create_policy(
        "curobo",                                  # alias: "cumotion"
        robot_config="ur5e.yml",
        world_config={"cuboid": {...}},
        action_horizon=16,
    )

    actions = policy.get_actions_sync(
        observation_dict={"observation.state": [0.0] * 6},
        instruction="reach for the red block",     # ignored by planners
        target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
    )

The ``nvidia-curobo`` package on PyPI is an unrelated v0.1 squatter — the
real cuRobo is published only as source on GitHub. Users opt in by
installing it from source before constructing this policy::

    git clone https://github.com/NVlabs/curobo.git
    pip install -e ./curobo

The ``[curobo]`` extra in ``pyproject.toml`` is intentionally empty until
cuRobo publishes a real PyPI wheel. The policy module raises a clear
:class:`ImportError` (via :func:`require_optional`) on construction when
the ``curobo`` Python package is missing.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from strands_robots.policies.base import Policy
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


# Joint-name allowlist regex - matches the same pattern used by
# ``mesh.security.validate_command`` and :class:`MoveIt2Policy` for
# ``target_joints`` keys, so a value the mesh accepts can flow
# end-to-end without a second allowlist mismatch.
_JOINT_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"

# Cap on how big a trajectory we will keep cached. cuRobo's interpolated
# plans are typically O(100s) of waypoints; this bound exists to fail
# loudly if a sidecar / config bug returns a multi-megabyte trajectory
# rather than silently consuming RAM.
_MAX_TRAJECTORY_WAYPOINTS = 100_000


class CuroboPolicy(Policy):
    """In-process cuRobo ``MotionGen`` wrapper.

    The policy is intentionally thin — all motion-planning state lives in
    the :class:`MotionGen` instance owned by this object. Because cuRobo
    runs on a CUDA device, this policy is **not** thread-safe across
    processes; callers that want fan-out should construct one
    ``CuroboPolicy`` per worker.

    Args:
        robot_config: Path to (or in-memory dict of) the cuRobo robot
            description YAML. cuRobo ships configs for many arms under
            ``curobo/content/configs/robot/``; for example ``"ur5e.yml"``
            or ``"franka.yml"``. May also be a pre-loaded dict (skips
            disk I/O — useful for tests and embedded deployments).
        world_config: Initial collision world. cuRobo accepts a dict with
            ``"cuboid"`` / ``"mesh"`` / ``"sphere"`` / ``"capsule"`` keys
            whose values are mappings from name to geometry params; or a
            pre-built ``WorldConfig`` instance, or ``None`` for free-space
            planning. Per-call overrides flow through the ``world_update``
            kwarg on :meth:`get_actions`.
        action_horizon: Number of waypoints to yield per call to
            :meth:`get_actions`. Matches the chunked-action contract used
            by the 50Hz execution loop in :class:`~strands_robots.robot.Robot`.
            Default 16 — same as :class:`~strands_robots.policies.groot.policy.Gr00tPolicy`'s
            inner-loop horizon.
        tensor_args: Optional cuRobo ``TensorDeviceType`` controlling the
            device (e.g. ``"cuda:0"``) and dtype. When omitted, cuRobo's
            default (``cuda:0``, ``fp32``) is used. Passing a string
            (``"cuda:0"`` / ``"cpu"``) is also accepted; it is converted
            internally.
        motion_gen_kwargs: Optional extra kwargs forwarded to
            :meth:`MotionGenConfig.load_from_robot_config` — e.g.
            ``{"interpolation_dt": 0.02, "num_trajopt_seeds": 12}``.
            Reserved for advanced tuning; defaults are sensible.
        motion_gen: Pre-built :class:`MotionGen` instance. When supplied,
            the policy skips its own ``MotionGenConfig.load_from_robot_config``
            + ``MotionGen(...)`` construction. This is the seam unit tests
            use to inject a stub planner without a CUDA device. Production
            callers should leave this ``None`` and pass ``robot_config``.
        warmup: When ``True`` (default), call ``MotionGen.warmup()`` after
            construction so the first ``get_actions`` call is not paying
            JIT-compile cost. Set ``False`` only for tests where warmup
            is expensive or undesirable.
        **kwargs: Forward-compatibility absorber for the smart-string
            resolution path. Per the #300 contract, providers MUST ignore
            unknown kwargs rather than raising.

    Raises:
        ImportError: If ``[curobo]`` extra is not installed and no
            pre-built ``motion_gen`` is supplied.
        ValueError: If ``action_horizon`` < 1, or both ``robot_config``
            and ``motion_gen`` are missing.

    Examples:
        Direct construction::

            from strands_robots.policies.curobo import CuroboPolicy

            policy = CuroboPolicy(
                robot_config="ur5e.yml",
                action_horizon=16,
            )

        Via the registry::

            from strands_robots.policies import create_policy

            policy = create_policy("curobo", robot_config="ur5e.yml")
            policy = create_policy("cumotion", robot_config="ur5e.yml")  # alias

        Per-call goal::

            actions = policy.get_actions_sync(
                observation_dict={"observation.state": [0.0] * 6},
                instruction="",                               # unused
                target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
    """

    def __init__(
        self,
        robot_config: str | dict[str, Any] | None = None,
        world_config: dict[str, Any] | None = None,
        action_horizon: int = 16,
        tensor_args: Any = None,
        motion_gen_kwargs: dict[str, Any] | None = None,
        motion_gen: Any = None,
        warmup: bool = True,
        **kwargs: Any,
    ) -> None:
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")

        self.robot_config = robot_config
        self.world_config = world_config
        self.action_horizon = int(action_horizon)
        self._motion_gen_kwargs = dict(motion_gen_kwargs or {})

        # State for trajectory chunking: cache the full plan, yield
        # ``action_horizon`` rows per call until exhausted, then re-plan
        # on the next call.
        self._robot_state_keys: list[str] = []
        self._cached_trajectory: list[list[float]] = []
        self._cached_cursor: int = 0

        # When the caller supplies a pre-built ``motion_gen`` (e.g. from
        # tests), use it directly. Otherwise build one from the cuRobo
        # APIs. The lazy import lives behind ``require_optional`` so the
        # error message points at the ``[curobo]`` extra cleanly.
        if motion_gen is not None:
            self._motion_gen = motion_gen
        else:
            if robot_config is None:
                raise ValueError(
                    "CuroboPolicy requires either ``robot_config`` (path or dict) "
                    "or a pre-built ``motion_gen`` instance. Pass robot_config="
                    "'ur5e.yml' to load one of the cuRobo built-in configs."
                )
            self._motion_gen = self._build_motion_gen(
                robot_config=robot_config,
                world_config=world_config,
                tensor_args=tensor_args,
            )
            if warmup:
                self._safe_warmup()

        # Per the #300 contract: silently ignore unknown kwargs. The
        # smart-string resolver and ``register_policy`` may fan extra
        # kwargs through this constructor; the policy only consumes a
        # short, documented set.
        if kwargs:
            logger.debug(
                "CuroboPolicy ignoring unknown constructor kwargs: %s",
                sorted(kwargs.keys()),
            )

        logger.info(
            "CuroboPolicy ready [robot_config=%r action_horizon=%d]",
            robot_config if isinstance(robot_config, str) else "<dict>",
            self.action_horizon,
        )

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "curobo"

    @property
    def requires_images(self) -> bool:
        """cuRobo plans from joint state + collision world, never images.

        Returning ``False`` lets the simulation skip camera rendering for
        this provider — same throughput optimisation
        :class:`~strands_robots.policies.moveit2.MoveIt2Policy` and
        :class:`~strands_robots.policies.mock.MockPolicy` expose.
        """
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the joint names this policy emits actions for.

        Used to map the per-row joint values cuRobo returns onto per-joint
        action dicts. When unset, ``get_actions`` falls back to
        ``observation.state`` length and emits ``"joint_<i>"`` keys
        (consistent with :class:`MockPolicy` / :class:`MoveIt2Policy`).
        """
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Drop the cached trajectory and reset cuRobo's per-episode state.

        cuRobo's ``MotionGen.reset()`` clears any seed / partial-plan
        state held by the planner. The ``seed`` argument is not forwarded
        to cuRobo (its trajopt RNG is configured at construction time);
        it is accepted for API parity with the rest of the non-VLA family.

        Best-effort — any failure (planner doesn't expose ``reset``,
        endpoint raises) is logged and swallowed. Eval correctness is
        preserved even when reset is a no-op (the next ``get_actions``
        call re-plans from the current observation).
        """
        # Always clear the cached trajectory so the next ``get_actions``
        # re-plans from the (likely-different) starting state.
        self._cached_trajectory = []
        self._cached_cursor = 0

        # Forward to ``MotionGen.reset`` if available. Older / stub
        # planners may not expose it; that's fine.
        reset_fn = getattr(self._motion_gen, "reset", None)
        if reset_fn is None:
            logger.debug("CuroboPolicy.reset: motion_gen has no reset(); cleared cache only")
            return
        try:
            reset_fn()
            logger.debug("CuroboPolicy.reset: forwarded to motion_gen (seed=%r)", seed)
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info(
                "CuroboPolicy.reset: motion_gen.reset() raised (seed=%r): %s; "
                "continuing without per-episode planner-side reset",
                seed,
                e,
            )

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Plan a collision-free trajectory and yield ``action_horizon`` chunks.

        On the first call (or after the cached trajectory is exhausted),
        this method:

        1. Reads the goal from ``**kwargs`` (or, as a fallback for
           LLM-driven workflows, parses it out of ``instruction``).
        2. Optionally refreshes the cuRobo collision world via
           ``world_update``.
        3. Builds a :class:`JointState` start configuration from
           ``observation_dict["observation.state"]``.
        4. Calls ``MotionGen.plan_single`` and unpacks the interpolated
           plan into a list of joint-target rows.
        5. Caches the full trajectory.

        Every call (cache-miss or cache-hit) returns the next
        ``action_horizon`` rows from the cached trajectory as per-step
        action dicts. When the cache empties, the next call re-plans
        from the current observation.

        Args:
            observation_dict: Robot observation. ``observation.state`` is
                used as the start joint configuration. ``observation.velocity``
                is used as the start joint velocity if present, else zeros.
                The natural-language ``instruction`` is forwarded to the
                fallback :meth:`_parse_target` only if no structured
                kwargs are supplied.
            instruction: Natural-language instruction. Used only as a
                fallback parse target for LLM-driven workflows when
                neither ``target_pose`` nor ``target_joints`` is supplied
                via kwargs. Planner providers consume goals through
                structured kwargs; this fallback exists for API parity
                with the LLM-agent demos.
            **kwargs: Well-known goal payload from #300:

                * ``target_pose`` (``list[float]``):
                  ``[x, y, z, qw, qx, qy, qz]`` in the robot base frame.
                * ``target_joints`` (``dict[str, float]``): joint-space
                  goal keyed by joint name (radians / metres).
                * ``world_update`` (``dict | None``): per-call world
                  refresh for collision-aware planning.
                * ``replan`` (``bool``): force a re-plan even if the
                  cache still has waypoints. Default ``False``.

                Unknown kwargs are silently ignored.

        Returns:
            List of action dicts; up to ``action_horizon`` entries per
            call. May be shorter on the final chunk of a trajectory.

        Raises:
            ValueError: If neither structured goal nor a parseable
                ``instruction`` is provided, or if the goal payload is
                malformed.
            RuntimeError: If cuRobo returns ``success=False`` (no
                collision-free path).
        """
        # 1. Pull goals from kwargs (or fall back to the instruction).
        target_pose = kwargs.get("target_pose")
        target_joints = kwargs.get("target_joints")
        world_update = kwargs.get("world_update")
        replan = bool(kwargs.get("replan", False))

        if target_pose is None and target_joints is None:
            target_pose, target_joints = self._parse_target(instruction)

        if target_pose is None and target_joints is None:
            raise ValueError(
                "CuroboPolicy.get_actions requires at least one of "
                "target_pose=[x,y,z,qw,qx,qy,qz] or target_joints={joint:value}. "
                "These are the well-known kwargs from issue #300; the "
                "natural-language `instruction` is parsed as a fallback "
                "only when it contains a JSON object with a 'target_pose' "
                "or 'target_joints' field."
            )

        # 2. Validation - reject malformed goals up-front.
        if target_pose is not None:
            self._validate_target_pose(target_pose)
        if target_joints is not None:
            self._validate_target_joints(target_joints)

        # 3. Cache check — if the previous trajectory still has unyielded
        # rows AND no new goal forces a replan, just stream the next
        # chunk. Otherwise re-plan from the current state.
        if not self._cache_has_waypoints() or replan:
            joint_state = self._extract_joint_state(observation_dict)
            self._plan_and_cache(
                joint_state=joint_state,
                target_pose=target_pose,
                target_joints=target_joints,
                world_update=world_update,
            )

        return self._next_chunk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_motion_gen(
        self,
        robot_config: str | dict[str, Any],
        world_config: dict[str, Any] | None,
        tensor_args: Any,
    ) -> Any:
        """Construct a :class:`MotionGen` instance from the cuRobo APIs.

        Lives in its own method so the constructor seam stays clean and
        unit tests can override ``__init__`` paths without touching this
        path. Importing cuRobo is gated by :func:`require_optional` so
        the ``[curobo]`` extra is the actionable error.
        """
        require_optional(
            "curobo",
            # cuRobo is NOT on PyPI (the ``nvidia-curobo`` v0.1 package
            # is an unrelated squatter). Real install is from source:
            #   git clone https://github.com/NVlabs/curobo.git
            #   pip install -e ./curobo
            pip_install="-e git+https://github.com/NVlabs/curobo.git#egg=curobo",
            extra="curobo",
            purpose="CuroboPolicy motion planning",
        )
        # Import lazily so module load doesn't pay the CUDA-init cost
        # for users who never construct a CuroboPolicy.
        from curobo.geom.types import WorldConfig  # type: ignore[import-not-found]
        from curobo.types.base import TensorDeviceType  # type: ignore[import-not-found]
        from curobo.wrap.reacher.motion_gen import (  # type: ignore[import-not-found]
            MotionGen,
            MotionGenConfig,
        )

        # Resolve tensor_args. Accept None / str / TensorDeviceType.
        resolved_tensor_args: Any
        if tensor_args is None:
            resolved_tensor_args = TensorDeviceType()
        elif isinstance(tensor_args, str):
            resolved_tensor_args = TensorDeviceType(device=tensor_args)
        else:
            resolved_tensor_args = tensor_args

        # Resolve world_config. Accept None / dict / WorldConfig.
        resolved_world: Any
        if world_config is None:
            resolved_world = WorldConfig()
        elif isinstance(world_config, dict):
            resolved_world = WorldConfig.from_dict(world_config)
        else:
            resolved_world = world_config

        cfg = MotionGenConfig.load_from_robot_config(
            robot_config,
            resolved_world,
            tensor_args=resolved_tensor_args,
            **self._motion_gen_kwargs,
        )
        return MotionGen(cfg)

    def _safe_warmup(self) -> None:
        """Call :meth:`MotionGen.warmup` if available; log on failure."""
        warmup_fn = getattr(self._motion_gen, "warmup", None)
        if warmup_fn is None:
            return
        try:
            warmup_fn()
        except Exception as e:  # noqa: BLE001 - warmup is best-effort
            logger.warning(
                "CuroboPolicy: motion_gen.warmup() raised (%s); first get_actions call will pay JIT-compile cost",
                e,
            )

    def _cache_has_waypoints(self) -> bool:
        return self._cached_cursor < len(self._cached_trajectory)

    def _next_chunk(self) -> list[dict[str, Any]]:
        """Yield the next ``action_horizon`` rows from the cached trajectory."""
        if not self._cache_has_waypoints():
            return []
        end = min(self._cached_cursor + self.action_horizon, len(self._cached_trajectory))
        rows = self._cached_trajectory[self._cached_cursor : end]
        self._cached_cursor = end
        keys = self._resolve_joint_keys(len(rows[0]) if rows else 0)
        actions: list[dict[str, Any]] = []
        for row in rows:
            actions.append({k: float(v) for k, v in zip(keys, row, strict=False)})
        return actions

    def _plan_and_cache(
        self,
        joint_state: list[float] | None,
        target_pose: list[float] | None,
        target_joints: dict[str, float] | None,
        world_update: dict[str, Any] | None,
    ) -> None:
        """Build the cuRobo request, call ``plan_single``, cache the result.

        Imports cuRobo types lazily so the smoke tests that inject a stub
        ``motion_gen`` never touch the cuRobo package at all.
        """
        # Refresh the collision world if requested.
        if world_update is not None:
            self._apply_world_update(world_update)

        # Build the start state. cuRobo's ``JointState.from_position``
        # accepts a tensor; we leave the conversion to cuRobo so this
        # path stays small.
        start_state = self._build_start_state(joint_state)

        # Build the goal. cuRobo's ``plan_single`` takes a Pose for
        # Cartesian goals and a JointState for joint-space goals via
        # ``plan_single_js``. We dispatch on which one was supplied.
        try:
            if target_pose is not None:
                goal = self._build_goal_pose(target_pose)
                result = self._motion_gen.plan_single(start_state, goal)
            else:
                # target_joints is non-None here (validated upstream).
                goal_js = self._build_goal_joint_state(target_joints or {})
                # ``plan_single_js`` is the cuRobo joint-space planner;
                # fall back to ``plan_single`` if a stub planner only
                # exposes the latter (covers test paths cleanly).
                plan_js = getattr(self._motion_gen, "plan_single_js", None)
                if plan_js is None:
                    result = self._motion_gen.plan_single(start_state, goal_js)
                else:
                    result = plan_js(start_state, goal_js)
        except Exception as e:
            # Re-raise as RuntimeError with goal context so the runner
            # gets a clear message instead of an opaque cuRobo trace.
            raise RuntimeError(
                f"CuroboPolicy planning failed: target_pose={target_pose!r}, target_joints={target_joints!r}: {e}"
            ) from e

        if not getattr(result, "success", True):
            status = getattr(result, "status", "unknown")
            raise RuntimeError(
                f"CuroboPolicy planning failed: status={status!r}, "
                f"target_pose={target_pose!r}, target_joints={target_joints!r}"
            )

        trajectory = self._extract_trajectory(result)
        if len(trajectory) > _MAX_TRAJECTORY_WAYPOINTS:
            raise RuntimeError(
                f"CuroboPolicy got {len(trajectory)} waypoints, exceeds "
                f"{_MAX_TRAJECTORY_WAYPOINTS} guard. Likely a misconfigured "
                "interpolation_dt. Refusing to cache."
            )
        self._cached_trajectory = trajectory
        self._cached_cursor = 0

    def _apply_world_update(self, world_update: dict[str, Any]) -> None:
        """Forward a per-call collision-world refresh to cuRobo.

        cuRobo exposes ``MotionGen.update_world(WorldConfig)``; we accept
        a plain dict so callers don't have to import cuRobo types.
        """
        update_world = getattr(self._motion_gen, "update_world", None)
        if update_world is None:
            logger.warning(
                "CuroboPolicy: motion_gen has no update_world(); world_update=%r ignored",
                sorted(world_update.keys()) if isinstance(world_update, dict) else world_update,
            )
            return
        # Lazy import - same pattern as ``_build_motion_gen``. Guarded so
        # the stub-injection test path doesn't import cuRobo at all.
        try:
            from curobo.geom.types import WorldConfig  # type: ignore[import-not-found]

            new_world = WorldConfig.from_dict(world_update)
        except ImportError:
            # Stub injection path: pass the raw dict through and let the
            # stub interpret it.
            new_world = world_update  # type: ignore[assignment]
        update_world(new_world)

    def _build_start_state(self, joint_state: list[float] | None) -> Any:
        """Build a cuRobo :class:`JointState` from a Python list."""
        if joint_state is None:
            # Without a start state, defer to whatever the planner has
            # configured (its retract config, typically). Stub planners
            # ignore the start state anyway.
            return None
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types.base import TensorDeviceType  # type: ignore[import-not-found]
            from curobo.types.state import JointState  # type: ignore[import-not-found]
        except ImportError:
            # Stub-injection path: pass the raw list through. The stub
            # planner is responsible for interpreting it.
            return joint_state

        tensor_args = TensorDeviceType()
        position = torch.tensor(joint_state, **vars(tensor_args)).unsqueeze(0)
        return JointState.from_position(position)

    def _build_goal_pose(self, target_pose: list[float]) -> Any:
        """Build a cuRobo :class:`Pose` from ``[x, y, z, qw, qx, qy, qz]``."""
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types.base import TensorDeviceType  # type: ignore[import-not-found]
            from curobo.types.math import Pose  # type: ignore[import-not-found]
        except ImportError:
            # Stub path - pass the raw list through.
            return target_pose

        tensor_args = TensorDeviceType()
        pos = torch.tensor(target_pose[0:3], **vars(tensor_args)).unsqueeze(0)
        quat = torch.tensor(target_pose[3:7], **vars(tensor_args)).unsqueeze(0)
        return Pose(position=pos, quaternion=quat)

    def _build_goal_joint_state(self, target_joints: dict[str, float]) -> Any:
        """Build a cuRobo :class:`JointState` from a name->value dict."""
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types.base import TensorDeviceType  # type: ignore[import-not-found]
            from curobo.types.state import JointState  # type: ignore[import-not-found]
        except ImportError:
            return target_joints

        tensor_args = TensorDeviceType()
        # Order keys deterministically. If ``set_robot_state_keys`` was
        # called we honour that order; otherwise sorted for stability.
        if self._robot_state_keys and set(target_joints).issubset(set(self._robot_state_keys)):
            keys = [k for k in self._robot_state_keys if k in target_joints]
        else:
            keys = sorted(target_joints.keys())
        position = torch.tensor([target_joints[k] for k in keys], **vars(tensor_args)).unsqueeze(0)
        return JointState.from_position(position, joint_names=keys)

    @staticmethod
    def _extract_trajectory(result: Any) -> list[list[float]]:
        """Pull the joint-position trajectory out of a cuRobo ``MotionGenResult``.

        The canonical way to read the trajectory off a ``MotionGenResult``
        is ``result.get_interpolated_plan()`` which returns a
        :class:`JointState` whose ``position`` is a ``[T, ndof]`` tensor.

        Stub planners may emit a plain ``list[list[float]]`` directly via
        ``result.trajectory`` to keep the test seam lightweight.
        """
        # Stub path first - if the result already exposes a list-of-lists
        # at ``trajectory``, prefer it.
        traj = getattr(result, "trajectory", None)
        if isinstance(traj, list):
            return [[float(v) for v in row] for row in traj]

        # Real cuRobo path: ``get_interpolated_plan().position`` is a
        # ``[T, ndof]`` torch tensor.
        get_plan = getattr(result, "get_interpolated_plan", None)
        if get_plan is None:
            raise RuntimeError(
                "CuroboPolicy: motion_gen result is missing both "
                "``trajectory`` (stub path) and ``get_interpolated_plan`` "
                "(real path); cannot extract waypoints"
            )
        plan = get_plan()
        position = getattr(plan, "position", plan)
        # ``position`` is typically ``torch.Tensor``; ``.cpu().tolist()``
        # produces a list-of-lists. Fall back to ``list(...)`` for stub
        # objects.
        try:
            return [list(map(float, row)) for row in position.cpu().tolist()]
        except AttributeError:
            return [list(map(float, row)) for row in position]

    def _extract_joint_state(self, observation_dict: dict[str, Any]) -> list[float] | None:
        """Pull ``observation.state`` out of the observation dict.

        Accepts list / tuple / numpy array / torch tensor; returns a plain
        Python list of floats so cuRobo's tensor builders get a known
        input shape.
        """
        state = observation_dict.get("observation.state")
        if state is None:
            return None
        try:
            if hasattr(state, "tolist"):
                state = state.tolist()
            return [float(x) for x in state]
        except (TypeError, ValueError) as e:
            logger.warning(
                "CuroboPolicy: failed to extract joint_state from observation.state=%r (%s); "
                "letting planner use its own retract configuration",
                state,
                e,
            )
            return None

    def _resolve_joint_keys(self, n: int) -> list[str]:
        """Resolve the joint key names for an n-element trajectory row.

        If ``set_robot_state_keys`` was called with a matching length,
        use those names; otherwise fall back to positional ``joint_<i>``
        labels (consistent with :class:`MockPolicy` and :class:`MoveIt2Policy`).
        """
        if self._robot_state_keys and len(self._robot_state_keys) == n:
            return list(self._robot_state_keys)
        return [f"joint_{i}" for i in range(n)]

    @staticmethod
    def _parse_target(instruction: str) -> tuple[list[float] | None, dict[str, float] | None]:
        """Best-effort fallback parse of the natural-language instruction.

        For LLM-driven workflows (``Robot.start_task(..., policy_provider="curobo")``),
        the agent may pack a goal into the instruction string as a JSON
        snippet. This helper extracts ``target_pose`` / ``target_joints``
        from such a payload so the LLM-agent demo path works without
        forcing the agent to learn a new kwargs API.

        Returns ``(None, None)`` when no goal is found — the caller will
        then raise :class:`ValueError`.
        """
        if not instruction or not isinstance(instruction, str):
            return None, None
        # Try to find a JSON object embedded in the instruction.
        match = re.search(r"\{.*\}", instruction, re.DOTALL)
        if not match:
            return None, None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None
        if not isinstance(payload, dict):
            return None, None
        target_pose = payload.get("target_pose")
        target_joints = payload.get("target_joints")
        if isinstance(target_pose, list):
            tp: list[float] | None = [float(v) for v in target_pose]
        else:
            tp = None
        if isinstance(target_joints, dict):
            tj: dict[str, float] | None = {str(k): float(v) for k, v in target_joints.items()}
        else:
            tj = None
        return tp, tj

    @staticmethod
    def _validate_target_pose(target_pose: Any) -> None:
        """Validate ``target_pose`` is a 7-element list of finite floats."""
        try:
            poses = list(target_pose)
        except TypeError as e:
            raise ValueError(f"target_pose must be a 7-element list, got {type(target_pose).__name__}") from e
        if len(poses) != 7:
            raise ValueError(f"target_pose must have exactly 7 elements [x,y,z,qw,qx,qy,qz], got {len(poses)}")
        for i, v in enumerate(poses):
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_pose[{i}] must be a number, got {type(v).__name__}") from e
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_pose[{i}]={f!r} must be finite")

    @staticmethod
    def _validate_target_joints(target_joints: Any) -> None:
        """Validate ``target_joints`` is a name->finite-float mapping."""
        if not isinstance(target_joints, dict):
            raise ValueError(f"target_joints must be a dict[str, float], got {type(target_joints).__name__}")
        pattern = re.compile(_JOINT_NAME_PATTERN)
        for k, v in target_joints.items():
            if not isinstance(k, str) or not pattern.match(k):
                raise ValueError(
                    f"target_joints key {k!r} must match {_JOINT_NAME_PATTERN!r} (letters, digits, underscore, hyphen)"
                )
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_joints[{k!r}]={v!r} must be a number") from e
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_joints[{k!r}]={f!r} must be finite")


__all__ = ["CuroboPolicy"]
