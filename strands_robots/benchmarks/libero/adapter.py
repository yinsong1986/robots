"""``LiberoAdapter`` - :class:`BenchmarkProtocol` driven by a LIBERO BDDL file.

LIBERO is a suite of ~130 tabletop manipulation tasks built around a Franka
Panda. Each task ships as a BDDL problem file + an MJCF scene. The adapter
compiles the BDDL ``:goal`` into a sparse success predicate via
:mod:`strands_robots.benchmarks.libero.bddl_parser` and drives the scene
through the standard :class:`BenchmarkProtocol` lifecycle:

1. :meth:`on_episode_start` - optional ``sim.load_scene(scene_path)``, then
   the base ``BenchmarkProtocol`` compatibility check (Panda-only), then
   per-episode jitter of ``(:init ...)`` object positions.
2. :meth:`on_step` - sparse: ``StepInfo(reward=0.0, done=False)``. LIBERO
   does not define a dense reward.
3. :meth:`is_success` - walks the compiled ``:goal`` predicate tree against
   the current sim state.

**Panda-only by design.** LIBERO's scene MJCFs ``<include>`` Panda geometry
and BDDL predicates reference Panda gripper body names
(``robot0_gripper_*``). Retargeting to a different robot would require
rewriting every BDDL predicate against different body names and is out of
scope for this adapter. Subclass :class:`LiberoAdapter` and override
:attr:`supported_robots` + :attr:`default_robot` if you know what you're
doing.

The adapter does NOT require the ``libero`` Python package to be installed -
only a BDDL string / file and (optionally) an MJCF scene path. The
:func:`strands_robots.benchmarks.libero.suite.load_libero_suite` helper is
the one that pulls in the upstream package to discover task files.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.benchmarks.libero.bddl_parser import (
    BDDLParseError,
    BDDLProblem,
    Node,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo

if TYPE_CHECKING:
    import random

    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)


class LiberoAdapter(BenchmarkProtocol):
    """Panda-only :class:`BenchmarkProtocol` driven by a parsed LIBERO BDDL task.

    Construct with a BDDL file path (``from_file``) or raw BDDL text
    (``from_text``) - direct ``__init__`` is for advanced use when you
    already have a :class:`BDDLProblem`.

    Example::

        from strands_robots.benchmarks.libero import LiberoAdapter

        adapter = LiberoAdapter.from_file(
            "libero/tasks/libero_spatial/pick_up_the_red_cube.bddl",
            scene_path="libero/assets/scenes/libero_spatial_scene.xml",
        )
        sim.register_benchmark("pick-red-cube", adapter)
        sim.evaluate_benchmark("pick-red-cube", policy_provider="mock",
                               n_episodes=10, seed=42)

    Attributes:
        max_steps: Default 300 (LIBERO convention). Override per-task by
            passing ``max_steps=`` to the constructor or mutating the
            attribute after construction.
        problem: The parsed :class:`BDDLProblem`. Stored for introspection
            (agents may read ``problem.language`` as the instruction).
    """

    max_steps: int = 300
    supported_robots_list: list[str] = ["panda"]
    default_robot_name: str = "panda"

    #: Cameras the ``libero_panda`` ``Gr00tDataConfig`` expects to find on the
    #: sim. Names match the bare keys of its ``video_keys`` (``video.image``
    #: → ``image``, ``video.wrist_image`` → ``wrist_image``) so the policy's
    #: ``_build_service_observation`` picks them up directly without an
    #: explicit ``observation_mapping``.
    #:
    #: Poses are world-fixed approximations of LIBERO's RoboSuite-conventional
    #: views (third-person "agentview" + wrist view). The real LIBERO setup
    #: parents ``robot0_eye_in_hand_image`` to the gripper body; that requires
    #: a proper LIBERO scene MJCF (which the upstream pip package does NOT
    #: ship). Until those scene XMLs are wired in via ``scene_path=``, the
    #: wrist camera here is a *static* top-down workspace view - the model
    #: still gets *an* image, but it doesn't track the end-effector. Override
    #: by passing ``cameras={"wrist_image": {"position": [...], ...}}`` to
    #: the constructor.
    LIBERO_CAMERAS: dict[str, dict[str, Any]] = {
        "image": {
            "position": [1.0, 0.0, 1.5],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
        "wrist_image": {
            "position": [0.0, 0.0, 1.4],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
    }

    def __init__(
        self,
        problem: BDDLProblem,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.02,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
    ):
        """Construct from a pre-parsed :class:`BDDLProblem`.

        Args:
            problem: Parsed BDDL problem with a non-``None`` ``goal``.
            scene_path: Optional MJCF to ``sim.load_scene()`` on each
                episode start. ``None`` → assume scene is pre-loaded.
            max_steps: Override the class-level 300.
            init_jitter: Per-episode ±jitter (metres) applied to xy of every
                object referenced by ``(:init (on A B))`` clauses. Set to 0
                to disable jitter.
            install_cameras: When ``True`` (default), install the cameras
                in :attr:`LIBERO_CAMERAS` (or ``cameras`` override) on
                episode start. Set to ``False`` if your scene MJCF already
                declares the cameras the policy needs - the adapter will
                skip the install step entirely.
            cameras: Override / extend :attr:`LIBERO_CAMERAS`. Keyed by
                camera name, each value is forwarded as ``**kwargs`` to
                :meth:`Simulation.add_camera`. Passing an empty dict
                disables camera installation regardless of
                ``install_cameras``.
            eef_body_name: MuJoCo body name whose pose is read for the
                LIBERO ``state.x/y/z/roll/pitch/yaw`` keys. Default
                ``"hand"`` matches MuJoCo Menagerie's Panda. Use
                ``"<robot_name>/hand"`` in multi-Panda scenes (the
                lookup goes through the namespace-aware
                :meth:`Simulation.get_body_state`, so the bare name is
                usually fine).
            gripper_joint_name: Joint name whose ``qpos`` is read for the
                LIBERO ``state.gripper`` key. Default ``"finger_joint1"``
                matches the Menagerie Panda; the second finger
                (``finger_joint2``) mirrors via an MJCF equality
                constraint, so reading just one is sufficient.
            inject_eef_state: When ``True`` (default), the adapter's
                :meth:`augment_observation` injects ``x`` / ``y`` / ``z``
                / ``roll`` / ``pitch`` / ``yaw`` / ``gripper`` keys
                into the per-step observation so the ``libero_panda``
                ``Gr00tDataConfig`` finds them. Set to ``False`` when the
                sim already exposes those keys (e.g. via a custom
                ``observation_mapping`` on the policy or a backend that
                returns Cartesian state natively).

        Raises:
            ValueError: If ``problem.goal`` is ``None``.
        """
        if problem.goal is None:
            raise ValueError(f"LiberoAdapter: BDDL problem {problem.name!r} has no (:goal ...) block")
        self.problem = problem
        self.scene_path = scene_path
        self._init_jitter = float(init_jitter)
        if self._init_jitter < 0:
            raise ValueError(f"init_jitter must be >= 0, got {init_jitter}")
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._install_cameras = bool(install_cameras)
        # Snapshot the camera config at construction time so subsequent
        # mutations to LIBERO_CAMERAS don't leak across instances.
        self._cameras: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in cameras.items()}
            if cameras is not None
            else {k: dict(v) for k, v in self.LIBERO_CAMERAS.items()}
        )
        self._eef_body_name = str(eef_body_name)
        self._gripper_joint_name = str(gripper_joint_name)
        self._inject_eef_state = bool(inject_eef_state)
        self._success_fn: Callable[[SimEngine], bool] = compile_goal(problem.goal)

    # Construction helpers

    @classmethod
    def from_file(
        cls,
        bddl_path: str | Path,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.02,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
    ) -> LiberoAdapter:
        """Parse a ``.bddl`` file from disk and build an adapter.

        Raises :class:`FileNotFoundError` / :class:`BDDLParseError` on bad
        input - callers that want structured error dicts should catch and
        convert.
        """
        problem = parse_bddl_file(bddl_path)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            gripper_joint_name=gripper_joint_name,
            inject_eef_state=inject_eef_state,
        )

    @classmethod
    def from_text(
        cls,
        bddl_text: str,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.02,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
    ) -> LiberoAdapter:
        """Parse a BDDL string directly - useful in tests."""
        problem = parse_bddl(bddl_text)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            gripper_joint_name=gripper_joint_name,
            inject_eef_state=inject_eef_state,
        )

    # BenchmarkProtocol interface

    @property
    def supported_robots(self) -> list[str]:
        return list(self.supported_robots_list)

    @property
    def default_robot(self) -> str:
        return self.default_robot_name

    @property
    def instruction(self) -> str:
        """Language instruction from the BDDL ``:language`` clause, or ``""``."""
        return self.problem.language or ""

    def on_episode_start(self, sim: SimEngine, rng: random.Random) -> None:
        """Load the declared scene (if any), validate Panda, install cameras, then jitter.

        Order matters:

        1. ``load_scene`` (if ``scene_path`` set) - so the base
           compatibility check sees the scene's Panda rather than reporting
           "sim is empty → load default_robot".
        2. ``super().on_episode_start`` - base compat check + auto-load
           ``default_robot`` if the sim is empty.
        3. ``_install_libero_cameras`` - inject the cameras the
           ``libero_panda`` ``Gr00tDataConfig`` expects (``image`` /
           ``wrist_image``). MUST happen after the robot is loaded so the
           cameras render against a populated scene.
        4. ``_apply_init_jitter`` - per-episode RNG-seeded ±jitter to
           init-subject bodies.
        """
        if self.scene_path:
            load_scene = getattr(sim, "load_scene", None)
            if load_scene is None:
                logger.warning(
                    "LiberoAdapter: sim has no load_scene(); skipping scene_path=%r",
                    self.scene_path,
                )
            else:
                result = load_scene(self.scene_path)
                if isinstance(result, dict) and result.get("status") == "error":
                    msg = (result.get("content") or [{}])[0].get("text", "")
                    raise RuntimeError(f"LiberoAdapter: load_scene({self.scene_path!r}) failed: {msg}")
        super().on_episode_start(sim, rng)
        if self._install_cameras:
            self._install_libero_cameras(sim)
        if self._init_jitter > 0:
            self._apply_init_jitter(sim, rng)

    def on_step(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> StepInfo:
        """Sparse step: zero reward, never ``done``. Success is detected by
        :meth:`is_success` at the outer eval loop."""
        return StepInfo(reward=0.0, done=False)

    def augment_observation(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject ``x`` / ``y`` / ``z`` / ``roll`` / ``pitch`` / ``yaw`` / ``gripper``
        for the ``libero_panda`` ``Gr00tDataConfig`` schema.

        The ``libero_panda`` data_config declares
        ``state_keys = ["state.x", "state.y", "state.z", "state.roll",
        "state.pitch", "state.yaw", "state.gripper"]``. The policy's
        ``_build_service_observation`` strips the ``state.`` prefix and
        looks up bare keys (``x``, ``y``, …) directly in the robot
        observation. ``Simulation.get_observation()`` only returns
        joint-space readings, so without this hook the server rejects
        every request with ``Server error: State key 'state.x' must be
        in observation``.

        Implementation:

        1. Read end-effector pose via ``sim.get_body_state(self._eef_body_name)``
           (default body ``"hand"`` for MuJoCo Menagerie's Panda).
        2. Convert MuJoCo's ``(w, x, y, z)`` quaternion to extrinsic XYZ
           Euler ``(roll, pitch, yaw)`` to match the LIBERO/RoboSuite
           ``mat2euler(..., axes='sxyz')`` convention the dataset and
           policy were trained on.
        3. Read gripper opening from ``obs[self._gripper_joint_name]``
           (already populated by ``Simulation.get_observation``; default
           ``"finger_joint1"`` matches Menagerie Panda).

        Best-effort: if any source is missing (sim doesn't expose
        ``get_body_state``, body name unknown, gripper joint absent),
        the corresponding key is omitted with a debug log. The original
        observation is returned with the resolved keys merged in - we
        never delete or overwrite an obs key the sim already provided
        (so a backend that natively returns Cartesian state wins).

        Disable this entirely with ``inject_eef_state=False`` on the
        constructor.
        """
        if not self._inject_eef_state:
            return obs

        merged = dict(obs)

        # End-effector pose - via get_body_state which is namespace-aware
        # (the `panda_arm/hand` form works in multi-robot scenes).
        get_body_state = getattr(sim, "get_body_state", None)
        if get_body_state is not None:
            try:
                state_result = get_body_state(body_name=self._eef_body_name)
            except Exception as e:  # noqa: BLE001 - never abort eval on a state lookup
                logger.debug("LiberoAdapter: get_body_state(%r) raised: %s", self._eef_body_name, e)
                state_result = None
            position, quat = _extract_pose(state_result)
            if position is not None:
                # Don't overwrite if a backend already supplied these
                # (e.g. via a custom mapping).
                merged.setdefault("x", float(position[0]))
                merged.setdefault("y", float(position[1]))
                merged.setdefault("z", float(position[2]))
            if quat is not None:
                roll, pitch, yaw = _quat_wxyz_to_rpy_xyz(quat)
                merged.setdefault("roll", roll)
                merged.setdefault("pitch", pitch)
                merged.setdefault("yaw", yaw)
        else:
            logger.debug("LiberoAdapter: sim has no get_body_state(); skipping EEF state injection")

        # Gripper - read from the (already collected) joint observation.
        # The Menagerie Panda's two-finger constraint mirrors finger_joint1
        # to finger_joint2, so reading just the first one is sufficient
        # *as a value* — but the checkpoint was trained on
        # ``robot0_gripper_qpos`` from LIBERO/RoboSuite which is a
        # 2-element array (one qpos per finger), and the server
        # boolean-masks the state vector by the per-key feature dimension.
        # Packing ``gripper`` as a scalar fails with
        # ``boolean index did not match indexed array along dimension 1;
        # dimension is 1 but corresponding boolean dimension is 2``. So
        # mirror the value into a 2-element list to match the trained
        # shape.
        gripper_value = obs.get(self._gripper_joint_name)
        if gripper_value is None:
            # Some backends namespace joint keys; try the suffix match.
            for key, val in obs.items():
                if isinstance(key, str) and key.endswith("/" + self._gripper_joint_name):
                    gripper_value = val
                    break
        if isinstance(gripper_value, (int, float)) and not isinstance(gripper_value, bool):
            merged.setdefault("gripper", [float(gripper_value), float(gripper_value)])
        else:
            logger.debug(
                "LiberoAdapter: gripper joint %r not found in obs; omitting state.gripper",
                self._gripper_joint_name,
            )

        return merged

    def is_success(self, sim: SimEngine) -> bool:
        return bool(self._success_fn(sim))

    # Internals

    def _install_libero_cameras(self, sim: SimEngine) -> None:
        """Inject the cameras the ``libero_panda`` data_config expects.

        Best-effort: the LIBERO ``Gr00tDataConfig`` declares
        ``video_keys = ["video.image", "video.wrist_image"]`` and the policy's
        ``_build_service_observation`` reads those from the robot observation
        as ``obs["image"]`` / ``obs["wrist_image"]``. Without these cameras
        in the sim, every direct-client call to a LIBERO server fails with
        ``Video key 'video.image' must be in observation`` (#148, Failure 1).

        Cameras already present in the sim (declared by a loaded scene MJCF
        that beats us to the name) are skipped silently. Other failures are
        logged at WARNING but never fatal - one missing camera shouldn't
        kill the whole eval.
        """
        add_camera = getattr(sim, "add_camera", None)
        if add_camera is None:
            logger.debug("LiberoAdapter: sim has no add_camera(); skipping camera install")
            return

        # Cheap check for already-installed cameras: most backends expose a
        # ``_world.cameras`` dict. If we can't see it, just try add_camera
        # and let it return its own "already exists" error.
        existing: set[str] = set()
        world = getattr(sim, "_world", None)
        cameras_attr = getattr(world, "cameras", None) if world is not None else None
        if isinstance(cameras_attr, dict):
            existing = set(cameras_attr.keys())

        for cam_name, cam_kwargs in self._cameras.items():
            if cam_name in existing:
                logger.debug("LiberoAdapter: camera %r already in sim; skipping install", cam_name)
                continue
            try:
                result = add_camera(name=cam_name, **cam_kwargs)
            except Exception as e:  # noqa: BLE001 - one bad camera shouldn't kill the eval
                logger.warning("LiberoAdapter: add_camera(%r) raised: %s", cam_name, e)
                continue
            if isinstance(result, dict) and result.get("status") == "error":
                msg = (result.get("content") or [{}])[0].get("text", "")
                # "already exists" is benign - the scene XML beat us to it.
                if "already exists" in msg.lower():
                    logger.debug("LiberoAdapter: camera %r already declared by scene", cam_name)
                else:
                    logger.warning("LiberoAdapter: add_camera(%r) failed: %s", cam_name, msg)

    def _apply_init_jitter(self, sim: SimEngine, rng: random.Random) -> None:
        """Apply ±jitter to xy of every body referenced by ``(:init (on A B))``.

        Best-effort: if the sim doesn't expose ``move_object`` / ``get_body_state``,
        or the body isn't in the scene, silently skip. This matches LIBERO's
        "small random perturbation per episode" convention without requiring
        full BDDL init semantics.
        """
        move_object = getattr(sim, "move_object", None)
        if move_object is None:
            logger.debug("LiberoAdapter: sim has no move_object(); skipping init jitter")
            return
        get_body_state = getattr(sim, "get_body_state", None)
        if get_body_state is None:
            return

        # Gather the set of bodies we want to jitter - BDDL init uses the same
        # Pred grammar, so (on cube_1 table_1) means "jitter cube_1".
        from strands_robots.benchmarks.libero.bddl_parser import Pred as _Pred

        seen: set[str] = set()
        for node in self.problem.init:
            for body in _extract_init_targets(node):
                seen.add(body)
        _ = _Pred  # referenced for clarity; actual test is inside _extract_init_targets

        for body in sorted(seen):
            try:
                state = get_body_state(body_name=body)
            except Exception as e:  # noqa: BLE001 - defensive
                logger.debug("jitter lookup for %r failed: %s", body, e)
                continue
            if not isinstance(state, dict) or state.get("status") != "success":
                continue
            pos = _extract_position(state)
            if pos is None:
                continue
            jx = rng.uniform(-self._init_jitter, self._init_jitter)
            jy = rng.uniform(-self._init_jitter, self._init_jitter)
            new_pos = [pos[0] + jx, pos[1] + jy, pos[2]]
            try:
                move_object(name=body, position=new_pos)
            except Exception as e:  # noqa: BLE001 - jitter failures are not fatal
                logger.debug("jitter apply for %r failed: %s", body, e)


def _extract_init_targets(node: Node) -> list[str]:
    """Return the first-arg body name of every leaf predicate in ``node``.

    Init clauses like ``(on cube_1 table_1)`` and ``(upright bottle_1)``
    share the convention that the first argument is the "subject" body -
    the thing whose position we may want to jitter. Nested
    ``and``/``or``/``not`` are traversed; non-predicates are ignored.
    """
    from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred

    if isinstance(node, Pred):
        return [node.args[0]] if node.args else []
    if isinstance(node, (And, Or)):
        out: list[str] = []
        for c in node.clauses:
            out.extend(_extract_init_targets(c))
        return out
    if isinstance(node, Not):
        return _extract_init_targets(node.clause)
    return []


def _extract_position(state: dict[str, Any]) -> list[float] | None:
    """Pull ``{"json": {"position": [...]}}`` from a status-dict payload."""
    for block in state.get("content", []) or []:
        if isinstance(block, dict) and isinstance(block.get("json"), dict):
            pos = block["json"].get("position")
            if isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos):
                return [float(c) for c in pos]
    return None


def _extract_pose(state: dict[str, Any] | None) -> tuple[list[float] | None, list[float] | None]:
    """Pull ``(position, quaternion_wxyz)`` from a ``get_body_state`` payload.

    Both fields are optional; this returns ``(None, None)`` for any
    error / shape mismatch so the caller can selectively inject just
    the keys it has. The MuJoCo backend always reports both, so in
    the happy path you get both arrays back.
    """
    if not isinstance(state, dict) or state.get("status") != "success":
        return (None, None)
    pos: list[float] | None = None
    quat: list[float] | None = None
    for block in state.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        json_block = block.get("json")
        if not isinstance(json_block, dict):
            continue
        raw_pos = json_block.get("position")
        if isinstance(raw_pos, list) and len(raw_pos) == 3 and all(isinstance(c, (int, float)) for c in raw_pos):
            pos = [float(c) for c in raw_pos]
        raw_quat = json_block.get("quaternion")
        if isinstance(raw_quat, list) and len(raw_quat) == 4 and all(isinstance(c, (int, float)) for c in raw_quat):
            quat = [float(c) for c in raw_quat]
    return (pos, quat)


def _quat_wxyz_to_rpy_xyz(quat_wxyz: list[float]) -> tuple[float, float, float]:
    """MuJoCo ``(w, x, y, z)`` quaternion → extrinsic XYZ Euler ``(roll, pitch, yaw)``.

    Matches RoboSuite/LIBERO's ``mat2euler(..., axes='sxyz')`` convention -
    i.e. rotations applied about the *static* world frame in the order
    X (roll), Y (pitch), Z (yaw). This is also what
    ``scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_euler('xyz')``
    returns (lowercase ``'xyz'`` = extrinsic in scipy).

    Pure numpy / stdlib - **does not import scipy**, which is not a
    declared dependency of strands_robots. Math reference:

        R = R_x(roll) · R_y(pitch) · R_z(yaw)  (extrinsic XYZ)

    For unit quat ``q = (w, x, y, z)``, the rotation-matrix elements
    needed for the canonical extraction are:

        R[0,2] =  2 (xz + wy)        →  sin(pitch)
        R[0,0] =  1 - 2 (y² + z²)
        R[0,1] = -2 (wz - xy)
        R[1,2] = -2 (wx - yz)
        R[2,2] =  1 - 2 (x² + y²)

    Gimbal lock (``|sin(pitch)| ≥ 1 - 1e-6``) collapses roll into yaw;
    we use the ``atan2(R[1,0], R[1,1])`` resolution that matches scipy.

    Returns:
        ``(roll, pitch, yaw)`` in **radians**, each in the principal
        range used by ``atan2`` / ``asin``: ``roll ∈ (-π, π]``,
        ``pitch ∈ [-π/2, π/2]``, ``yaw ∈ (-π, π]``.
    """
    import math

    w, x, y, z = quat_wxyz
    # Clamp argument to asin to handle minor numerical drift on unit quats.
    sin_pitch = max(-1.0, min(1.0, 2.0 * (x * z + w * y)))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) >= 1.0 - 1e-6:
        # Gimbal-lock branch: roll absorbed into yaw.
        roll = 0.0
        yaw = math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    else:
        roll = math.atan2(-2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
        yaw = math.atan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


__all__ = [
    "BDDLParseError",
    "LiberoAdapter",
]
