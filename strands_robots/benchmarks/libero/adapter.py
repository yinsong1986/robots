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

import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.benchmarks.libero.bddl_parser import (
    BDDLParseError,
    BDDLProblem,
    Node,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.utils import get_base_dir, require_optional

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
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        bddl_source: str | None = None,
        bddl_path: str | None = None,
    ):
        """Construct from a pre-parsed :class:`BDDLProblem`.

        Args:
            problem: Parsed BDDL problem with a non-``None`` ``goal``.
            scene_path: Optional MJCF to ``sim.load_scene()`` on each
                episode start. ``None`` triggers ``auto_generate_scene``
                if enabled (see below).
            max_steps: Override the class-level 300.
            init_jitter: Per-episode ±jitter (metres) applied to xy of every
                object referenced by ``(:init (on A B))`` clauses. Default
                ``0.0`` matches LIBERO's deterministic-reset convention -
                the upstream training data is generated with fixed init
                states per ``(task, seed)`` and the GR00T-LIBERO checkpoint
                expects to see those exact poses (#166). Pass a positive
                value (e.g. ``0.02``) to layer per-episode randomization on
                top of the canonical state - useful for evaluating
                *generalization*, but expect lower nominal success rates
                because the policy is operating slightly out-of-distribution.
            install_cameras: When ``True`` (default), install the cameras
                in :attr:`LIBERO_CAMERAS` (or ``cameras`` override) on
                episode start. Set to ``False`` if your scene MJCF already
                declares the cameras the policy needs - the adapter will
                skip the install step entirely. Generated scenes have their
                cameras renamed to ``image`` / ``wrist_image`` (see
                ``scene_camera_aliases``), so the install step naturally
                no-ops on auto-generated scenes.
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
            auto_generate_scene: When ``True`` (default) AND ``scene_path``
                is ``None``, :meth:`on_episode_start` calls
                :meth:`_generate_scene_from_bddl` to build the scene MJCF
                via the upstream ``libero`` package's procedural
                generator. The generated XML is cached on disk so
                subsequent episodes / processes reuse it without
                re-running ``libero``. Set to ``False`` to keep the
                pre-#164 behaviour of running against a bare Panda when
                no ``scene_path`` is provided.
            scene_cache_dir: Filesystem location for the generated-scene
                cache. Defaults to ``$STRANDS_BASE_DIR/scene_cache/libero/``
                (typically ``~/.strands_robots/scene_cache/libero/``).
                Cache key is SHA256 of the BDDL source so two adapters
                built from the same BDDL share a cached XML.
            scene_camera_aliases: Mapping from MJCF camera name (as
                emitted by LIBERO) to the policy-side observation key
                expected by the ``libero_panda`` data_config. Default
                ``{"agentview": "image", "robot0_eye_in_hand_image": "wrist_image"}``
                renames RoboSuite/LIBERO's two canonical cameras so
                ``Gr00tPolicy._build_service_observation`` finds them by
                bare-key lookup. Pass an empty dict to disable renaming.
            apply_scene_keyframe: When ``True`` (default) AND a scene was
                loaded, :meth:`on_episode_start` restores qpos/qvel to the
                scene's canonical home state AFTER ``super().on_episode_start``
                and any camera install. Two branches:

                * **Preferred** — when ``model.nkey > 0`` (MJCF declares a
                  ``<keyframe>``, which LIBERO-authored hand-written scenes
                  do): calls ``mujoco.mj_resetDataKeyframe(model, data,
                  scene_keyframe_index)``.
                * **Fallback** — when ``model.nkey == 0`` (MJCFs from the
                  procedural :meth:`_generate_scene_from_bddl` path don't
                  carry a keyframe): snapshot-and-restore. The first
                  episode after a scene compile captures
                  ``data.qpos.copy()`` / ``data.qvel.copy()``; every
                  subsequent episode does ``np.copyto(data.qpos,
                  snapshot.qpos)`` + ``mj_forward`` so derived state
                  reflects the canonical pose. This is the actual fix for
                  #166's ``success_rate=0.00`` symptom on the codepath
                  ``examples/libero_mujoco.py`` exercises.

                The two branches produce equivalent observable state, so
                tests that pin one work for the other; see
                ``TestApplyCanonicalState``. Set to ``False`` to disable
                both branches (useful for diagnostic comparisons against
                the pre-fix behaviour).
            scene_keyframe_index: Which ``<keyframe>`` to apply when
                the keyframe branch is taken (``model.nkey > 0``). Defaults
                to ``0`` (first keyframe), which is the LIBERO convention.
                Pass a different index to select a non-default home pose.
                Ignored when the snapshot fallback fires.
            bddl_source: Original BDDL text - stored on the adapter so
                the scene generator can pass it back to ``libero`` (which
                only accepts a *file* path). Set automatically by
                :meth:`from_text`. Tests may set it explicitly when
                building from a pre-parsed :class:`BDDLProblem` and they
                want auto-generation to work.
            bddl_path: Original BDDL file path - same purpose as
                ``bddl_source`` but lets the scene generator skip the
                temp-file step. Set automatically by :meth:`from_file`.

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
        self._auto_generate_scene = bool(auto_generate_scene)
        self._scene_cache_dir = scene_cache_dir
        # Default camera-name alias map matches RoboSuite/LIBERO's two
        # canonical camera names to the bare keys (``image`` /
        # ``wrist_image``) that ``libero_panda``'s Gr00tDataConfig
        # expects. Passing an empty dict disables renaming.
        self._scene_camera_aliases: dict[str, str] = (
            dict(scene_camera_aliases)
            if scene_camera_aliases is not None
            else {
                "agentview": "image",
                "robot0_eye_in_hand_image": "wrist_image",
            }
        )
        self._apply_canonical_state_enabled = bool(apply_scene_keyframe)
        self._scene_keyframe_index = int(scene_keyframe_index)
        # Snapshot-and-restore fallback for procedurally-generated MJCFs that
        # don't ship a <keyframe> (the case the post-#168 verification
        # exposed). Captured on the first episode after super() +
        # _install_libero_cameras have run; replayed on every subsequent
        # episode so qpos/qvel land on the same canonical state every time.
        self._canonical_qpos: np.ndarray | None = None
        self._canonical_qvel: np.ndarray | None = None
        self._bddl_source = bddl_source
        self._bddl_path = bddl_path
        self._success_fn: Callable[[SimEngine], bool] = compile_goal(problem.goal)

    # Construction helpers

    @classmethod
    def from_file(
        cls,
        bddl_path: str | Path,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
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
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            bddl_path=str(bddl_path),
        )

    @classmethod
    def from_text(
        cls,
        bddl_text: str,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str = "hand",
        gripper_joint_name: str = "finger_joint1",
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
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
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            bddl_source=bddl_text,
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
        """Auto-generate scene (if needed), load it, capture-or-restore
        canonical state, validate Panda, install cameras, then apply jitter.

        Order matters:

        1. **Scene resolution.** When ``scene_path`` is ``None`` and
           ``auto_generate_scene`` is true, build the scene MJCF from the
           BDDL via the upstream ``libero`` package's procedural generator
           and cache it on disk. Subsequent episodes / processes reuse the
           cached XML without re-running ``libero``.
        2. ``load_scene`` (if a path is now set) - so the base
           compatibility check sees the scene's Panda rather than reporting
           "sim is empty → load default_robot".
        3. **Canonical-state apply.** ``mj_makeData`` (in
           :meth:`Simulation.load_scene`) and ``mj_resetData`` (in
           :meth:`Simulation.reset`) both initialise qpos from the
           joint-default ``qpos0`` and **silently ignore MJCF
           ``<keyframe>`` blocks**. RoboSuite-emitted LIBERO scenes encode
           the canonical home pose in a ``<keyframe>`` (rare); the
           procedurally-generated MJCFs from #165 don't ship one
           (``model.nkey == 0``), so :meth:`_apply_canonical_state` falls
           back to snapshot-and-restore - capture qpos/qvel on the first
           episode, replay on subsequent ones. **Applied IMMEDIATELY after
           ``load_scene``** (before super / install_cameras) so the
           snapshot captures the post-load canonical state without any
           recompile-induced qpos drift; super() and install_cameras
           below then operate on top of canonical state.
        4. ``super().on_episode_start`` - base compat check + auto-load
           ``default_robot`` if the sim is empty. (Note: this *can*
           recompile the spec via ``add_robot``; MuJoCo's ``spec.recompile``
           preserves qpos for existing joints, so the canonical state we
           just restored survives.)
        5. ``_install_libero_cameras`` - inject the cameras the
           ``libero_panda`` ``Gr00tDataConfig`` expects (``image`` /
           ``wrist_image``). Detects scene-supplied cameras via the
           compiled model so the static-pose fallbacks only fire when the
           scene genuinely didn't provide them - this matters for not
           recompiling the spec on top of our just-restored canonical
           state (#166 review finding).
        6. ``_apply_init_jitter`` - per-episode RNG-seeded ±jitter to
           init-subject bodies, layered on top of canonical state.
        """
        if self.scene_path is None and self._auto_generate_scene:
            try:
                generated = self._generate_scene_from_bddl()
            except Exception as e:  # noqa: BLE001 - never abort eval on a setup-time error
                logger.warning(
                    "LiberoAdapter: scene auto-generation failed (%s); falling back to bare Panda. "
                    "Install the [benchmark-libero] extra (pip install 'strands-robots[benchmark-libero]') "
                    "or pass scene_path= explicitly to silence this warning.",
                    e,
                )
                generated = None
            if generated is not None:
                self.scene_path = generated

        scene_was_loaded = False
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
                scene_was_loaded = True
        # Pre-register the default robot in world.robots BEFORE super()
        # runs. Otherwise super().on_episode_start (the base
        # BenchmarkProtocol) would see an empty list_robots() (because
        # Simulation.load_scene resets world.robots = {}) and call its
        # own sim.add_robot — which recompiles the spec, jumping
        # model.nq from N1 → N2 (#166 second-round verification: probe
        # showed 44 → 53 with a LIBERO scene). That recompile would
        # invalidate any qpos snapshot we then capture, since ep1's
        # snapshot would be at N2 but ep2's load_scene resets back to
        # N1 and add_robot fires again. By pre-registering here, super()
        # skips its add_robot and goes straight to the compatibility
        # check; subsequent episodes also pre-register so the snapshot
        # shape is stable across episodes.
        if scene_was_loaded:
            self._register_default_robot(sim)
        # Apply canonical state RIGHT AFTER load_scene + pre-register so
        # the snapshot captures the post-load + post-add_robot state -
        # before super() and install_cameras get a chance to do anything
        # else (#166 review: snapshot taken at the wrong lifecycle point
        # was the prior round's failure mode).
        if scene_was_loaded and self._apply_canonical_state_enabled:
            self._apply_canonical_state(sim)
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

    def _generate_scene_from_bddl(self) -> str | None:
        """Build the LIBERO scene MJCF from the BDDL via the upstream ``libero`` package.

        Returns the absolute path to a cached MJCF file, or ``None`` when
        the BDDL source isn't recoverable (the adapter was constructed
        from a pre-parsed :class:`BDDLProblem` without ``bddl_source`` /
        ``bddl_path``). Raises on any other failure path so callers in
        :meth:`on_episode_start` can decide whether to abort or fall back
        to bare-Panda.

        Procedure:

        1. Locate (or write) a ``.bddl`` file on disk - ``libero`` only
           accepts a path. Existing ``bddl_path`` is reused as-is.
        2. Compute SHA256 of the BDDL bytes; cache key is
           ``<scene_cache_dir>/<sha>.xml``. Cache hit → return path
           without touching ``libero`` at all (no GPU / robosuite import).
        3. Cache miss → ``require_optional("libero")`` lazy-imports the
           upstream package, then ``libero.libero.envs.env_wrapper.ControlEnv(
           bddl_file_name=..., has_offscreen_renderer=False, has_renderer=False,
           use_camera_obs=False)`` constructs a robosuite env without
           opening a GL context. Robosuite's ``env.sim.model.get_xml()``
           returns the compiled MJCF as a string.
        4. Apply :attr:`_scene_camera_aliases` via a targeted XML
           rename so the policy-side cameras (``image`` / ``wrist_image``)
           resolve to the LIBERO-canonical viewpoints.
        5. Write the renamed XML to the cache and return the path.

        The LIBERO env is closed after extraction; no robosuite state
        survives this method.
        """
        bddl_path = self._resolve_bddl_path_for_libero()
        if bddl_path is None:
            logger.debug(
                "LiberoAdapter: no BDDL source available for scene generation - "
                "constructed from a pre-parsed BDDLProblem without bddl_source / bddl_path"
            )
            return None

        bddl_bytes = bddl_path.read_bytes()
        sha = hashlib.sha256(bddl_bytes).hexdigest()
        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        cache_path = cache_dir / f"{sha}.xml"
        if cache_path.exists():
            logger.debug("LiberoAdapter: scene cache hit %s", cache_path)
            return str(cache_path)

        # Cache miss - lazy-import libero, build the scene.
        env_wrapper = require_optional(
            "libero.libero.envs.env_wrapper",
            pip_install="libero",
            extra="benchmark-libero",
            purpose="LIBERO scene generation from BDDL",
        )
        ControlEnv = env_wrapper.ControlEnv  # type: ignore[attr-defined]

        # ``has_offscreen_renderer=False`` + ``has_renderer=False`` skip
        # the GL-context bring-up that ``OffScreenRenderEnv`` would
        # otherwise require - we only need the *compiled* model, not
        # rendered frames. ``use_camera_obs=False`` further disables
        # camera observation collection during reset, which would also
        # touch the renderer.
        env = ControlEnv(
            bddl_file_name=str(bddl_path),
            has_offscreen_renderer=False,
            has_renderer=False,
            use_camera_obs=False,
        )
        try:
            xml = _extract_compiled_mjcf(env)
        finally:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001 - close errors are non-fatal
                logger.debug("LiberoAdapter: env.close() raised after extraction: %s", e)

        if self._scene_camera_aliases:
            xml = _rename_mjcf_cameras(xml, self._scene_camera_aliases)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(xml)
        logger.info(
            "LiberoAdapter: generated scene MJCF for %s -> %s",
            self.problem.name,
            cache_path,
        )
        return str(cache_path)

    def _resolve_bddl_path_for_libero(self) -> Path | None:
        """Return a ``Path`` to a ``.bddl`` file libero can open, or ``None``.

        - If the adapter was constructed via :meth:`from_file`,
          ``self._bddl_path`` already points at a real file - reuse it.
        - If constructed via :meth:`from_text`, write the source text to
          a stable temp file (keyed by SHA256 of the text) so libero has
          a real path. The temp file lives under
          ``<scene_cache_dir>/.bddl/`` so it's cleaned up alongside the
          scene cache.
        - If neither is set, return ``None``.
        """
        if self._bddl_path is not None:
            p = Path(self._bddl_path).expanduser()
            if p.is_file():
                return p
            logger.debug(
                "LiberoAdapter: bddl_path=%s not on disk; falling back to bddl_source",
                p,
            )
        if self._bddl_source is None:
            return None

        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        bddl_dir = cache_dir / ".bddl"
        bddl_dir.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(self._bddl_source.encode("utf-8")).hexdigest()
        tmp = bddl_dir / f"{sha}.bddl"
        if not tmp.exists():
            tmp.write_text(self._bddl_source)
        return tmp

    def _register_default_robot(self, sim: SimEngine) -> None:
        """Pre-register the default robot in ``sim._world.robots`` if absent.

        Goal: make ``sim.list_robots()`` return non-empty BEFORE
        ``super().on_episode_start`` runs, so the base
        :class:`BenchmarkProtocol` skips its unconditional
        ``sim.add_robot(name="robot", ...)`` call. Otherwise that
        unconditional call recompiles the spec on every episode (the
        scene-supplied Panda is a *separate* kinematic chain from the
        injected ``robot/`` namespaced one — both end up in the merged
        model at ``nq = N1 + 9``).

        That recompile is what invalidates the qpos snapshot across
        episodes (#166 round-3 verification: ``model.nq`` jumps
        ``44 → 53`` on the LIBERO SCENE5 task). By pre-registering with
        the SAME name super() would have used (``"robot"``), super()
        finds ``list_robots() == ["robot"]`` and goes straight to the
        compatibility check.

        Best-effort:

        * Sims without ``add_robot`` (non-MuJoCo backends) → debug-log
          + skip; super() will then fall through to its own ``add_robot``
          path which the backend handles however it likes.
        * ``add_robot`` returns an error dict → log WARNING and skip;
          super() will retry with the same kwargs and surface the same
          error consistently.
        * Robot already registered (from a prior episode where the
          adapter pre-registered AND the world has been preserved across
          ``load_scene`` calls — currently never the case but defensive)
          → no-op.
        """
        list_robots = getattr(sim, "list_robots", None)
        add_robot = getattr(sim, "add_robot", None)
        if list_robots is None or add_robot is None:
            logger.debug("LiberoAdapter: sim missing list_robots / add_robot; skipping pre-register")
            return

        try:
            existing = list(list_robots())
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("LiberoAdapter: list_robots() raised: %s", e)
            return
        if "robot" in existing:
            return  # super() will see it and skip its own add

        try:
            result = add_robot(name="robot", data_config=self.default_robot)
        except Exception as e:  # noqa: BLE001 - never abort eval on a setup-time error
            logger.warning(
                "LiberoAdapter: pre-register sim.add_robot('robot', %r) raised: %s; "
                "super() will retry and surface the same error",
                self.default_robot,
                e,
            )
            return
        if isinstance(result, dict) and result.get("status") == "error":
            msg = (result.get("content") or [{}])[0].get("text", "")
            logger.warning(
                "LiberoAdapter: pre-register sim.add_robot('robot', %r) failed: %s; "
                "super() will retry and surface the same error",
                self.default_robot,
                msg,
            )

    def _install_libero_cameras(self, sim: SimEngine) -> None:
        """Inject the cameras the ``libero_panda`` data_config expects.

        Best-effort: the LIBERO ``Gr00tDataConfig`` declares
        ``video_keys = ["video.image", "video.wrist_image"]`` and the policy's
        ``_build_service_observation`` reads those from the robot observation
        as ``obs["image"]`` / ``obs["wrist_image"]``. Without these cameras
        in the sim, every direct-client call to a LIBERO server fails with
        ``Video key 'video.image' must be in observation`` (#148, Failure 1).

        Cameras already present in the sim are skipped silently. "Already
        present" means *either*:

        * the runtime camera registry on ``sim._world.cameras`` (added via
          a previous ``sim.add_camera`` call, including by this adapter on
          a prior episode), OR
        * the *compiled MuJoCo model* (declared via ``<camera>`` elements
          in a scene MJCF that ``sim.load_scene`` just loaded).

        The model-side check is critical for #166 because
        :meth:`Simulation.load_scene` creates a fresh ``SimWorld`` whose
        ``cameras`` registry starts empty even when the loaded MJCF
        declares cameras. Without the model-side check the install would
        re-add the same cameras on top of the scene's ones, triggering a
        spec recompile that resets qpos away from the canonical state we
        just restored in :meth:`_apply_canonical_state`.

        Other ``add_camera`` failures are logged at WARNING but never
        fatal - one missing camera shouldn't kill the whole eval.
        """
        add_camera = getattr(sim, "add_camera", None)
        if add_camera is None:
            logger.debug("LiberoAdapter: sim has no add_camera(); skipping camera install")
            return

        existing = self._existing_camera_names(sim)

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

    @staticmethod
    def _existing_camera_names(sim: SimEngine) -> set[str]:
        """Union of registry-side and model-side camera names known to ``sim``.

        Backends without a MuJoCo-compiled model (or with mujoco not
        importable) fall back to the registry-only check - that's the
        pre-#166-review behaviour, retained as a defensive fallback for
        non-MuJoCo engines that still want LIBERO eval.

        Critical for #166: :meth:`Simulation.load_scene` creates a fresh
        ``SimWorld`` whose ``cameras`` dict starts empty even when the
        loaded MJCF declares ``<camera>`` elements. Without enumerating
        the compiled model's cameras here, ``_install_libero_cameras``
        would unconditionally try to inject ``image`` / ``wrist_image``
        on top of scene-declared ones, triggering a spec recompile that
        in turn resets qpos and undoes :meth:`_apply_canonical_state`.
        """
        names: set[str] = set()
        world = getattr(sim, "_world", None)

        # Registry-side: cameras added via sim.add_camera() previously.
        cameras_attr = getattr(world, "cameras", None) if world is not None else None
        if isinstance(cameras_attr, dict):
            names.update(cameras_attr.keys())

        # Model-side: cameras declared in a loaded scene MJCF.
        model = getattr(world, "_model", None) if world is not None else None
        if model is None:
            return names
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping model-side camera check")
            return names
        try:
            ncam = int(getattr(model, "ncam", 0))
            for i in range(ncam):
                name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    names.add(name)
        except Exception as e:  # noqa: BLE001 - never fatal during camera-existence check
            logger.debug("LiberoAdapter: model-side camera enumeration failed: %s", e)
        return names

    def _apply_canonical_state(self, sim: SimEngine) -> None:
        """Restore qpos / qvel to the scene's canonical home state.

        Two branches, in order of preference:

        1. **Keyframe** (``model.nkey > 0``): call
           ``mujoco.mj_resetDataKeyframe(model, data, scene_keyframe_index)``.
           The MJCF carries the canonical pose explicitly via a
           ``<keyframe>`` element - LIBERO-authored hand-written scenes
           (the ones in upstream ``libero/libero/assets/scenes/``) ship one.
        2. **Snapshot-and-restore** (``model.nkey == 0``): cache
           ``data.qpos`` / ``data.qvel`` on the FIRST episode after a
           scene compile (after ``super().on_episode_start`` and
           ``_install_libero_cameras`` have run); restore the cached
           snapshot on every subsequent episode. The procedurally-
           generated MJCFs from :meth:`_generate_scene_from_bddl` (PR #165)
           don't carry a keyframe, so this branch is the one that
           actually fires on the codepath ``examples/libero_mujoco.py``
           exercises today (#166's reported symptom).

        Both branches end with ``mj_forward`` so derived state
        (``xpos`` / ``xquat`` / sensor data) reflects the canonical
        ``qpos`` before the next ``get_observation`` / ``render`` call.

        Best-effort:

        * Sims without an exposed compiled MuJoCo model → debug-log + skip.
        * ``scene_keyframe_index`` out of range when ``nkey > 0`` → log
          at WARNING and skip (out-of-range is a config error).
        * ``mujoco`` not importable → debug-log + skip.
        * Snapshot shape mismatches the current ``qpos`` (e.g. the
          model recompiled with a different ``nq`` between episodes,
          which is unusual) → re-capture instead of restoring.

        Holds ``sim._lock`` if the sim exposes one to match the locking
        contract of :meth:`Simulation.reset` and :meth:`Simulation.send_action`
        - prevents racing a worker holding a stale qpos pointer.
        """
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None
        if model is None or data is None:
            logger.debug("LiberoAdapter: sim has no compiled MuJoCo model/data; skipping canonical-state apply")
            return

        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping canonical-state apply")
            return

        nkey = int(getattr(model, "nkey", 0))
        lock = getattr(sim, "_lock", None)

        if nkey > 0:
            self._apply_keyframe_branch(sim, model, data, _mj, lock, nkey)
        else:
            self._apply_snapshot_branch(sim, model, data, _mj, lock)

    def _apply_keyframe_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_snapshot_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
        nkey: int,
    ) -> None:
        """Keyframe branch of :meth:`_apply_canonical_state`."""
        if self._scene_keyframe_index < 0 or self._scene_keyframe_index >= nkey:
            logger.warning(
                "LiberoAdapter: scene_keyframe_index=%d out of range [0, %d); skipping",
                self._scene_keyframe_index,
                nkey,
            )
            return
        try:
            if lock is not None:
                with lock:
                    mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                    mj.mj_forward(model, data)
            else:
                mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning(
                "LiberoAdapter: mj_resetDataKeyframe(%d) failed: %s",
                self._scene_keyframe_index,
                e,
            )
            return
        logger.debug(
            "LiberoAdapter: applied <keyframe> %d to canonical qpos",
            self._scene_keyframe_index,
        )

    def _apply_snapshot_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_keyframe_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
    ) -> None:
        """Snapshot-and-restore branch of :meth:`_apply_canonical_state`.

        First episode: capture ``data.qpos`` / ``data.qvel``. Subsequent
        episodes: restore the cached snapshot via ``np.copyto`` and
        ``mj_forward``. Procedurally-generated MJCFs (#165) hit this
        branch because they don't ship a ``<keyframe>``.
        """
        try:
            qpos = data.qpos
            qvel = data.qvel
        except AttributeError as e:
            logger.debug("LiberoAdapter: data has no qpos/qvel attrs: %s", e)
            return

        # First episode (or model recompile changed nq) -> capture, don't
        # restore. The snapshot is taken after super() + _install_libero_cameras
        # so it reflects the post-setup canonical state.
        needs_capture = (
            self._canonical_qpos is None
            or self._canonical_qpos.shape != qpos.shape
            or self._canonical_qvel is None
            or self._canonical_qvel.shape != qvel.shape
        )
        if needs_capture:
            try:
                self._canonical_qpos = np.array(qpos, copy=True)
                self._canonical_qvel = np.array(qvel, copy=True)
            except Exception as e:  # noqa: BLE001 - capture is best-effort
                logger.debug("LiberoAdapter: snapshot capture failed: %s", e)
                self._canonical_qpos = None
                self._canonical_qvel = None
                return
            logger.debug(
                "LiberoAdapter: captured canonical qpos snapshot (nq=%d, nv=%d)",
                self._canonical_qpos.shape[0],
                self._canonical_qvel.shape[0],
            )
            return

        # Subsequent episode - restore the snapshot. The needs_capture
        # check above guarantees the snapshot fields are non-None here;
        # narrow for mypy.
        assert self._canonical_qpos is not None
        assert self._canonical_qvel is not None
        canonical_qpos = self._canonical_qpos
        canonical_qvel = self._canonical_qvel
        try:
            if lock is not None:
                with lock:
                    np.copyto(qpos, canonical_qpos)
                    np.copyto(qvel, canonical_qvel)
                    mj.mj_forward(model, data)
            else:
                np.copyto(qpos, canonical_qpos)
                np.copyto(qvel, canonical_qvel)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning("LiberoAdapter: snapshot restore failed: %s", e)
            return
        logger.debug("LiberoAdapter: restored canonical qpos snapshot")

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


# Scene-generation helpers (#164)


def _default_scene_cache_dir() -> Path:
    """Filesystem location for cached LIBERO scene MJCFs.

    Uses :func:`strands_robots.utils.get_base_dir` so the cache lives
    under ``$STRANDS_BASE_DIR`` (typically ``~/.strands_robots/``)
    alongside other strands-robots state. Created on demand by
    :meth:`LiberoAdapter._generate_scene_from_bddl` - this helper just
    returns the path.
    """
    return get_base_dir() / "scene_cache" / "libero"


def _extract_compiled_mjcf(env: Any) -> str:
    """Pull the compiled MJCF XML out of a ``libero`` ControlEnv.

    ``ControlEnv.env`` is the underlying robosuite manipulation env;
    its ``.sim.model.get_xml()`` returns the merged / compiled MJCF as
    a string (with all ``<include>``s resolved and assets inlined).
    Robosuite renamed this accessor over the years, so we try a small
    set of fallbacks before giving up - and we never look at any
    non-public attributes.
    """
    accessors = (
        # Newer robosuite (>=1.4) - canonical path through the env's MjSim.
        lambda: env.env.sim.model.get_xml(),
        # Older robosuite (<1.4) - sometimes exposes the model directly.
        lambda: env.env.model.get_xml(),
        # Fallback: ManipulationEnv subclasses sometimes expose the
        # compiled XML via an explicit ``model.get_model_xml`` helper.
        lambda: env.env.model.get_model_xml(),  # type: ignore[attr-defined]
    )
    last_err: Exception | None = None
    for accessor in accessors:
        try:
            xml = accessor()
        except Exception as e:  # noqa: BLE001 - try the next accessor
            last_err = e
            continue
        if isinstance(xml, str) and xml.strip():
            return xml
    raise RuntimeError(f"could not extract compiled MJCF from libero env (last error: {last_err!r})")


# Match a complete ``<camera ... name="OLD" ...>`` declaration so the
# rename only touches camera definitions, not e.g. material names that
# happen to share a string. Anchored on the ``camera`` element name and
# guarded by ``\s+`` to avoid partial-word matches.
_CAMERA_NAME_RE = re.compile(r'(<camera\b[^>]*\bname=")([^"]+)(")')


def _rename_mjcf_cameras(xml: str, aliases: dict[str, str]) -> str:
    """Rename ``<camera name="OLD"...>`` → ``<camera name="NEW"...>`` per ``aliases``.

    Targeted regex only - we don't parse the whole MJCF. The rename is
    safe because MuJoCo doesn't allow duplicate ``<camera>`` names within
    a model, and camera references from external code (e.g.
    ``sim.render(camera_name=...)``) come from outside the XML so they
    aren't affected.

    Names not in ``aliases`` pass through unchanged.
    """
    if not aliases:
        return xml

    def _sub(match: re.Match[str]) -> str:
        head, name, tail = match.group(1), match.group(2), match.group(3)
        return head + aliases.get(name, name) + tail

    return _CAMERA_NAME_RE.sub(_sub, xml)


__all__ = [
    "BDDLParseError",
    "LiberoAdapter",
]
