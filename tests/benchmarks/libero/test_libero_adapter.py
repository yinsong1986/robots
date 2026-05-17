"""Tests for :class:`LiberoAdapter`.

Covers:

* Construction via ``from_file`` / ``from_text`` / raw ``__init__``.
* ``supported_robots`` / ``default_robot`` = Panda-only.
* ``instruction`` surfaces the BDDL ``:language`` string.
* ``is_success`` positive + negative cases against fake sims (no MuJoCo
  needed - the predicates poll ``get_body_state`` / ``get_contacts``).
* ``on_episode_start`` loads the scene (or errors cleanly) and applies
  per-episode jitter when the sim exposes ``move_object``.
* Integration with ``PolicyRunner.evaluate`` and
  ``SimEngine.evaluate_benchmark`` via a minimal ``FakeSim`` stub.
* Error surface: unknown task via ``evaluate_benchmark`` returns a
  structured error dict, never raises.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.benchmarks.libero import (
    BDDLParseError,
    LiberoAdapter,
)
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    register_benchmark,
)
from strands_robots.simulation.policy_runner import PolicyRunner


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


# Representative BDDL fragments

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

COMPOUND_BDDL = """
(define (problem libero_grasp_and_upright)
  (:language "grasp the bottle and keep it upright")
  (:goal (and (grasped bottle_1) (upright bottle_1))))
"""

NEGATED_BDDL = """
(define (problem libero_release)
  (:goal (not (grasped cube_1))))
"""


# Fake sim helpers


class _FakeRobot:
    def __init__(self, data_config: str):
        self.data_config = data_config


class _FakeWorld:
    def __init__(self, robots: dict[str, _FakeRobot]):
        self.robots = dict(robots)
        self.cameras: dict[str, dict[str, Any]] = {}


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` with get_body_state / get_contacts / move_object / add_camera."""

    def __init__(
        self,
        bodies: dict[str, dict[str, Any]] | None = None,
        contacts: list[dict[str, str]] | None = None,
        data_config: str = "panda",
        preexisting_cameras: list[str] | None = None,
        add_camera_fail: bool = False,
    ):
        self._bodies = dict(bodies or {})
        self._contacts = list(contacts or [])
        self._reset_count = 0
        self._move_calls: list[tuple[str, list[float]]] = []
        self._world = _FakeWorld({"fake_panda": _FakeRobot(data_config)})
        self._scenes_loaded: list[str] = []
        self._add_camera_calls: list[tuple[str, dict[str, Any]]] = []
        self._add_camera_fail = add_camera_fail
        for cam in preexisting_cameras or []:
            self._world.cameras[cam] = {"position": [0, 0, 0], "target": [0, 0, 0]}

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        dc = kw.get("data_config") or "panda"
        self._world.robots[name] = _FakeRobot(dc)
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name):
        return ["j0", "j1"]

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self.robot_joint_names(robot_name or "fake_panda")}

    def send_action(self, action, robot_name=None, n_substeps=1):
        pass

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}

    # Optional helpers used by predicates + adapter

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {
                    "json": {
                        "position": self._bodies[body_name].get("position", [0, 0, 0]),
                        "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
                        "mass": 1.0,
                    }
                },
            ],
        }

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}},
            ],
        }

    def move_object(self, *, name: str, position: list[float]) -> dict[str, Any]:
        self._move_calls.append((name, list(position)))
        self._bodies.setdefault(name, {})["position"] = list(position)
        return {"status": "success"}

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        self._scenes_loaded.append(scene_path)
        return {"status": "success"}

    def add_camera(
        self,
        name: str,
        position: list[float] | None = None,
        target: list[float] | None = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
    ) -> dict[str, Any]:
        kwargs = {
            "position": position,
            "target": target,
            "fov": fov,
            "width": width,
            "height": height,
        }
        self._add_camera_calls.append((name, dict(kwargs)))
        if self._add_camera_fail:
            return {"status": "error", "content": [{"text": "fake injection failed"}]}
        if name in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"add_camera: camera '{name}' already exists. Remove it first."}],
            }
        self._world.cameras[name] = kwargs
        return {"status": "success", "content": [{"text": f"📷 Camera '{name}' added"}]}


# Construction


class TestConstruction:
    def test_from_text_happy_path(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter.supported_robots == ["panda"]
        assert adapter.default_robot == "panda"
        assert adapter.max_steps == 300
        assert adapter.instruction == "pick up the red cube and place it on the plate"

    def test_from_text_respects_max_steps_override(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=75)
        assert adapter.max_steps == 75

    def test_from_text_rejects_negative_jitter(self):
        with pytest.raises(ValueError):
            LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=-0.1)

    def test_rejects_bddl_without_goal(self):
        text = '(define (problem no_goal) (:language "no goal block"))'
        with pytest.raises(ValueError, match="no \\(:goal"):
            LiberoAdapter.from_text(text)

    def test_from_file(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(p)
        assert adapter.problem.name == "libero_spatial_pick_cube"

    def test_from_text_propagates_parse_errors(self):
        with pytest.raises(BDDLParseError):
            LiberoAdapter.from_text("(define (problem p) (:goal (telekinesis cube_1)))")


# Lifecycle hooks


class TestIsSuccess:
    def test_positive_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.25]},
                "plate_1": {"position": [0, 0, 0.1]},
            }
        )
        assert adapter.is_success(sim) is True

    def test_negative_case_on(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.05]},  # not above plate
                "plate_1": {"position": [0, 0, 0.1]},
            }
        )
        assert adapter.is_success(sim) is False

    def test_compound_goal_needs_both(self):
        adapter = LiberoAdapter.from_text(COMPOUND_BDDL)
        sim_neither = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_upright_only = FakeSim(bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        sim_both = FakeSim(
            bodies={"bottle_1": {"quaternion": [1.0, 0, 0, 0]}},
            contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "bottle_1"}],
        )
        assert adapter.is_success(sim_neither) is False
        assert adapter.is_success(sim_upright_only) is False
        assert adapter.is_success(sim_both) is True

    def test_negated_goal(self):
        adapter = LiberoAdapter.from_text(NEGATED_BDDL)
        sim_empty = FakeSim()
        sim_gripped = FakeSim(contacts=[{"geom1": "robot0_gripper_finger_l", "geom2": "cube_1"}])
        assert adapter.is_success(sim_empty) is True
        assert adapter.is_success(sim_gripped) is False


class TestOnEpisodeStart:
    def test_loads_scene_before_compat_check(self, tmp_path):
        """``scene_path`` load must happen before ``super().on_episode_start``
        so the base compat check sees the scene's Panda robot."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/fake/scene.xml")
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))
        assert sim._scenes_loaded == ["/fake/scene.xml"]

    def test_scene_load_error_raises(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, scene_path="/bad/path.xml")

        sim = FakeSim(data_config="panda")
        # Override load_scene to report failure.
        sim.load_scene = lambda path: {  # type: ignore[assignment]
            "status": "error",
            "content": [{"text": f"no such file: {path}"}],
        }
        with pytest.raises(RuntimeError, match="load_scene"):
            adapter.on_episode_start(sim, random.Random(0))

    def test_jitter_applied_when_init_declares_subject(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.01,
        )
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0.0, 0.0, 0.1]},
                "table_1": {"position": [0.0, 0.0, 0.0]},
            },
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        # cube_1 got jittered; table_1 is only the reference and not moved.
        assert any(call[0] == "cube_1" for call in sim._move_calls)
        assert not any(call[0] == "table_1" for call in sim._move_calls)

    def test_jitter_disabled_when_zero(self):
        adapter = LiberoAdapter.from_text(
            """
            (define (problem p)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
            """,
            init_jitter=0.0,
        )
        sim = FakeSim(
            bodies={"cube_1": {"position": [0, 0, 0.1]}, "table_1": {"position": [0, 0, 0]}},
            data_config="panda",
        )
        adapter.on_episode_start(sim, random.Random(42))
        assert sim._move_calls == []

    def test_non_panda_robot_rejected_by_base_compat_check(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(data_config="so100")  # not panda
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        with pytest.raises(BenchmarkCompatibilityError) as exc:
            adapter.on_episode_start(sim, random.Random(0))
        assert exc.value.supported == ["panda"]
        assert exc.value.data_config == "so100"


# LIBERO camera installation (#148 / Failure 1)


class TestLiberoCameraInstall:
    def test_default_cameras_installed(self):
        """``image`` and ``wrist_image`` cameras must be installed so the
        ``libero_panda`` Gr00tDataConfig finds them in the observation."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"image", "wrist_image"}
        assert "image" in sim._world.cameras
        assert "wrist_image" in sim._world.cameras

    def test_camera_install_happens_after_robot_load(self):
        """Camera install must run AFTER super().on_episode_start so it
        installs into a populated scene rather than an empty world."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)

        events: list[str] = []

        class TrackingFakeSim(FakeSim):
            def add_robot(self, name, **kw):
                events.append("add_robot")
                return super().add_robot(name, **kw)

            def add_camera(self, name, **kw):
                events.append(f"add_camera:{name}")
                return super().add_camera(name, **kw)

        sim = TrackingFakeSim(data_config="panda")
        # Empty world so super() triggers add_robot for default_robot.
        sim._world.robots.clear()
        adapter.on_episode_start(sim, random.Random(0))

        # add_robot must come before any add_camera call.
        first_camera_idx = next(i for i, e in enumerate(events) if e.startswith("add_camera:"))
        first_robot_idx = events.index("add_robot")
        assert first_robot_idx < first_camera_idx

    def test_install_cameras_disabled(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0, install_cameras=False)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))
        assert sim._add_camera_calls == []

    def test_custom_cameras_override_defaults(self):
        custom = {
            "wide_view": {
                "position": [2.0, 2.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "fov": 90.0,
                "width": 128,
                "height": 128,
            }
        }
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0, cameras=custom)
        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        # Defaults are completely replaced - only the custom camera is installed.
        assert installed_names == {"wide_view"}

    def test_existing_cameras_skipped(self):
        """A scene MJCF that already declares ``image`` should beat us to it -
        the adapter must not error out when the camera already exists."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda", preexisting_cameras=["image"])
        adapter.on_episode_start(sim, random.Random(0))

        # ``image`` was preexisting - we must not call add_camera for it.
        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"wrist_image"}

    def test_model_side_cameras_skip_install_no_recompile(self):
        """Critical for #166: ``Simulation.load_scene`` creates a fresh
        ``SimWorld`` whose registry is empty even when the loaded MJCF
        declares cameras. ``_install_libero_cameras`` MUST detect the
        scene-supplied cameras via the compiled model (not just the
        registry) so it doesn't trigger a spec recompile that resets
        qpos and undoes ``_apply_canonical_state``."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return ["image", "wrist_image"][idx]

        class _FakeMjModel:
            ncam = 2

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        # Registry is empty (mimics post-load_scene state) but the
        # compiled model declares both LIBERO cameras.
        sim._world.cameras.clear()
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            adapter.on_episode_start(sim, random.Random(0))

        # No add_camera calls - both names detected as already-present
        # in the compiled model.
        assert sim._add_camera_calls == []

    def test_partial_model_side_cameras_fills_missing(self):
        """If the scene declares ONLY ``image`` (not ``wrist_image``) in
        the compiled model, we install just the missing one."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return ["image"][idx]

        class _FakeMjModel:
            ncam = 1

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda")
        sim._world.cameras.clear()
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            adapter.on_episode_start(sim, random.Random(0))

        installed_names = {name for name, _ in sim._add_camera_calls}
        assert installed_names == {"wrist_image"}

    def test_existing_camera_names_unions_registry_and_model(self):
        """Direct exercise of ``_existing_camera_names`` - union of
        registry and model camera names."""

        class _FakeMjEnum:
            mjOBJ_CAMERA = 7

        class _FakeMjModule:
            mjtObj = _FakeMjEnum()

            @staticmethod
            def mj_id2name(model, obj_type, idx):  # noqa: ARG004
                return "agentview"

        class _FakeMjModel:
            ncam = 1

        sim = FakeSim(data_config="panda", preexisting_cameras=["topdown"])
        sim._world._model = _FakeMjModel()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"mujoco": _FakeMjModule()}):
            names = LiberoAdapter._existing_camera_names(sim)

        assert names == {"topdown", "agentview"}

    def test_camera_install_failures_are_logged_not_fatal(self, caplog):
        """A backend that refuses every add_camera shouldn't kill the eval.

        The adapter's contract: log at WARNING and continue so other
        cameras + the rollout itself still run.
        """
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        sim = FakeSim(data_config="panda", add_camera_fail=True)
        with caplog.at_level("WARNING"):
            # Must not raise.
            adapter.on_episode_start(sim, random.Random(0))
        # Both cameras attempted; both reported as warning.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 2

    def test_default_camera_keys_match_libero_panda_data_config(self):
        """Regression: the adapter's camera names MUST match the bare keys of
        the ``libero_panda`` Gr00tDataConfig's video_keys, otherwise
        ``_build_service_observation`` won't find them in the observation."""
        # Bare keys after removeprefix("video.").
        expected = {"image", "wrist_image"}
        assert set(LiberoAdapter.LIBERO_CAMERAS.keys()) == expected

    def test_sim_without_add_camera_silently_skipped(self):
        """Backends that don't expose add_camera (future engines) shouldn't
        crash the adapter - it falls back to "no camera install"."""

        class BareSim(FakeSim):
            # Override add_camera to ``None`` so getattr returns falsy at runtime.
            add_camera = None  # type: ignore[assignment]

        sim = BareSim(data_config="panda")
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.0)
        # Must not raise even though sim.add_camera is None.
        adapter.on_episode_start(sim, random.Random(0))


# Step semantics


class TestOnStep:
    def test_sparse_reward_zero_and_not_done(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        info = adapter.on_step(FakeSim(), {}, {})
        assert info.reward == 0.0
        assert info.done is False


# augment_observation - LIBERO state bridge (#156)


class TestAugmentObservation:
    """``LiberoAdapter.augment_observation`` injects ``x`` / ``y`` / ``z`` /
    ``roll`` / ``pitch`` / ``yaw`` / ``gripper`` so the ``libero_panda``
    Gr00tDataConfig finds them in the per-step observation."""

    def test_default_panda_layout(self):
        """Identity quaternion ⇒ all Euler angles are zero."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(
            bodies={
                "hand": {
                    "position": [0.5, -0.1, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],  # identity
                },
            }
        )
        # Provide finger_joint1 in the input obs (the runner gets it from
        # sim.get_observation; here we simulate that contract directly).
        obs = {"finger_joint1": 0.04, "joint1": 0.0}
        out = adapter.augment_observation(sim, obs)
        # Cartesian
        assert out["x"] == pytest.approx(0.5)
        assert out["y"] == pytest.approx(-0.1)
        assert out["z"] == pytest.approx(0.3)
        # Euler - identity quat → all zeros.
        assert out["roll"] == pytest.approx(0.0, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(0.0, abs=1e-9)
        # Gripper from the finger_joint1 reading. Packed as a 2-element
        # list to match `robot0_gripper_qpos` (the LIBERO/RoboSuite
        # 2-finger array the GR00T-N1.7-LIBERO checkpoint was trained
        # on); both entries mirror the same finger_joint1 reading
        # because the Menagerie Panda's MJCF equality constraint
        # forces finger_joint1 == finger_joint2.
        assert out["gripper"] == pytest.approx([0.04, 0.04])
        # Original keys preserved.
        assert out["finger_joint1"] == 0.04
        assert out["joint1"] == 0.0

    def test_quaternion_to_euler_z_rotation(self):
        """90° rotation about Z ⇒ yaw = π/2, roll = pitch = 0."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        # MuJoCo quat (w,x,y,z) for 90° about Z: (cos(45°), 0, 0, sin(45°))
        quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["roll"] == pytest.approx(0.0, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(math.pi / 2, abs=1e-9)

    def test_quaternion_to_euler_x_rotation(self):
        """90° rotation about X ⇒ roll = π/2, pitch = yaw = 0."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["roll"] == pytest.approx(math.pi / 2, abs=1e-9)
        assert out["pitch"] == pytest.approx(0.0, abs=1e-9)
        assert out["yaw"] == pytest.approx(0.0, abs=1e-9)

    def test_quaternion_gimbal_lock_does_not_crash(self):
        """90° rotation about Y is gimbal-locked; the resolution may
        push roll into yaw, but pitch must equal π/2 and the call MUST
        return without raising."""
        import math

        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": quat}})
        out = adapter.augment_observation(sim, {})
        assert out["pitch"] == pytest.approx(math.pi / 2, abs=1e-6)

    def test_eef_state_keys_match_libero_panda_data_config(self):
        """Regression: keys MUST exactly equal the bare names of the
        ``libero_panda`` data_config's state_keys, otherwise the policy
        won't find them."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {"finger_joint1": 0.0})
        # libero_panda state_keys.removeprefix("state.") = bare names below.
        for required in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            assert required in out, f"missing {required!r} in augmented obs"

    def test_missing_body_drops_pose_keys_silently(self):
        """If the EEF body isn't in the sim, the adapter must NOT raise -
        the missing keys are simply omitted so the policy surfaces a
        clearer 'state.x must be in observation' error than a Python
        traceback."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={})  # no "hand" body
        out = adapter.augment_observation(sim, {"finger_joint1": 0.04})
        assert "x" not in out
        assert "y" not in out
        assert "roll" not in out
        # Gripper still injected because finger_joint1 is in obs;
        # mirrored into the 2-element shape the trained checkpoint
        # expects (see test_pose_and_gripper_inject_into_obs above).
        assert out["gripper"] == pytest.approx([0.04, 0.04])

    def test_missing_gripper_omits_gripper_key(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {})  # no finger_joint1
        assert "gripper" not in out

    def test_inject_eef_state_false_disables_hook(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, inject_eef_state=False)
        sim = FakeSim(bodies={"hand": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]}})
        obs = {"finger_joint1": 0.04, "joint1": 0.0}
        out = adapter.augment_observation(sim, obs)
        # Returned obs is unchanged - no x/y/z/roll/pitch/yaw/gripper.
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
            assert key not in out
        assert out["finger_joint1"] == 0.04

    def test_does_not_overwrite_keys_already_in_obs(self):
        """If a backend already supplies x/y/..., the adapter must NOT
        clobber them - users may have configured an observation_mapping."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [9.9, 9.9, 9.9], "quaternion": [1, 0, 0, 0]}})
        obs = {"x": 0.0, "finger_joint1": 0.04}
        out = adapter.augment_observation(sim, obs)
        # User's x wins over the FK lookup.
        assert out["x"] == 0.0
        # But fields the user didn't supply still get filled in.
        assert out["y"] == pytest.approx(9.9)

    def test_custom_eef_body_name(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, eef_body_name="custom_eef")
        sim = FakeSim(
            bodies={
                "hand": {"position": [9, 9, 9], "quaternion": [1, 0, 0, 0]},
                "custom_eef": {"position": [0.5, 0.5, 0.5], "quaternion": [1, 0, 0, 0]},
            }
        )
        out = adapter.augment_observation(sim, {})
        assert out["x"] == pytest.approx(0.5)
        # Default "hand" body NOT used.
        assert out["x"] != 9

    def test_custom_gripper_joint_name(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, gripper_joint_name="my_grip")
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        out = adapter.augment_observation(sim, {"my_grip": 0.025, "finger_joint1": 9.9})
        # Reads from my_grip, not finger_joint1.
        assert out["gripper"] == pytest.approx([0.025, 0.025])

    def test_namespaced_gripper_joint_resolves(self):
        """Multi-robot scenes namespace joints as ``<robot>/<joint>``. The
        adapter must accept a suffix match so users don't have to set
        ``gripper_joint_name='panda_arm/finger_joint1'`` manually."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        sim = FakeSim(bodies={"hand": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}})
        # Namespaced key as it would appear in a multi-robot sim.
        obs = {"panda_arm/finger_joint1": 0.04}
        out = adapter.augment_observation(sim, obs)
        assert out["gripper"] == pytest.approx([0.04, 0.04])

    def test_sim_without_get_body_state(self):
        """Backends that don't expose get_body_state (future engines)
        must skip pose injection silently rather than crash."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)

        class BareSim:
            pass

        sim = BareSim()
        # Must not raise.
        out = adapter.augment_observation(sim, {"finger_joint1": 0.04})  # type: ignore[arg-type]
        assert "x" not in out
        # Gripper still injected from the obs lookup.
        assert out["gripper"] == pytest.approx([0.04, 0.04])


class TestQuaternionToEuler:
    """Pure-math regression tests for ``_quat_wxyz_to_rpy_xyz`` so future
    refactors to the helper don't silently rotate the LIBERO state by π."""

    def test_identity(self):
        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        roll, pitch, yaw = _quat_wxyz_to_rpy_xyz([1.0, 0.0, 0.0, 0.0])
        assert abs(roll) < 1e-12
        assert abs(pitch) < 1e-12
        assert abs(yaw) < 1e-12

    @pytest.mark.parametrize(
        "quat,expected",
        [
            # 90° about X
            ([0.7071067811865476, 0.7071067811865475, 0.0, 0.0], (1.5707963267948966, 0.0, 0.0)),
            # 90° about Z
            ([0.7071067811865476, 0.0, 0.0, 0.7071067811865475], (0.0, 0.0, 1.5707963267948966)),
            # 180° about X - atan2(±0, -1) may return ±π, both are correct.
            ([0.0, 1.0, 0.0, 0.0], (3.141592653589793, 0.0, 0.0)),
            # 180° about Z
            ([0.0, 0.0, 0.0, 1.0], (0.0, 0.0, 3.141592653589793)),
        ],
    )
    def test_principal_axis_rotations(self, quat, expected):
        """Compare modulo 2π so atan2's ±π ambiguity at 180° rotations
        doesn't make the test brittle to floating-point sign-of-zero."""
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        out = _quat_wxyz_to_rpy_xyz(quat)
        for actual, want in zip(out, expected, strict=True):
            diff = ((actual - want + math.pi) % (2 * math.pi)) - math.pi
            assert abs(diff) < 1e-9, f"got {out}, want {expected} (mod 2π)"

    def test_gimbal_lock_pitch_pi_over_2(self):
        """90° about Y is the canonical gimbal-lock case; pitch must
        still come out as π/2 even though roll/yaw absorb each other."""
        import math

        from strands_robots.benchmarks.libero.adapter import _quat_wxyz_to_rpy_xyz

        quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        _roll, pitch, _yaw = _quat_wxyz_to_rpy_xyz(quat)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-6)


# Scene auto-generation from BDDL (#164)


class TestSceneGeneration:
    """``LiberoAdapter._generate_scene_from_bddl`` builds the per-task MJCF
    via the upstream ``libero`` package's procedural generator, with
    SHA256-keyed disk caching and a graceful fallback when ``libero``
    isn't installed."""

    def test_explicit_scene_path_skips_generation(self, tmp_path):
        """``scene_path=...`` set on the constructor wins over auto-gen.
        The generator must NOT be called when an explicit path is given."""

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(tmp_path / "explicit.xml"),
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
            sim = FakeSim(data_config="panda")
            # The explicit path doesn't exist; load_scene returns success
            # only because FakeSim's load_scene is canned. Real Simulation
            # would error - that's the test's purpose: confirm we never
            # touched the generator.
            adapter.on_episode_start(sim, random.Random(0))

        mock_gen.assert_not_called()
        assert sim._scenes_loaded == [str(tmp_path / "explicit.xml")]

    def test_auto_generate_scene_false_falls_back_to_bare_panda(self):
        """Pre-#164 behavior preserved: with auto_generate_scene=False
        and no scene_path, the adapter goes straight to the bare-Panda
        path without trying to generate."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            auto_generate_scene=False,
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch.object(adapter, "_generate_scene_from_bddl") as mock_gen:
            sim = FakeSim(data_config="panda")
            adapter.on_episode_start(sim, random.Random(0))

        mock_gen.assert_not_called()
        assert sim._scenes_loaded == []
        assert adapter.scene_path is None

    def test_libero_missing_falls_back_with_warning(self, caplog, tmp_path):
        """If ``libero`` isn't installed, the adapter must log a warning
        and continue with the legacy bare-Panda path - eval still runs,
        just against the wrong world."""
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Simulate a missing pip package - require_optional raises ImportError.
        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            side_effect=ImportError("'libero' is required"),
        ):
            with caplog.at_level("WARNING"):
                sim = FakeSim(data_config="panda")
                adapter.on_episode_start(sim, random.Random(0))

        # Eval continues without a scene loaded.
        assert sim._scenes_loaded == []
        assert adapter.scene_path is None
        # User gets a clear hint about [benchmark-libero].
        assert any(
            "scene auto-generation failed" in rec.message and "[benchmark-libero]" in rec.message
            for rec in caplog.records
        )

    def test_cache_hit_skips_libero_import(self, tmp_path):
        """If the SHA-keyed cache file already exists, the adapter must
        return its path without importing libero at all."""
        cache_dir = tmp_path / "scene_cache"
        cache_dir.mkdir()

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Pre-populate the cache file at the SHA the adapter computes.
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        cache_path = cache_dir / f"{sha}.xml"
        cache_path.write_text("<mujoco/>")

        # Sentinel: any libero import attempt would fail this test.
        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            side_effect=AssertionError("require_optional should not be called on cache hit"),
        ):
            generated = adapter._generate_scene_from_bddl()

        assert generated == str(cache_path)

    def test_cache_miss_invokes_libero_and_writes_xml(self, tmp_path):
        """Cache miss path: libero is imported, ControlEnv is constructed
        with rendering disabled, the compiled MJCF is extracted, cameras
        are renamed, and the result is written to the cache."""
        cache_dir = tmp_path / "scene_cache"

        # Mock the entire libero.libero.envs.env_wrapper module surface.
        fake_xml = (
            '<mujoco model="libero_pick"><worldbody>'
            '<camera name="agentview" pos="1 0 1"/>'
            '<camera name="robot0_eye_in_hand_image" pos="0 0 0.5"/>'
            "</worldbody></mujoco>"
        )

        fake_env = MagicMock()
        fake_env.env.sim.model.get_xml.return_value = fake_xml
        fake_module = MagicMock()
        fake_module.ControlEnv.return_value = fake_env

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            return_value=fake_module,
        ):
            generated = adapter._generate_scene_from_bddl()

        assert generated is not None
        # ControlEnv must have been constructed with rendering disabled
        # so we don't need a GL context.
        construct_kwargs = fake_module.ControlEnv.call_args.kwargs
        assert construct_kwargs["has_offscreen_renderer"] is False
        assert construct_kwargs["has_renderer"] is False
        assert construct_kwargs["use_camera_obs"] is False
        # The compiled XML was extracted from env.env.sim.model.get_xml().
        fake_env.env.sim.model.get_xml.assert_called_once()
        # Cameras renamed to libero_panda data_config bare keys.
        text = Path(generated).read_text()
        assert 'name="image"' in text
        assert 'name="wrist_image"' in text
        assert 'name="agentview"' not in text
        assert 'name="robot0_eye_in_hand_image"' not in text
        # env was closed after extraction.
        fake_env.close.assert_called_once()

    def test_camera_rename_disabled_with_empty_aliases(self, tmp_path):
        """Passing ``scene_camera_aliases={}`` keeps the LIBERO-canonical
        camera names verbatim - useful for users who want to manage the
        camera-name mapping themselves via the policy's
        observation_mapping."""
        cache_dir = tmp_path / "scene_cache"

        fake_xml = '<mujoco><worldbody><camera name="agentview" pos="1 0 1"/></worldbody></mujoco>'

        fake_env = MagicMock()
        fake_env.env.sim.model.get_xml.return_value = fake_xml
        fake_module = MagicMock()
        fake_module.ControlEnv.return_value = fake_env

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            scene_camera_aliases={},  # disable
            init_jitter=0.0,
            install_cameras=False,
        )

        with patch(
            "strands_robots.benchmarks.libero.adapter.require_optional",
            return_value=fake_module,
        ):
            generated = adapter._generate_scene_from_bddl()

        text = Path(generated).read_text()
        # agentview retained verbatim.
        assert 'name="agentview"' in text
        assert 'name="image"' not in text

    def test_cache_key_is_sha256_of_bddl(self, tmp_path):
        """Two adapters built from the SAME BDDL share a cached XML.
        Two adapters with DIFFERENT BDDL get distinct cache files."""
        cache_dir = tmp_path / "scene_cache"

        a1 = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )
        a2 = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )
        a3 = LiberoAdapter.from_text(
            COMPOUND_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
        )

        path1 = a1._resolve_bddl_path_for_libero()
        path2 = a2._resolve_bddl_path_for_libero()
        path3 = a3._resolve_bddl_path_for_libero()
        assert path1 is not None and path2 is not None and path3 is not None
        sha1 = hashlib.sha256(path1.read_bytes()).hexdigest()
        sha2 = hashlib.sha256(path2.read_bytes()).hexdigest()
        sha3 = hashlib.sha256(path3.read_bytes()).hexdigest()
        assert sha1 == sha2  # same BDDL → same key
        assert sha1 != sha3  # different BDDL → different key

    def test_resolve_bddl_path_uses_existing_file_when_set(self, tmp_path):
        """``from_file`` constructed adapters reuse the original BDDL path
        instead of writing a new temp file."""
        bddl_file = tmp_path / "task.bddl"
        bddl_file.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(
            bddl_file,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
        )
        resolved = adapter._resolve_bddl_path_for_libero()
        assert resolved == bddl_file

    def test_resolve_bddl_path_returns_none_for_unsourced_problem(self, tmp_path):
        """Adapters built from a pre-parsed ``BDDLProblem`` without
        ``bddl_source`` / ``bddl_path`` can't auto-generate a scene -
        the resolver returns None and the generator surfaces that as
        a no-op."""
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        problem = parse_bddl(PICK_CUBE_BDDL)
        # Construct directly, bypassing from_file / from_text.
        adapter = LiberoAdapter(
            problem,
            scene_cache_dir=str(tmp_path / "cache"),
            init_jitter=0.0,
        )
        resolved = adapter._resolve_bddl_path_for_libero()
        assert resolved is None

        # The generator must return None (not raise) so the on_episode_start
        # fallback path stays alive.
        assert adapter._generate_scene_from_bddl() is None

    def test_on_episode_start_loads_generated_scene(self, tmp_path):
        """End-to-end: scene_path=None, auto_generate_scene=True ⇒
        on_episode_start generates the scene, mutates self.scene_path,
        and calls sim.load_scene with the cached path."""
        cache_dir = tmp_path / "scene_cache"
        cache_dir.mkdir()

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            init_jitter=0.0,
            install_cameras=False,
        )

        # Pre-populate the cache so we don't need a fake libero env.
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        cache_path = cache_dir / f"{sha}.xml"
        cache_path.write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        adapter.on_episode_start(sim, random.Random(0))

        assert adapter.scene_path == str(cache_path)
        assert sim._scenes_loaded == [str(cache_path)]


class TestRenameMjcfCameras:
    """``_rename_mjcf_cameras`` helper - targeted regex, doesn't touch
    non-camera elements."""

    def test_renames_camera_by_name(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="agentview" pos="1 0 1"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert 'name="image"' in out
        assert 'name="agentview"' not in out

    def test_passes_through_unmapped_cameras(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="other_cam" fovy="60"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert out == xml

    def test_does_not_touch_material_named_after_camera(self):
        """Targeted regex: only ``<camera ... name="X" ...>`` is rewritten,
        not e.g. ``<material name="agentview">`` (a contrived case but it
        guards against future false-positives)."""
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<material name="agentview" rgba="1 0 0 1"/><camera name="agentview" pos="1 0 1"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        # Material name unchanged.
        assert '<material name="agentview"' in out
        # Camera name renamed.
        assert '<camera name="image"' in out

    def test_empty_aliases_is_noop(self):
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera name="agentview"/>'
        assert _rename_mjcf_cameras(xml, {}) == xml

    def test_handles_attributes_before_name(self):
        """Real LIBERO MJCFs put pose attributes before ``name="..."``."""
        from strands_robots.benchmarks.libero.adapter import _rename_mjcf_cameras

        xml = '<camera pos="1 0 1" fovy="60" name="agentview"/>'
        out = _rename_mjcf_cameras(xml, {"agentview": "image"})
        assert 'name="image"' in out


# Scene <keyframe> application (#166 follow-up)


class TestApplySceneKeyframe:
    """``LiberoAdapter._apply_canonical_state`` calls
    ``mj_resetDataKeyframe`` after every ``load_scene`` so qpos starts at
    the canonical home pose RoboSuite encoded in the MJCF ``<keyframe>``.
    Without this, ``mj_makeData`` and ``mj_resetData`` initialise from
    joint-default ``qpos0`` and free-joint objects (mugs, plates) snap
    to the origin - the #166 root cause for ``success_rate=0.00``.
    """

    def _make_mock_mujoco(self, calls: list[tuple[str, int]] | None = None):
        """Return a stub ``mujoco`` module that records mj_resetDataKeyframe calls."""
        if calls is None:
            calls = []

        class _Module:
            @staticmethod
            def mj_resetDataKeyframe(model, data, key):  # noqa: ARG004
                calls.append(("mj_resetDataKeyframe", int(key)))

            @staticmethod
            def mj_forward(model, data):  # noqa: ARG004
                calls.append(("mj_forward", 0))

        return _Module(), calls

    def test_applies_keyframe_when_model_has_one(self, tmp_path):
        """End-to-end: scene was loaded, model has nkey>0 ⇒ keyframe applied."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-populate the cache so on_episode_start uses the cached path.
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # mj_resetDataKeyframe(model, data, 0) MUST have fired.
        assert ("mj_resetDataKeyframe", 0) in calls
        # And mj_forward must follow so derived state reflects the new qpos
        # before the first observation / render.
        assert ("mj_forward", 0) in calls

    def test_no_keyframe_in_model_skips_application(self, tmp_path):
        """Model with ``nkey == 0`` AND data without qpos/qvel attrs (e.g. a
        non-MuJoCo backend) must skip the keyframe path entirely - no
        ``mj_resetDataKeyframe`` calls fire. The snapshot fallback also
        no-ops because the data object has no ``qpos`` to capture."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # no qpos/qvel  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Keyframe branch never fires (nkey==0); snapshot branch can't
        # capture because data has no qpos attr -> overall no-op.
        assert calls == []
        assert adapter._canonical_qpos is None

    def test_apply_scene_keyframe_false_disables(self, tmp_path):
        """Opt-out: apply_scene_keyframe=False skips the call even when
        the model declares a keyframe."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert calls == []

    def test_skipped_when_no_scene_loaded(self):
        """Bare-Panda fallback (no scene_path, auto_generate failed) must
        not try to apply a keyframe - there's no scene to apply it to."""

        class _FakeModel:
            nkey = 1

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,  # no scene generation
            apply_scene_keyframe=True,
        )
        # scene_path stays None -> load_scene never called -> no keyframe.

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert calls == []

    def test_custom_keyframe_index(self, tmp_path):
        """``scene_keyframe_index=1`` selects the second ``<keyframe>``."""

        class _FakeModel:
            nkey = 3

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            scene_keyframe_index=2,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        assert ("mj_resetDataKeyframe", 2) in calls

    def test_out_of_range_keyframe_index_skipped_with_warning(self, tmp_path, caplog):
        """``scene_keyframe_index >= model.nkey`` is a config error - log
        WARNING and skip, don't crash the episode."""

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            scene_keyframe_index=5,  # out of range
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with caplog.at_level("WARNING"):
            with patch.dict("sys.modules", {"mujoco": mock_mj}):
                adapter.on_episode_start(sim, random.Random(0))

        assert calls == []
        assert any("out of range" in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    def test_canonical_state_applied_before_install_cameras(self, tmp_path):
        """Order matters (#166 review finding): the canonical-state apply
        must run RIGHT AFTER load_scene, BEFORE super() / install_cameras
        get a chance to recompile and shift qpos away from canonical.
        Otherwise the snapshot is taken post-recompile - already non-
        canonical - and replays a non-canonical state on every reset.
        """

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=True,  # camera install runs
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = object()  # type: ignore[attr-defined]

        ordering: list[str] = []

        def track_add_camera(self_, name, **kw):  # noqa: ARG002
            ordering.append(f"add_camera:{name}")
            return {"status": "success", "content": [{"text": "ok"}]}

        mock_mj, _ = self._make_mock_mujoco()
        original = mock_mj.mj_resetDataKeyframe

        def tracked_keyframe(model, data, key):
            ordering.append(f"keyframe:{key}")
            return original(model, data, key)

        mock_mj.mj_resetDataKeyframe = tracked_keyframe

        with patch.object(FakeSim, "add_camera", track_add_camera, create=False):
            with patch.dict("sys.modules", {"mujoco": mock_mj}):
                adapter.on_episode_start(sim, random.Random(0))

        camera_indices = [i for i, e in enumerate(ordering) if e.startswith("add_camera:")]
        keyframe_indices = [i for i, e in enumerate(ordering) if e.startswith("keyframe:")]
        assert keyframe_indices, "expected the keyframe to be applied"
        # The keyframe call MUST precede every add_camera call - the
        # canonical state needs to land before any recompile-prone step.
        if camera_indices:
            assert max(keyframe_indices) < min(camera_indices), (
                f"canonical-state apply at {keyframe_indices} should precede all camera installs at {camera_indices}"
            )

    def test_no_model_skips_silently(self):
        """Backends without a compiled model attribute (future engines) skip
        the keyframe application without raising."""

        class _BareWorld:
            cameras: dict[str, Any] = {}

        class _BareSim:
            _world = _BareWorld()  # no _model / _data

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            install_cameras=False,
            auto_generate_scene=False,
        )
        # Direct exercise of the helper - must not raise.
        adapter._apply_canonical_state(_BareSim())  # type: ignore[arg-type]


# Snapshot-and-restore branch (#166 follow-up review)


class TestApplyCanonicalStateSnapshot:
    """``_apply_canonical_state`` falls back to snapshot-and-restore when
    ``model.nkey == 0`` (procedurally-generated MJCFs from #165 don't ship
    a ``<keyframe>``). First episode after super() + camera install
    captures ``data.qpos`` / ``data.qvel``; subsequent episodes restore
    via ``np.copyto`` + ``mj_forward``. This is what actually fixes #166's
    ``success_rate=0.00`` symptom on ``examples/libero_mujoco.py``.
    """

    def _make_mock_mujoco(self, calls: list[tuple[str, Any]] | None = None):
        """Stub ``mujoco`` that records mj_resetDataKeyframe + mj_forward calls."""
        if calls is None:
            calls = []

        class _Module:
            @staticmethod
            def mj_resetDataKeyframe(model, data, key):  # noqa: ARG004
                calls.append(("mj_resetDataKeyframe", int(key)))

            @staticmethod
            def mj_forward(model, data):  # noqa: ARG004
                calls.append(("mj_forward", None))

        return _Module(), calls

    def _make_data(self, qpos_vals: list[float], qvel_vals: list[float] | None = None):
        """Bare ``data``-like object with mutable ``qpos`` / ``qvel`` arrays."""
        import numpy as _np

        class _D:
            qpos: Any
            qvel: Any

        d = _D()
        d.qpos = _np.array(qpos_vals, dtype=_np.float64)
        d.qvel = _np.array(qvel_vals if qvel_vals is not None else [], dtype=_np.float64)
        return d

    def test_first_call_captures_snapshot_no_restore(self, tmp_path):
        """First episode after a scene compile: snapshot is captured but
        no ``np.copyto`` / ``mj_forward`` runs (nothing to restore TO yet)."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0, 2.0, 3.0], [0.1, 0.2])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot captured; no restore-side calls fired.
        assert adapter._canonical_qpos is not None
        assert list(adapter._canonical_qpos) == [1.0, 2.0, 3.0]
        assert adapter._canonical_qvel is not None
        assert list(adapter._canonical_qvel) == [0.1, 0.2]
        # Neither mj_resetDataKeyframe (nkey=0) nor mj_forward (no
        # restore on first capture) should have fired.
        assert calls == []

    def test_second_call_restores_from_snapshot_and_calls_mj_forward(self, tmp_path):
        """Second episode after the same scene compile: snapshot is
        ``np.copyto``'d into ``data.qpos`` and ``mj_forward`` runs so
        derived state reflects the canonical pose."""
        import numpy as _np

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed the snapshot to simulate "this is episode 2+".
        adapter._canonical_qpos = _np.array([10.0, 20.0, 30.0], dtype=_np.float64)
        adapter._canonical_qvel = _np.array([1.1, 2.2], dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        # Current data is at "post-rollout state" - non-canonical.
        sim._world._data = self._make_data([99.0, 99.0, 99.0], [9.9, 9.9])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # qpos / qvel must now equal the canonical snapshot, not the
        # 99.0 values that were there before.
        assert list(sim._world._data.qpos) == [10.0, 20.0, 30.0]
        assert list(sim._world._data.qvel) == [1.1, 2.2]
        # mj_forward MUST have fired so derived state (xpos/xquat) is current.
        assert ("mj_forward", None) in calls
        # mj_resetDataKeyframe MUST NOT have fired (nkey == 0).
        assert not any(c[0] == "mj_resetDataKeyframe" for c in calls)

    def test_snapshot_shape_mismatch_recaptures(self, tmp_path):
        """If a model recompile between episodes changed ``nq`` (unusual
        but possible), the snapshot-shape check re-captures instead of
        crashing on a copyto length mismatch."""
        import numpy as _np

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed a 5-element snapshot.
        adapter._canonical_qpos = _np.zeros(5, dtype=_np.float64)
        adapter._canonical_qvel = _np.zeros(4, dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        # Current data has 3-element qpos - shape mismatch.
        sim._world._data = self._make_data([1.0, 2.0, 3.0], [0.1, 0.2])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot was re-captured (now 3 elements).
        assert adapter._canonical_qpos.shape == (3,)
        assert list(adapter._canonical_qpos) == [1.0, 2.0, 3.0]
        # No mj_forward since this counts as a "first call" for the new shape.
        assert calls == []

    def test_apply_scene_keyframe_false_disables_snapshot_branch_too(self, tmp_path):
        """``apply_scene_keyframe=False`` MUST disable BOTH branches -
        keyframe AND snapshot. Otherwise users opting out can't actually
        opt out."""

        class _FakeModel:
            nkey = 0

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
            apply_scene_keyframe=False,  # opt out
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0, 2.0, 3.0])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Snapshot branch never ran.
        assert adapter._canonical_qpos is None
        assert calls == []

    def test_keyframe_branch_takes_priority_over_snapshot_when_nkey_positive(self, tmp_path):
        """When ``model.nkey > 0`` AND a snapshot has been previously
        captured, the keyframe branch wins - we don't restore an old
        snapshot over a model that explicitly declares its canonical
        state via a ``<keyframe>``."""
        import numpy as _np

        class _FakeModel:
            nkey = 1

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_cache_dir=str(cache_dir),
            install_cameras=False,
        )
        bddl_path = adapter._resolve_bddl_path_for_libero()
        assert bddl_path is not None
        sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        (cache_dir / f"{sha}.xml").write_text("<mujoco/>")

        # Pre-seed a stale snapshot.
        adapter._canonical_qpos = _np.array([99.0], dtype=_np.float64)
        adapter._canonical_qvel = _np.array([9.9], dtype=_np.float64)

        sim = FakeSim(data_config="panda")
        sim._world._model = _FakeModel()  # type: ignore[attr-defined]
        sim._world._data = self._make_data([1.0], [0.1])  # type: ignore[attr-defined]

        mock_mj, calls = self._make_mock_mujoco()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            adapter.on_episode_start(sim, random.Random(0))

        # Keyframe path fired; data NOT mutated by the snapshot-restore path.
        assert ("mj_resetDataKeyframe", 0) in calls
        # qpos was NOT clobbered with the stale snapshot value.
        assert list(sim._world._data.qpos) == [1.0]


# init_jitter default (#167 Probe 2 folded into #168 per review)


class TestInitJitterDefault:
    """``init_jitter`` defaults to 0.0 to match LIBERO's deterministic-reset
    convention. The GR00T-LIBERO checkpoint trains against fixed init
    states per ``(task, seed)``; positive default jitter pushes the
    policy slightly out-of-distribution from t=0.
    """

    def test_default_is_zero(self):
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL)
        assert adapter._init_jitter == 0.0

    def test_default_is_zero_via_from_file(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text(PICK_CUBE_BDDL)
        adapter = LiberoAdapter.from_file(p)
        assert adapter._init_jitter == 0.0

    def test_default_is_zero_via_direct_constructor(self):
        from strands_robots.benchmarks.libero.bddl_parser import parse_bddl

        problem = parse_bddl(PICK_CUBE_BDDL)
        adapter = LiberoAdapter(problem)
        assert adapter._init_jitter == 0.0

    def test_explicit_value_still_works(self):
        """``init_jitter=0.02`` still works for users who want
        per-episode randomization (e.g. evaluating *generalization*)."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=0.02)
        assert adapter._init_jitter == 0.02


# PolicyRunner + evaluate_benchmark integration


class TestEvaluateBenchmarkIntegration:
    def test_evaluate_with_mock_policy_succeeds(self):
        """Mock policy drives a loop; the benchmark loop returns a success_rate
        without crashing even though the mock policy doesn't actually win."""
        adapter = LiberoAdapter.from_text(PICK_CUBE_BDDL, max_steps=4)
        register_benchmark("libero-test-pick", adapter)
        # Sim is loaded with Panda but predicate positions don't match the
        # goal, so every episode should fall through to max_steps with
        # success=False - that's fine, we're testing the loop.
        sim = FakeSim(
            bodies={
                "cube_1": {"position": [0, 0, 0.0]},
                "plate_1": {"position": [0, 0, 0.0]},
            },
            data_config="panda",
        )
        # Rename so SimEngine.evaluate_benchmark can resolve the sole robot.
        sim._world.robots.clear()
        sim._world.robots["panda_arm"] = _FakeRobot("panda")

        result = sim.evaluate_benchmark(
            benchmark_name="libero-test-pick",
            policy_provider="mock",
            n_episodes=2,
            seed=0,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_episodes"] == 2
        assert payload["benchmark_class"] == "LiberoAdapter"

    def test_unknown_task_returns_structured_error(self):
        sim = FakeSim(data_config="panda")
        result = sim.evaluate_benchmark(benchmark_name="libero-nonexistent")
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_runner_counts_success_when_predicate_holds(self):
        """Seed the sim with predicate-satisfying state so is_success returns
        True on the first step - success_rate should be 1.0."""
        adapter = LiberoAdapter.from_text(
            "(define (problem p) (:goal (upright bottle)))",
            max_steps=3,
        )
        sim = FakeSim(
            bodies={"bottle": {"quaternion": [1.0, 0, 0, 0]}},
            data_config="panda",
        )
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_panda"))
        result = PolicyRunner(sim).evaluate("fake_panda", policy, spec=adapter, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 1
        assert payload["episodes"][0]["steps"] == 1  # terminated on first step
