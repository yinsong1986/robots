"""Tests for multi-camera snapshot + background recording."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_requires_mujoco = pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)


@_requires_mujoco
def test_render_all_returns_every_camera(tmp_path: Path) -> None:
    """render_all() should return one image block per camera."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    # add 3 cameras
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_c", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.step(n_steps=5)

    r = sim.render_all(width=64, height=48)
    assert r["status"] == "success", r
    image_blocks = [c for c in r["content"] if isinstance(c, dict) and "image" in c]
    # Should include at least the 3 user-added cameras (plus any default)
    assert len(image_blocks) >= 3, f"expected >=3 image blocks, got {len(image_blocks)}"

    # Subset mode
    r2 = sim.render_all(cameras=["cam_a", "cam_c"], width=48, height=32)
    assert r2["status"] == "success"
    imgs = [c for c in r2["content"] if isinstance(c, dict) and "image" in c]
    assert len(imgs) == 2

    sim.destroy()


@_requires_mujoco
def test_start_stop_cameras_recording_writes_one_mp4_per_camera(tmp_path: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("top", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.add_camera("side", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.step(n_steps=5)

    r = sim.start_cameras_recording(
        cameras=["top", "side"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="integ_test",
    )
    assert r["status"] == "success", r

    # Let it record for ~0.4s of wall time
    time.sleep(0.4)

    status = sim.get_cameras_recording_status()
    assert status["status"] == "success"
    assert "🟢" in status["content"][0]["text"]

    stop = sim.stop_cameras_recording()
    assert stop["status"] == "success"

    # Two MP4 files should exist
    files = sorted(tmp_path.glob("*.mp4"))
    names = [f.name for f in files]
    assert any("top" in n for n in names), names
    assert any("side" in n for n in names), names
    for f in files:
        assert f.stat().st_size > 0, f"empty file: {f}"

    sim.destroy()


@_requires_mujoco
def test_stop_without_start_is_idempotent() -> None:
    """T16: idempotent - stop_cameras_recording without a running recording
    returns success with 'Was not recording' instead of erroring."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.stop_cameras_recording()
    assert r["status"] == "success"
    assert "Was not recording" in r["content"][0]["text"]
    sim.destroy()


@_requires_mujoco
def test_status_when_idle_is_success() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.get_cameras_recording_status()
    assert r["status"] == "success"
    assert "⚪" in r["content"][0]["text"]


# Render-time scene_option pass-through (#168 round 9 bug E)


def test_get_viz_option_returns_none_when_unset() -> None:
    """``RenderingMixin._get_viz_option`` returns ``None`` when no
    adapter has populated ``world._backend_state['viz_option']``. This
    is the default path for non-LIBERO sims; ``Renderer.update_scene``
    accepts ``scene_option=None`` as the no-op meaning."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    assert sim._get_viz_option() is None
    sim.destroy()


def test_get_viz_option_reads_from_backend_state() -> None:
    """When an adapter (e.g. LiberoAdapter) sets
    ``world._backend_state['viz_option']``, the rendering layer reads it
    via ``_get_viz_option`` and threads it through to
    ``Renderer.update_scene(scene_option=...)``."""
    import mujoco

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[0] = 0  # any non-default value to verify pass-through
    assert sim._world is not None
    sim._world._backend_state["viz_option"] = opt

    retrieved = sim._get_viz_option()
    assert retrieved is opt
    assert int(retrieved.geomgroup[0]) == 0
    sim.destroy()


def test_get_viz_option_handles_missing_backend_state() -> None:
    """Defensive: ``world._backend_state`` not being a dict (e.g. unusual
    test stub) returns None silently rather than raising."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    # Force backend_state into an unexpected shape.
    assert sim._world is not None
    sim._world._backend_state = "oops not a dict"  # type: ignore[assignment]
    assert sim._get_viz_option() is None
    sim.destroy()


def test_get_viz_option_handles_missing_world() -> None:
    """``self._world is None`` -> _get_viz_option returns None.
    Reaches this state pre-create_world; defensive guard."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    # Don't call create_world - self._world stays None.
    assert sim._get_viz_option() is None


@_requires_mujoco
def test_render_passes_scene_option_to_renderer(tmp_path: Path) -> None:
    """End-to-end: ``Simulation.render(camera_name=...)`` reads
    viz_option from backend_state and threads it through to the
    underlying ``Renderer.update_scene`` so the rendered frame
    reflects the option (e.g. group=0 geoms hidden).

    Pin for #168 round 9: the round-9 fix moves collision-geom hiding
    from MJCF rgba edits to renderer-level mjvOption. Without
    backend_state -> Renderer threading, viz_option would be
    populated by adapters but ignored at render time."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])

    # Default render (no viz_option) - just verify it returns success.
    default = sim.render(camera_name="cam", width=64, height=64)
    assert default["status"] == "success"

    # Install a viz_option that turns OFF every visible group. The
    # rendered frame should be uniform / nearly-empty (no geoms drawn).
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for g in range(6):
        opt.geomgroup[g] = 0  # hide ALL geom groups
    for sg in range(6):
        opt.sitegroup[sg] = 0
    assert sim._world is not None
    sim._world._backend_state["viz_option"] = opt

    masked = sim.render(camera_name="cam", width=64, height=64)
    assert masked["status"] == "success"
    # The rendered image should have very low pixel variance now that
    # everything is hidden - just background sky / floor.
    masked_var = next(
        c["json"]["pixel_variance"]
        for c in masked["content"]
        if isinstance(c, dict) and "json" in c and "pixel_variance" in c["json"]
    )
    default_var = next(
        c["json"]["pixel_variance"]
        for c in default["content"]
        if isinstance(c, dict) and "json" in c and "pixel_variance" in c["json"]
    )
    # Hiding everything should significantly reduce variance.
    assert masked_var < default_var, (
        f"viz_option not honoured: masked variance {masked_var} should be < default {default_var}"
    )

    sim.destroy()
    sim.destroy()
