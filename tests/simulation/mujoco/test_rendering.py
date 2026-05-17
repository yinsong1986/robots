"""Tests for multi-camera snapshot + background recording."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

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


# Recorder warmup before thread launch (#168 round-11 bug D)


@_requires_mujoco
def test_recorder_thread_warms_up_renderer_before_capture_loop() -> None:
    """The recorder thread does TWO synchronous renders per camera at
    the START of its ``_loop`` body, BEFORE the timing loop begins
    capturing into buffers.

    Why two passes (round-13 fix): MuJoCo's shared ``Renderer``
    rebinds the active camera on each ``update_scene(camera=X)``
    call. The FIRST render after a camera switch returns a
    cold-start readback even if the GL context is warm. With one
    pass per camera (the round-12 attempt), warming ended on the
    LAST camera and the first capture render of the FIRST camera
    cold-started again, producing a skybox-only gradient at t=0.
    Two passes guarantee every camera has had two consecutive
    renders by the time capture starts; whichever camera is rendered
    first in capture iter 1 has already been warmed twice and lands
    on the warm path.

    Why thread-side and not main-thread (round-11 vs round-12):
    MuJoCo's ``mujoco.GLContext.make_current()`` binds to the
    calling thread. A warmup performed in the main thread (i.e.
    before ``state["thread"].start()``) doesn't propagate to the
    daemon thread - the daemon has its own cold-start GL context on
    its first call. Round 11 attempted main-thread warmup; round-11
    verification confirmed t=0 frame still rendered as a gradient
    because of this thread boundary.

    Test mechanism: mock ``threading.Thread`` to capture the
    ``_loop`` target without starting it, then invoke ``_loop``
    directly with ``state["running"]`` pre-set to False so the
    timing loop exits immediately after warmup. Render calls
    accumulated during the synchronous invocation are exactly the
    warmup renders (two per camera = 2 x n_cameras).
    """
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_a", position=[0.5, 0, 0.5], target=[0, 0, 0])
    sim.add_camera("cam_b", position=[-0.5, 0, 0.5], target=[0, 0, 0])

    render_calls: list[str] = []
    original_render = sim.render

    def counting_render(camera_name: str, width=None, height=None) -> dict:
        render_calls.append(camera_name)
        return original_render(camera_name=camera_name, width=width, height=height)

    captured_targets: list = []

    class _CaptureThread:
        def __init__(self, target=None, daemon=None) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            captured_targets.append(self.target)

        def is_alive(self) -> bool:
            return False

        def join(self, timeout=None) -> None:
            pass

    with patch.object(sim, "render", side_effect=counting_render):
        with patch("threading.Thread", _CaptureThread):
            r = sim.start_cameras_recording(
                cameras=["cam_a", "cam_b"],
                output_dir="/tmp",
                fps=20,
                width=64,
                height=48,
                name="warmup_test",
            )
            assert r["status"] == "success", r
            # Round 12: NO render calls have happened yet - warmup is
            # inside _loop, which the mocked Thread didn't start.
            assert render_calls == [], f"unexpected render calls before _loop ran: {render_calls}"
            # The thread target was captured.
            assert len(captured_targets) == 1

            # Stop the timing loop before invoking _loop so it returns
            # immediately after the warmup. Render calls collected
            # below are exclusively warmup calls.
            sim._cams_rec_state["running"] = False
            captured_targets[0]()  # invoke _loop synchronously

    # Round 13 contract: TWO warmup passes, in scan order.
    # cam_a, cam_b, cam_a, cam_b - 4 total renders, alternating.
    assert render_calls == ["cam_a", "cam_b", "cam_a", "cam_b"], (
        f"expected two warmup passes (4 renders alternating), got {render_calls}"
    )

    sim.destroy()


@_requires_mujoco
def test_recorder_thread_warmup_failure_does_not_abort() -> None:
    """If the thread-side warmup render raises, the timing loop still
    starts (and accumulates ``state['errors'][cam]`` per the standard
    error-tracking path).

    Required because warmup failure shouldn't crash the whole
    recorder thread; the timing loop's exception handler will
    surface persistent failures via
    :meth:`get_cameras_recording_status`."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.5, 0, 0.5], target=[0, 0, 0])

    def boom(camera_name: str, width=None, height=None):
        raise RuntimeError(f"simulated warmup failure on {camera_name}")

    captured_targets: list = []

    class _CaptureThread:
        def __init__(self, target=None, daemon=None) -> None:
            self.target = target

        def start(self) -> None:
            captured_targets.append(self.target)

        def is_alive(self) -> bool:
            return False

    with patch.object(sim, "render", side_effect=boom):
        with patch("threading.Thread", _CaptureThread):
            r = sim.start_cameras_recording(
                cameras=["cam"],
                output_dir="/tmp",
                fps=20,
                width=64,
                height=48,
                name="warmup_fail_test",
            )
            assert r["status"] == "success", r
            assert len(captured_targets) == 1
            # Stop the timing loop before invoking _loop so it returns
            # after warmup attempts (which all raise).
            sim._cams_rec_state["running"] = False
            # Must not raise even though warmup renders all raise.
            captured_targets[0]()

    sim.destroy()


@_requires_mujoco
def test_recorder_first_frame_is_real_geometry(tmp_path: Path) -> None:
    """End-to-end: the FIRST frame written to the MP4 is real geometry
    (col-std > 30), not the skybox-only gradient (col-std ~0.6).

    Pin for #168 round-11 bug D: pre-fix, the recorder thread's first
    captured frame was a cold-start gradient because the renderer's
    scene buffer hadn't been populated. The synchronous warmup before
    thread launch ensures the first thread-side render() lands on the
    warm path.

    Gated behind ``_requires_mujoco`` because it needs a real GL
    context to actually render meaningful pixels."""
    os.environ.setdefault("MUJOCO_GL", "glfw")

    import imageio.v2 as imageio
    import numpy as np

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.4, -0.4, 0.5], target=[0.0, 0.0, 0.1])
    sim.step(n_steps=5)

    # Brand-new recorder; first frame must be warm.
    r = sim.start_cameras_recording(
        cameras=["cam"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="first_frame_test",
    )
    assert r["status"] == "success", r

    time.sleep(0.2)
    sim.stop_cameras_recording()

    # Decode the MP4 and inspect the first frame's column stddev.
    mp4_files = list(tmp_path.glob("*.mp4"))
    assert mp4_files, "expected at least one mp4 file"
    reader = imageio.get_reader(str(mp4_files[0]))
    try:
        frames: list = []
        for frame in reader.iter_data():  # type: ignore[attr-defined]
            frames.append(frame)
    finally:
        reader.close()
    assert len(frames) >= 1
    first = np.asarray(frames[0])
    # Real-geometry frames have col-std around 30+; cold-start gradient
    # frames have col-std around 0.6 (dominated by the smooth horizontal
    # skybox blue->grey gradient).
    col_std = float(first.std(axis=0).mean())
    assert col_std > 5.0, (
        f"first frame appears to be cold-start gradient, col_std={col_std:.2f}; "
        f"expected real geometry (col_std>5). Frame mean RGB: "
        f"({first[..., 0].mean():.1f}, {first[..., 1].mean():.1f}, {first[..., 2].mean():.1f})"
    )

    sim.destroy()
