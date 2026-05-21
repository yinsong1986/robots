"""End-to-end ZMQ wire-format regression test for ``Gr00tPolicy`` SERVICE mode.

Originally requested in #169 (closed by #172) and re-asked in #187 after
the success-rate gap reopened (in-process Gr00tPolicy=5/5 vs ZMQ docker
server=1-3/5 on the same checkpoint + task + seed).

What this catches: the *client-side* half of the pipeline (everything in
``Gr00tPolicy._service_get_actions`` + ``_build_service_observation`` +
``Gr00tInferenceClient.get_action``) that happens BEFORE the request hits
the network. Specifically:

- The exact bytes msgpack-packed onto the wire match the format upstream's
  ``run_gr00t_server`` expects for ``--use-sim-policy-wrapper`` (n1.7) and
  the legacy ``inference_service.py`` (n1.5 / n1.6).
- A realistic LIBERO observation built by ``LiberoAdapter.augment_observation``
  flows through ``_build_service_observation`` and lands as the canonical
  ``(B=1, T=1, ...)`` n1.7 wire shape with the right dtypes.
- The action chunk a real GR00T server returns (``{action.x: (1, 16, 1) f32, ...}``)
  unpacks via ``_unpack_service_actions`` into 16 per-step dicts that
  ``_LiberoOSCController.apply`` can consume without raising on the
  ``[value]`` packing produced by the no-mapping path.
- ``image_rotation_180`` lands on H/W (axes -3 / -2) for the n1.7 5-D
  wire shape, not on B/T.

What this does NOT catch: any divergence between the client-side payload
and what the *server-side* SimPolicyWrapper actually feeds the model.
That gap is the second half of #187 and requires a real GPU + container
to bisect (see :func:`tests_integ/.../test_libero_10_scene5_mujoco_engine_success_rate`
for the in-process equivalent).

Why a dedicated module: keeps the regression test discoverable under a
single class so future #169-style breakage shows up as a focused failure
("ZMQ wire round-trip on libero_panda regressed") instead of being lost
in the 1000-line ``test_policy.py``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

# E402: importorskip must execute before these imports to skip cleanly.
from strands_robots.policies.groot.client import MsgSerializer  # noqa: E402
from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402

# Canonical libero_panda channels from data_configs.json (state.x/y/z/roll/pitch/yaw/gripper +
# video.image / video.wrist_image + annotation.human.action.task_description). Hard-coded
# instead of imported so a future data_configs.json change shows up here as a
# loud failure rather than silent drift.
_LIBERO_STATE_BARE = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_LIBERO_VIDEO_BARE = ("image", "wrist_image")
_LIBERO_LANG_KEY = "annotation.human.action.task_description"


def _libero_observation(h: int = 64, w: int = 64) -> dict:
    """Construct a realistic LIBERO observation as ``LiberoAdapter.augment_observation``
    emits it: scalar pose channels, 2-element gripper, 3-D V-flipped uint8 frames."""
    rng = np.random.default_rng(seed=0)
    obs: dict = {}
    # Cartesian pose (scalars; ``_build_service_observation`` promotes 0-d to (1,)).
    obs["x"] = 0.123
    obs["y"] = -0.305
    obs["z"] = 0.443
    obs["roll"] = 3.139
    obs["pitch"] = 0.001
    obs["yaw"] = -0.005
    # Gripper: 2-element array (the post-#168 RoboSuite contract).
    obs["gripper"] = [0.0208, -0.0208]
    # Video: 3-D uint8, already V-flipped to OpenGL convention by the adapter.
    obs["image"] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    obs["wrist_image"] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return obs


def _server_action_chunk(horizon: int = 16) -> dict:
    """Construct a realistic n1.7 server response: per-key (B=1, T=horizon, D=1)
    float32 chunks for each of the 7 LIBERO action channels."""
    chunk: dict = {}
    rng = np.random.default_rng(seed=42)
    for bare in _LIBERO_STATE_BARE:  # libero_panda action keys mirror state keys
        chunk[f"action.{bare}"] = rng.standard_normal((1, horizon, 1)).astype(np.float32)
    return chunk


class TestZmqWireRoundTripLiberoPanda:
    """Pin the SERVICE-mode ZMQ pipeline end-to-end for ``data_config='libero_panda'``.

    Each test exercises the full pipeline at the boundary that the issue
    cited: agent-side ``policy.get_actions(libero_obs, instruction)`` →
    msgpack pack → wire bytes → mock server response → msgpack unpack →
    per-step list. The mock server is a lambda that intercepts
    ``client.socket.send`` / ``recv``, so the test runs in-process at
    micro-second latency without bringing up a real ZMQ socket pair.

    Same fixture pattern as the existing ``test_call_endpoint_*`` tests
    in ``test_client.py``; no new infrastructure.
    """

    def _make_libero_policy(self) -> Gr00tPolicy:
        """Build a SERVICE-mode policy targeting libero_panda + n1.7 wire format.

        Port=19999 is the convention used elsewhere in this suite for
        no-network construction (the connect call doesn't actually
        contact the port; only the first ``send`` would).
        """
        p = Gr00tPolicy(data_config="libero_panda", host="localhost", port=19999)
        # Force n1.7 wire shape (3-D state → 5-D after fanout). Without
        # this the test would target the legacy n1.5/n1.6 4-D video /
        # 2-D state path which is not what the docker server in #187
        # accepts.
        p._groot_version = "n1.7"
        return p

    def _capture_send_decode_recv(self, policy: Gr00tPolicy, response: dict | tuple) -> list[dict]:
        """Replace the client's send/recv with capturing stubs.

        Returns a list that gets populated with the *decoded* request
        dicts (one per ``call_endpoint`` round-trip). The recv stub
        returns ``response`` msgpack-packed, mimicking what a real
        n1.7 server emits via ``run_gr00t_server``.
        """
        sent: list[dict] = []

        def _capture_send(data: bytes) -> None:
            sent.append(MsgSerializer.from_bytes(data))

        # msgpack accepts both dict and tuple at the top level; the
        # n1.6 / n1.7 server returns a (action, info) tuple. Cast to
        # ``dict`` to satisfy the type signature; the wire format is
        # the same whether we pass a tuple or a dict here.
        packed = MsgSerializer.to_bytes(response)  # type: ignore[arg-type]
        assert policy._client is not None, "policy must be in service mode for this test"
        policy._client.socket.send = _capture_send  # type: ignore[assignment]
        policy._client.socket.recv = lambda: packed  # type: ignore[assignment]
        return sent

    def test_wire_payload_has_canonical_libero_panda_schema(self):
        """The msgpack payload sent to the n1.7 server contains exactly the
        13 keys ``run_gr00t_server`` expects: 7 ``state.*`` (B=1,T=1,D=1) f32,
        2 ``video.*`` (B=1,T=1,H,W,C) u8, and 1 ``annotation.*`` list[str].

        Pre-#187 a stale wire format (e.g. missing T axis on state, or
        scalar ``state.gripper`` instead of (B,T,2)) silently shipped a
        request the server rejected with "expected shape (B,T,D), got
        (B,D)" or accepted but fed OOD data to the policy.
        """
        policy = self._make_libero_policy()
        sent = self._capture_send_decode_recv(policy, (_server_action_chunk(), {}))

        asyncio.run(policy.get_actions(_libero_observation(), "pick the cube"))

        assert len(sent) == 1, "exactly one ZMQ round-trip expected per get_actions call"
        request = sent[0]
        assert request["endpoint"] == "get_action"
        observation = request["data"]["observation"]

        # State: 7 keys, each (B=1, T=1, D=1) for scalar channels or
        # (B=1, T=1, D=2) for the gripper finger pair.
        for bare in _LIBERO_STATE_BARE:
            key = f"state.{bare}"
            assert key in observation, f"missing required state key {key!r} on the wire"
            arr = observation[key]
            assert arr.dtype == np.float32, (
                f"{key}: dtype must be float32 (n1.7 server rejects float64), got {arr.dtype}"
            )
            if bare == "gripper":
                assert arr.shape == (1, 1, 2), f"state.gripper: expected (B=1,T=1,D=2), got {arr.shape}"
            else:
                assert arr.shape == (1, 1, 1), f"{key}: expected (B=1,T=1,D=1), got {arr.shape}"

        # Video: 2 keys, each (B=1, T=1, H, W, C=3) uint8.
        for bare in _LIBERO_VIDEO_BARE:
            key = f"video.{bare}"
            assert key in observation, f"missing required video key {key!r} on the wire"
            arr = observation[key]
            assert arr.dtype == np.uint8, f"{key}: dtype must be uint8, got {arr.dtype}"
            assert arr.ndim == 5, f"{key}: expected 5-D (B,T,H,W,C), got {arr.ndim}-D ({arr.shape})"
            assert arr.shape[0] == 1 and arr.shape[1] == 1
            assert arr.shape[-1] == 3, f"{key}: trailing channel axis must be 3, got {arr.shape[-1]}"

        # Language: list[str] of length 1.
        assert _LIBERO_LANG_KEY in observation
        assert observation[_LIBERO_LANG_KEY] == ["pick the cube"]

        # Envelope: ``options`` must be present and None so the server's
        # ``policy.get_action(observation, options)`` kwargs spread works.
        assert request["data"]["options"] is None

    def test_image_rotation_lands_on_hw_after_fanout(self):
        """``image_rotation_180`` must rotate H/W axes regardless of where
        the leading B/T axes are added.

        Pre-#172 the rotation was applied AFTER the newaxis fanout via
        ``arr[:, ::-1, ::-1, :]`` which targeted the wrong axes for the
        n1.7 5-D shape, sending the policy upside-down images and
        ``success_rate=0`` against the docker server. Post-#172 the
        helper uses negative-axis indexing
        (``np.flip(arr, axis=(-3, -2))``) so it works on any leading-axis
        count.
        """
        policy = self._make_libero_policy()
        sent = self._capture_send_decode_recv(policy, (_server_action_chunk(), {}))

        h, w = 32, 32
        # Inject a unique colour at the top-left so we can detect a 180°
        # rotation end-to-end.
        obs = _libero_observation(h=h, w=w)
        obs["image"] = np.zeros((h, w, 3), dtype=np.uint8)
        obs["image"][0, 0] = [123, 45, 67]  # marker pixel
        obs["wrist_image"] = np.zeros((h, w, 3), dtype=np.uint8)

        asyncio.run(policy.get_actions(obs, "t"))

        wire_image = sent[0]["data"]["observation"]["video.image"]
        # Marker that started at top-left (0, 0) must end up at
        # bottom-right (h-1, w-1) after the 180° rotation lands on H/W.
        np.testing.assert_array_equal(wire_image[0, 0, h - 1, w - 1], [123, 45, 67])
        # And NOT at the original top-left position (would mean axes
        # B/T got flipped instead of H/W — the #169 / #172 bug).
        assert not np.array_equal(wire_image[0, 0, 0, 0], [123, 45, 67])

    def test_action_chunk_unpacks_to_horizon_dicts_with_libero_keys(self):
        """The server's ``(action.x: (1, 16, 1) f32, …)`` chunk must
        unpack into 16 per-step dicts containing all 7 bare LIBERO
        action keys (``x``, ``y``, …, ``gripper``), each value packed as
        a 1-element list.

        ``_LiberoOSCController.apply`` calls ``_to_scalar`` on each
        channel which handles either ``[value]`` or scalar input; this
        test pins the contract so a future refactor that drops
        ``[value]`` in favour of scalar (or vice versa) is caught
        before it hits production eval.
        """
        policy = self._make_libero_policy()
        chunk = _server_action_chunk(horizon=16)
        self._capture_send_decode_recv(policy, (chunk, {}))

        actions = asyncio.run(policy.get_actions(_libero_observation(), "t"))

        assert len(actions) == 16, f"expected 16 per-step dicts (horizon=16), got {len(actions)}"
        for step_idx, step in enumerate(actions):
            for bare in _LIBERO_STATE_BARE:  # action keys mirror state keys for libero_panda
                assert bare in step, f"step {step_idx}: missing {bare!r}; got keys={sorted(step)}"
                value = step[bare]
                # No-mapping path packs each per-step value as ``[v]``
                # (1-element list). ``_LiberoOSCController._to_scalar``
                # tolerates both forms; pin the current packing so it
                # doesn't change without a coordinated update.
                assert isinstance(value, list), f"step {step_idx}.{bare}: expected list, got {type(value).__name__}"
                assert len(value) == 1, f"step {step_idx}.{bare}: expected 1-element list, got {len(value)}"
                assert isinstance(value[0], float), (
                    f"step {step_idx}.{bare}: element must be float (after .tolist()), got {type(value[0]).__name__}"
                )

    def test_action_chunk_values_are_finite_and_within_libero_bounds(self):
        """Sanity-check that the unpacked floats are JSON-serializable
        finite numbers in the LIBERO Cartesian-delta action range
        (typically [-0.1, 0.1] m for x/y/z, [-0.5, 0.5] rad for
        roll/pitch/yaw, [-1, 1] for the binarised gripper).

        This catches a class of wire bugs where dtype confusion
        (float64 ↔ float32 ↔ float16 on a TensorRT engine) silently
        produces NaN/Inf that then propagate through the OSC and crash
        the eval mid-rollout. Pinning ``np.isfinite`` post-unpack means
        the regression surfaces at this test rather than as a mystery
        ``mj_step`` divergence.
        """
        policy = self._make_libero_policy()
        # Use a chunk with deliberately-bounded values so the test
        # asserts the path doesn't smuggle in random NaNs through
        # msgpack np.save / np.load corruption.
        chunk = {f"action.{bare}": np.full((1, 16, 1), 0.05, dtype=np.float32) for bare in _LIBERO_STATE_BARE}
        self._capture_send_decode_recv(policy, (chunk, {}))

        actions = asyncio.run(policy.get_actions(_libero_observation(), "t"))

        for step in actions:
            for bare, value in step.items():
                v = value[0] if isinstance(value, list) else value
                assert np.isfinite(v), f"non-finite action value: {bare}={v!r}"
                assert abs(v - 0.05) < 1e-6, f"{bare}: round-trip drift: expected 0.05, got {v!r}"

    def test_request_envelope_matches_n17_server_contract(self):
        """The n1.7 ``run_gr00t_server`` PolicyServer spreads
        ``request['data']`` as kwargs into
        ``policy.get_action(observation, options)``. The envelope must
        therefore contain EXACTLY the keys ``observation`` and
        ``options``; any extra key produces ``TypeError: get_action()
        got an unexpected keyword argument`` server-side.

        Pre-#172 a regression here would only surface as the server
        log message which most users never see (logs stay inside the
        docker container). Pin it here so the failure mode is a clear
        unit-test failure on the dev box.
        """
        policy = self._make_libero_policy()
        sent = self._capture_send_decode_recv(policy, (_server_action_chunk(), {}))

        asyncio.run(policy.get_actions(_libero_observation(), "t"))

        request = sent[0]
        assert set(request["data"].keys()) == {"observation", "options"}
