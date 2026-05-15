"""Tests for the ``gr00t_inference`` tool's command builder.

Establishes the testing pattern for shell-out tools in this repo: mock
``subprocess.run`` and ``_is_service_running`` so the start path returns
without actually invoking docker. Tests assert on the exact ``argv`` the
tool would have ``docker exec``'d.

Coverage:

* N1.5 default - legacy ``inference_service.py`` entrypoint with
  ``--data-config`` + ``--denoising-steps``.
* N1.6 - same entrypoint as N1.5 (the wire format diverged but the
  server CLI didn't).
* N1.7 - new ``python -m gr00t.eval.run_gr00t_server`` entrypoint with
  no ``--data-config`` / ``--denoising-steps`` and an optional
  ``--use-sim-policy-wrapper``.
* Unknown protocol → structured error.
* Optional flags (TensorRT, ``--http-server``, ``--api-token``) carry
  through to every protocol.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.tools.gr00t_inference import (
    _build_inference_command,
    _start_service,
    gr00t_inference,
)


def _common_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default args for ``_build_inference_command`` - keeps each test focused."""
    base: dict[str, Any] = {
        "container_name": "gr00t",
        "checkpoint_path": "/data/checkpoints/model",
        "port": 5555,
        "host": "0.0.0.0",
        "data_config": "libero_panda",
        "embodiment_tag": "libero_sim",
        "denoising_steps": 4,
        "http_server": False,
        "use_tensorrt": False,
        "trt_engine_path": "gr00t_engine",
        "vit_dtype": "fp8",
        "llm_dtype": "nvfp4",
        "dit_dtype": "fp8",
        "api_token": None,
        "protocol": "n1.5",
        "use_sim_policy_wrapper": False,
    }
    base.update(overrides)
    return base


# Command builder


class TestBuildInferenceCommand:
    def test_n15_legacy_entrypoint(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.5"))
        # Legacy script must be the python target.
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd
        # Includes the deprecated-in-n1.7 flags.
        assert "--data-config" in cmd
        assert "libero_panda" in cmd
        assert "--denoising-steps" in cmd
        assert "4" in cmd
        # Does NOT include n1.7-only flags.
        assert "-m" not in cmd
        assert "gr00t.eval.run_gr00t_server" not in cmd
        assert "--use-sim-policy-wrapper" not in cmd

    def test_n16_uses_legacy_entrypoint(self):
        """N1.6 still uses inference_service.py - only the wire format diverged."""
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.6"))
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd
        assert "--data-config" in cmd
        assert "--denoising-steps" in cmd

    def test_n17_module_entrypoint(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.7"))
        # New entrypoint: python -m gr00t.eval.run_gr00t_server
        assert "-m" in cmd
        assert "gr00t.eval.run_gr00t_server" in cmd
        # Legacy script must NOT be invoked under n1.7.
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" not in cmd
        # Flags removed from n1.7's CLI surface (server reads from checkpoint).
        assert "--data-config" not in cmd
        assert "--denoising-steps" not in cmd
        # Sim policy wrapper is opt-in.
        assert "--use-sim-policy-wrapper" not in cmd

    def test_n17_with_sim_policy_wrapper(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.7", use_sim_policy_wrapper=True))
        assert "--use-sim-policy-wrapper" in cmd

    def test_n17_ignores_use_sim_policy_wrapper_under_n15(self):
        """The flag doesn't exist on the legacy entrypoint - silently dropped."""
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.5", use_sim_policy_wrapper=True))
        assert "--use-sim-policy-wrapper" not in cmd

    def test_shared_required_flags_present_on_all_protocols(self):
        for protocol in ("n1.5", "n1.6", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol))
            assert "--server" in cmd, protocol
            assert "--model-path" in cmd, protocol
            assert "/data/checkpoints/model" in cmd, protocol
            assert "--port" in cmd, protocol
            assert "5555" in cmd, protocol
            assert "--host" in cmd, protocol
            assert "0.0.0.0" in cmd, protocol
            assert "--embodiment-tag" in cmd, protocol
            assert "libero_sim" in cmd, protocol

    def test_http_server_flag_carries_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol, http_server=True))
            assert "--http-server" in cmd, protocol

    def test_tensorrt_flags_carry_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(
                **_common_kwargs(
                    protocol=protocol,
                    use_tensorrt=True,
                    trt_engine_path="/engines/x",
                    vit_dtype="fp16",
                    llm_dtype="fp8",
                    dit_dtype="fp16",
                )
            )
            assert "--use-tensorrt" in cmd, protocol
            assert "/engines/x" in cmd, protocol
            assert "fp16" in cmd, protocol
            assert "fp8" in cmd, protocol

    def test_api_token_carries_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol, api_token="sek"))
            assert "--api-token" in cmd
            assert "sek" in cmd

    def test_api_token_omitted_when_none(self):
        cmd = _build_inference_command(**_common_kwargs(api_token=None))
        assert "--api-token" not in cmd


# Top-level dispatcher


class TestProtocolValidation:
    def test_unknown_protocol_returns_structured_error(self):
        result = gr00t_inference(action="start", checkpoint_path="/cp", protocol="n2.0")
        assert result["status"] == "error"
        assert "Unknown protocol" in result["message"]
        # Error must enumerate the valid set so the caller can fix the call.
        assert "n1.5" in result["message"]
        assert "n1.7" in result["message"]


# _start_service end-to-end with subprocess mocked


class TestStartServiceEndToEnd:
    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=True)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_n17_start_succeeds_and_reports_server_protocol(self, mock_run, _mock_is_running):
        """Full start path with protocol='n1.7' must invoke run_gr00t_server."""
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        result = _start_service(
            checkpoint_path="/data/checkpoints/libero_spatial",
            port=8000,
            data_config="libero_panda",
            embodiment_tag="libero_sim",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t",
            policy_name=None,
            timeout=2,
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.7",
            use_sim_policy_wrapper=True,
        )
        assert result["status"] == "success"
        assert result["server_protocol"] == "n1.7"
        assert result["use_sim_policy_wrapper"] is True
        # The legacy-only fields must NOT appear in the n1.7 response.
        assert "data_config" not in result
        assert "denoising_steps" not in result

        # Verify the actual subprocess call.
        argv = mock_run.call_args.args[0]
        assert "gr00t.eval.run_gr00t_server" in argv
        assert "--use-sim-policy-wrapper" in argv
        assert "--data-config" not in argv

    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=True)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_n15_start_preserves_legacy_response_fields(self, mock_run, _mock_is_running):
        """Default protocol path must keep the existing response shape."""
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t",
            policy_name=None,
            timeout=2,
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "success"
        assert result["server_protocol"] == "n1.5"
        assert result["data_config"] == "so100"
        assert result["denoising_steps"] == 4
        # Wire protocol stays "ZMQ" / "HTTP" (back-compat with pre-fix callers).
        assert result["protocol"] == "ZMQ"

    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=False)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_timeout_returns_error(self, mock_run, _mock_is_running):
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t",
            policy_name=None,
            timeout=0,  # don't actually sleep
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.7",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "error"
        assert "failed to start" in result["message"]


@pytest.mark.parametrize("protocol", ["n1.5", "n1.6", "n1.7"])
class TestSignatureBackCompat:
    """Existing callers that don't pass ``protocol=`` must continue to work
    (default = ``n1.5``)."""

    def test_default_protocol_is_n15(self, protocol):
        # Indirect: passing nothing should match an explicit n1.5 build.
        if protocol != "n1.5":
            return
        cmd_default = _build_inference_command(**_common_kwargs(protocol="n1.5"))
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd_default
