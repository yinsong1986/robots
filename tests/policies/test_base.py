"""Tests for ``strands_robots.policies.base.Policy`` ABC contract.

Covers the ``get_actions_sync`` event-loop dispatch paths: the 'no loop'
fast path and the 'already-in-event-loop' ThreadPoolExecutor fallback.
"""

from __future__ import annotations

import asyncio
from typing import Any

from strands_robots.policies.base import Policy


class _IdentityPolicy(Policy):
    """Minimal concrete Policy for testing Policy ABC's sync wrapper."""

    def __init__(self) -> None:
        self._keys = ["j0"]

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0": 0.1}, {"j0": 0.2}]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    @property
    def provider_name(self) -> str:
        return "identity"


def test_get_actions_sync_outside_event_loop_uses_asyncio_run():
    p = _IdentityPolicy()
    actions = p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_get_actions_sync_inside_event_loop_uses_threadpool():
    """When called from within a running event loop, the sync wrapper must
    off-load to a thread pool instead of raising 'already in a loop'."""
    p = _IdentityPolicy()

    async def inner():
        # Calling the sync wrapper here forces the thread-pool branch
        return p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")

    actions = asyncio.run(inner())
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_provider_name_and_state_keys():
    p = _IdentityPolicy()
    assert p.provider_name == "identity"
    p.set_robot_state_keys(["a", "b", "c"])
    assert p._keys == ["a", "b", "c"]


def test_requires_images_default_is_true():
    """The base ABC defaults requires_images=True; subclasses opt out."""
    p = _IdentityPolicy()
    assert p.requires_images is True


def test_reset_default_is_noop():
    """Default reset() returns None and must be safe to call without a seed."""
    p = _IdentityPolicy()
    assert p.reset() is None
    assert p.reset(seed=42) is None


def test_well_known_kwargs_are_accepted_by_contract():
    """Non-VLA providers receive goals via ``**kwargs`` (target_pose,
    target_joints, world_update). The Policy contract requires get_actions
    to ignore unknown kwargs rather than raising, so callers can pass
    shared keys across providers without coupling to a backend."""
    from strands_robots.policies.mock import MockPolicy

    p = MockPolicy()
    p.set_robot_state_keys(["j0", "j1"])
    obs = {"observation.state": [0.0, 0.0]}

    # All three well-known kwargs together must round-trip cleanly through
    # the sync wrapper -- this is the smoke test that pins the documented
    # API surface for non-VLA providers.
    actions = p.get_actions_sync(
        obs,
        instruction="",
        target_pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        target_joints={"j0": 0.1, "j1": -0.2},
        world_update=None,
    )
    assert isinstance(actions, list) and actions, "Policy must return a non-empty action list"
    assert all(isinstance(a, dict) for a in actions)


def test_non_vla_providers_can_skip_camera_rendering():
    """``requires_images=False`` is the opt-out for joint-state-only
    providers (MockPolicy, planners, MPC, scripted)."""
    from strands_robots.policies.mock import MockPolicy

    assert MockPolicy().requires_images is False
