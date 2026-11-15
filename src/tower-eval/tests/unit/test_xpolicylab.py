# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from tower_eval.policies.smoke import run_xpolicylab_smoke, smoke_observation
from tower_eval.policies.xpolicylab import client_root, decode_chunk, observation_to_xpolicylab


def test_observation_mapping_preserves_encoders_and_rgb():
    obs = smoke_observation()
    obs["joint_encoder"][0, 1] = 0.4321
    obs["gripper_width_m"][1] = -0.0004  # Validation belongs to the package's observation policy.
    mapped = observation_to_xpolicylab(obs)
    np.testing.assert_array_equal(mapped["state"]["joint_encoder"], obs["joint_encoder"])
    np.testing.assert_array_equal(mapped["state"]["gripper_width_m"], obs["gripper_width_m"])
    for name, pixels in obs["images"].items():
        np.testing.assert_array_equal(mapped["vision"][name]["color"], pixels)
    obs["joint_encoder"][0, 1] = 0
    assert mapped["state"]["joint_encoder"][0, 1] == 0.4321


def test_physical_observations_are_rejected():
    with pytest.raises(ValueError, match="encoder_arm_base"):
        observation_to_xpolicylab(dict(smoke_observation(), coordinate_frame="physical"))


def joint_row():
    return dict(
        left_arm_joint_state=np.arange(7) / 10,
        right_arm_joint_state=-np.arange(7) / 10,
        left_ee_joint_state=[0.02],
        right_ee_joint_state=[0.08],
    )


def test_joint_order_and_no_j2_conversion():
    chunk = decode_chunk([joint_row()], "joint_absolute")
    np.testing.assert_array_equal(
        chunk.actions[0], np.r_[np.arange(7) / 10, 0.02, -np.arange(7) / 10, 0.08]
    )
    assert chunk.dt_s == 0.1


@pytest.mark.parametrize(
    "change",
    [
        {"coordinate_frame": "world"},
        {"mode": "eef_delta"},
        {"control_hz": 20},
        {"units": {"joint": "deg"}},
        {"j8_actuator_mask": [True, False]},
        {"execute_steps": 2},
        {"right_ee_joint_state": [-0.01]},
        {"left_arm_joint_state": [float("nan")] * 7},
    ],
)
def test_action_contract_mismatches_rejected(change):
    with pytest.raises(ValueError):
        decode_chunk([dict(joint_row(), **change)], "joint_absolute")


def test_eef_targets_retain_absolute_anchor_and_precision():
    anchor = np.tile(np.eye(4), (2, 1, 1))
    anchor[0, 0, 3] = 0.123456789123
    target = anchor.copy()
    target[:, 1, 3] = 0.234567891234
    row = dict(
        joint_row(),
        mode="eef_delta",
        coordinate_frame="encoder_arm_base",
        left_ee_transform=target[0],
        right_ee_transform=target[1],
        chunk_anchor_encoder=anchor,
    )
    decoded = decode_chunk([row], "eef_delta", expected_anchor=anchor).actions[0]
    np.testing.assert_array_equal(decoded["eef_encoder_targets"], target)
    np.testing.assert_array_equal(decoded["chunk_anchor_encoder"], anchor)
    row.pop("coordinate_frame")
    with pytest.raises(ValueError, match="explicit encoder"):
        decode_chunk([row], "eef_delta", expected_anchor=anchor)


def test_demo_policy_loopback(capsys):
    pytest.importorskip("msgpack_numpy")
    pytest.importorskip("websockets.asyncio.server")
    try:
        client_root()
    except RuntimeError:
        pytest.skip("XPolicyLab submodule unavailable; run git submodule update --init")
    assert run_xpolicylab_smoke()["status"] == "ok"
    assert capsys.readouterr().out == ""


def eef_row(anchor):
    return dict(
        joint_row(),
        mode="eef_delta",
        coordinate_frame="encoder_arm_base",
        left_ee_transform=anchor[0],
        right_ee_transform=anchor[1],
        chunk_anchor_encoder=anchor.copy(),
    )


def fake_adapter(chunks):
    from tower_eval.policies.xpolicylab import XPolicyLabAdapter

    class Client:
        def __init__(self):
            self.calls = []

        def call(self, name, payload=None):
            self.calls.append((name, payload))
            if name in {"get_action", "get_action_batch"}:
                return chunks
            return None

        def close(self):
            self.calls.append(("close", None))

    adapter = XPolicyLabAdapter.__new__(XPolicyLabAdapter)
    adapter.mode = "eef_delta"
    adapter._client = Client()
    adapter._episode_active = False
    adapter._closed = False
    return adapter


def test_request_anchor_exact_and_consistent_on_every_row():
    obs = smoke_observation()
    row = eef_row(obs["eef_encoder"])
    adapter = fake_adapter([row, eef_row(obs["eef_encoder"])])
    assert len(adapter.act(obs).actions) == 2
    # A stale row within an otherwise valid chunk is rejected, even at 1 ULP.
    adapter._client.call("get_action")[1]["chunk_anchor_encoder"][0, 0, 3] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="anchor differs"):
        adapter.act(obs)
    with pytest.raises(ValueError, match="requires the request"):
        decode_chunk([row], "eef_delta")


def test_batch_anchors_match_each_environment_and_reject_reordering():
    first, second = smoke_observation(), smoke_observation()
    second["eef_encoder"][1, 2, 3] = 0.2
    chunks = [[eef_row(obs["eef_encoder"])] for obs in (first, second)]
    adapter = fake_adapter(chunks)
    assert len(adapter.act_batch([dict(first, env_idx=9), dict(second, env_idx=3)])) == 2
    chunks.reverse()
    with pytest.raises(ValueError, match="anchor differs"):
        adapter.act_batch([dict(first, env_idx=9), dict(second, env_idx=3)])


def test_close_and_reset_end_trials_once():
    adapter = fake_adapter([])
    adapter.reset(seed=1)
    adapter.reset(seed=2)
    adapter.close()
    adapter.close()
    names = [name for name, _ in adapter._client.calls]
    assert names == [
        "prepare_case",
        "reset",
        "trial_end",
        "prepare_case",
        "reset",
        "trial_end",
        "close",
    ]


def test_connect_settings_and_empty_url(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import tower_eval.policies.xpolicylab as bridge

    with pytest.raises(ValueError, match="URL must not be empty"):
        bridge.XPolicyLabAdapter("  ")
    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(bridge, "client_root", lambda _: tmp_path)
    monkeypatch.setattr(
        bridge.importlib, "import_module", lambda _: SimpleNamespace(WsModelClient=client)
    )
    bridge.XPolicyLabAdapter("ws://localhost").close()
    assert captured["max_connect_seconds"] >= 180
    assert captured["max_connect_attempts"] >= 180
    bridge.XPolicyLabAdapter(
        "ws://localhost", max_connect_attempts=3, connect_timeout_s=2, max_connect_seconds=4
    ).close()
    assert (
        captured["max_connect_attempts"],
        captured["connect_timeout_s"],
        captured["max_connect_seconds"],
    ) == (3, 2, 4)


@pytest.mark.parametrize("adapter", ["xpolicylab", "XPolicyLab", " XPOLICYLAB "])
@pytest.mark.parametrize("backend", ["cpu", "cpu-reference", "cpu-smoke", "reference"])
def test_cpu_config_rejects_xpl_before_connecting(backend, adapter):
    from tower_eval.config import EvaluationConfig

    with pytest.raises(ValueError, match="CPU backend provides neither"):
        EvaluationConfig(
            environment={"backend": backend},
            policy={"adapter": adapter, "server": "ws://localhost"},
        )


def test_encoder_chunks_fail_closed_before_execution():
    from tower_eval.envs import TowerEnv
    from tower_eval.policies.adapter import ActionChunk, normalize_actions
    from tower_eval.simulator import CpuReferenceBackend

    backend = CpuReferenceBackend()
    env = TowerEnv(backend)
    env.reset()
    for frame in ("encoder_arm_base", "unknown_frame"):
        chunk = ActionChunk(np.zeros((2, 16)), metadata={"coordinate_frame": frame})
        for value in (
            chunk,
            {
                "protocol": "tower-action-chunk-v1",
                "actions": chunk.actions,
                "dt_s": 0.1,
                "metadata": chunk.metadata,
            },
        ):
            with pytest.raises(ValueError, match="frame converter"):
                normalize_actions(value)
            with pytest.raises(ValueError, match="frame converter"):
                env.step(value)
        with pytest.raises(ValueError, match="frame converter"):
            normalize_actions([{"coordinate_frame": frame}])
    assert backend.read().time == 0
    physical = ActionChunk(np.zeros((1, 16)), metadata={"coordinate_frame": "physical"})
    env.step(normalize_actions(physical)[0])
    assert backend.read().time == 0.1
    env.close()


def test_parent_embodiment_matches_pinned_mirror():
    try:
        root = client_root()
    except RuntimeError:
        pytest.skip("XPolicyLab submodule unavailable")

    def files(directory):
        return {
            p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*")
            if p.is_file()
        }

    assert files(root.parent / "env_cfg") == files(root / "embodiments/nero_dual/env_cfg")
