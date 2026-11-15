# SPDX-License-Identifier: Apache-2.0
"""Strict input schemas retain the existing sparse report representation."""

from dataclasses import is_dataclass
from pathlib import Path

import pytest
from tower_eval.config import load_config
from tower_eval.controllers import RobotSpec


@pytest.mark.parametrize(
    "table,key,value",
    [
        ("environment", "max_steps", "10"),
        ("environment", "render_mode", '"rgb_array"'),
        ("environment", "camera", '"front"'),
        ("environment", "stage", '"scene.usd"'),
        ("controller", "name", '"direct"'),
        ("controller", "prefix", '"left"'),
        ("controller", "interpolation_fraction", "0.8"),
        ("controller", "gripper_step_fraction", "0.2"),
        ("policy.config", "unused_option", "1"),
        ("environment", "typo", "1"),
        ("policy", "typo", "1"),
    ],
)
def test_unused_or_unknown_keys_fail_before_execution(tmp_path, table, key, value):
    path = tmp_path / "config.toml"
    path.write_text(f"[{table}]\n{key} = {value}\n")
    with pytest.raises(ValueError, match=key):
        load_config(path)
    path.write_text("")
    with pytest.raises(ValueError, match=key):
        load_config(path, [f"{table}.{key}={value}"])


def test_empty_controller_is_also_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[controller]\n")
    with pytest.raises(ValueError, match=r"unused \[controller\] table"):
        load_config(path)


def test_typed_defaults_do_not_change_sparse_reports(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'output = "run"\n[policy]\nurl = "http://localhost"\n[policy.config]\npredict_timeout_s = 9\n'
    )
    config = load_config(path)
    assert is_dataclass(config.environment) and is_dataclass(config.policy)
    assert is_dataclass(config.policy["config"])
    assert config.policy.get("server", config.policy.get("url")) == "http://localhost"
    assert config.to_dict() == {
        "schema": "tower-eval-v1",
        "seed": 0,
        "episodes": 10,
        "duration_seconds": 600.0,
        "output": tmp_path / "run",
        "environment": {},
        "policy": {"url": "http://localhost", "config": {"predict_timeout_s": 9}},
        "controller": {},
    }


def test_xpolicylab_options_remain_typed_and_readable(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("""[environment]
backend = "isaac"
[policy]
adapter = "xpolicylab"
url = "ws://localhost:8000"
[policy.config]
action_mode = "eef_delta"
client_root = "/some/client"
max_connect_attempts = 3
connect_timeout_s = 2
max_connect_seconds = 5
health_timeout_s = 4
predict_timeout_s = 9
""")
    policy = load_config(path).policy
    assert policy["config"].to_dict() == {
        "action_mode": "eef_delta",
        "client_root": "/some/client",
        "max_connect_attempts": 3,
        "connect_timeout_s": 2,
        "max_connect_seconds": 5,
        "health_timeout_s": 4,
        "predict_timeout_s": 9,
    }


def test_shipped_run_configs_load():
    for path in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml"):
        load_config(path)


def test_robot_unknown_fields_fail(fake_sim):
    config = dict(
        names=fake_sim.robot.names,
        joint_indices=fake_sim.robot.joint_indices,
        grippers=[
            dict(index=g.index, lower=g.lower, upper=g.upper) for g in fake_sim.robot.grippers
        ],
        lower=fake_sim.robot.lower,
        upper=fake_sim.robot.upper,
    )
    RobotSpec.from_dict(config)
    with pytest.raises(ValueError, match="unknown robot fields.*unused"):
        RobotSpec.from_dict({**config, "unused": True})


def test_serialized_key_order_is_independent_of_hash_seed(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "evaluation.toml"
    path.write_text(
        '[environment]\ntask = "tower_transfer_2"\nbackend = "cpu"\n'
        '[policy]\nadapter = "reference"\n'
    )
    script = (
        "import json, sys; from tower_eval.config import load_config; "
        "print(json.dumps(load_config(sys.argv[1]).environment.to_dict()))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outputs) == 1
    assert list(__import__("json").loads(outputs.pop())) == ["task", "backend"]
