# SPDX-License-Identifier: Apache-2.0
"""Incrementally durable diagnostic records, separate from task success."""

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import av
    import h5py


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root, output, *, allowed_paths=None, include_untracked=False):
    root = Path(root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    if (root / ".git").exists():
        tracked = git("ls-files", "-z").decode().split("\0")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        files = tracked + (untracked if include_untracked else [])
        if allowed_paths is None:
            # Provenance is intentionally limited to public source/configuration files.
            allowed_suffixes = {".py", ".toml", ".json", ".md", ".lock", ".yml", ".yaml"}
            files = [
                p
                for p in files
                if (
                    p.startswith(("src/", "docs/", "tools/"))
                    or p in {"README.md", "pyproject.toml", "uv.lock", "LICENSE", "NOTICE"}
                )
                and (Path(p).suffix in allowed_suffixes or p in {"LICENSE", "NOTICE"})
                and not any(
                    token in Path(p).name.lower()
                    for token in ("secret", "credential", "token", ".env")
                )
            ]
        else:
            allow = {str(Path(p)) for p in allowed_paths}
            files = [p for p in files if p in allow]
        files = sorted(set(files))
        patch = git("diff", "--binary", "HEAD") if include_untracked else b""
        head = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain").decode()
        source_format = "git"
    else:
        # Reference archives intentionally omit .git. Hash the extracted
        # source itself so an archive snapshot remains usable and auditable.
        ignored = {".venv", ".venv-curobo", ".git", "__pycache__", ".pytest_cache", ".cache"}
        files = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
        ]
        if allowed_paths is not None:
            allow = {str(Path(p)) for p in allowed_paths}
            files = [p for p in files if p in allow]
        patch = b""
        head = None
        status = "archive snapshot (no .git metadata)"
        source_format = "archive"
    hashes = {p: sha256(root / p) for p in sorted(files) if p and (root / p).is_file()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source.patch").write_bytes(patch)
    return {
        "provenance_schema": "tower-provenance-v1",
        "root": root.name,
        "source_format": source_format,
        "head": head,
        "status": status,
        "file_sha256": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


class Recorder:
    def __init__(
        self,
        root,
        record_video=True,
        *,
        overwrite=False,
        video_fps=10,
        video_encoder="libx264",
        video_layout="all",
        encoder_gpu=0,
    ):
        if (
            isinstance(video_fps, bool)
            or not isinstance(video_fps, int)
            or not 1 <= video_fps <= 30
        ):
            raise ValueError("video_fps must be an integer from 1 to 30")
        if video_encoder not in ("libx264", "h264_nvenc") or video_layout not in ("all", "mosaic"):
            raise ValueError("Invalid video encoder or layout")
        self.video_encoder, self.video_layout, self.encoder_gpu = (
            video_encoder,
            video_layout,
            encoder_gpu,
        )
        self.video_fps = int(video_fps)
        import h5py

        self.root = Path(root)
        if (self.root / "states.hdf5").exists() and not overwrite:
            raise FileExistsError(f"recorder output already exists: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "requests").mkdir(parents=True, exist_ok=True)
        self.events = (self.root / "events.jsonl").open("w", buffering=1)
        self.states = h5py.File(self.root / "states.hdf5", "w")
        self.record_video = record_video
        self.videos = {}
        self.frame_count = 0

    def event(self, kind, **values):
        self.events.write(json.dumps({"event": kind, **values}, default=json_value) + "\n")

    def state(self, state):
        values = {"time": np.asarray(state.time), "qpos": state.qpos, "qvel": state.qvel}
        for key in ("effort", "blocks", "block_velocities", "eef_world"):
            value = getattr(state, key, None)
            if value is not None:
                values[key] = value
        for key, value in values.items():
            value = np.asarray(value)
            if key not in self.states:
                self.states.create_dataset(
                    key, (0, *value.shape), maxshape=(None, *value.shape), dtype=value.dtype
                )
            ds = cast("h5py.Dataset", self.states[key])
            ds.resize(len(ds) + 1, axis=0)
            ds[-1] = value
        self.states.flush()

    def snapshot(self, images, label):
        import cv2

        for name, rgb in images.items():
            if not cv2.imwrite(
                str(self.root / f"{label}_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise OSError("Could not write camera image")

    def frame(self, images, sim_time):
        if not self.record_video:
            self.snapshot(images, "initial")
            return
        import av
        import cv2

        views = list(images.values())
        if len(views) == 4:
            images = dict(
                images,
                four_camera_rollout=np.concatenate(
                    [np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)], axis=0
                ),
            )
            if self.video_layout == "mosaic":
                images = {"four_camera_rollout": images["four_camera_rollout"]}
        for name, rgb in images.items():
            if self.frame_count == 0 and not cv2.imwrite(
                str(self.root / f"initial_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise OSError("Could not write initial camera image")
            if name not in self.videos:
                container = av.open(str(self.root / f"{name}.mp4"), mode="w")
                stream = cast(
                    "av.VideoStream", container.add_stream(self.video_encoder, rate=self.video_fps)
                )
                stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = (
                    {"crf": "18", "preset": "fast"}
                    if self.video_encoder == "libx264"
                    else {"cq": "18", "preset": "p4", "rc": "vbr", "gpu": str(self.encoder_gpu)}
                )
                self.videos[name] = container, stream
            container, stream = self.videos[name]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = self.frame_count
            for packet in stream.encode(frame):
                container.mux(packet)
        self.event("video_frame", frame=self.frame_count, sim_time=sim_time)
        self.frame_count += 1

    def request(self, index, obs, response=None) -> None:
        if isinstance(obs, dict) and "state" in obs and "images" in obs and "prompt" in obs:
            values = {"state": obs["state"], **obs["images"], "prompt": np.asarray(obs["prompt"])}
            if response is not None and "actions" in response:
                values["predicted_actions"] = np.asarray(response["actions"])
            np.savez_compressed(self.root / "requests" / f"{index:06d}.npz", **values)
        else:
            with (self.root / "observations.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"index": index, "observation": obs, "response": response},
                        default=json_value,
                    )
                    + "\n"
                )

    def request_response(self, index, obs, response) -> None:
        """Keep the durable pre-inference snapshot without recompressing its images."""
        path = self.root / "requests" / f"{index:06d}.npz"
        if not path.exists():
            return self.request(index, obs, response)
        if "actions" not in response:
            return None
        temporary = path.with_suffix(".npz.tmp")
        shutil.copyfile(path, temporary)
        with zipfile.ZipFile(temporary, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            if "predicted_actions.npy" in archive.namelist():
                raise ValueError(f"Response already recorded for request {index}")
            with archive.open("predicted_actions.npy", "w") as stream:
                np.lib.format.write_array(
                    stream, np.asarray(response["actions"]), allow_pickle=False
                )
        temporary.replace(path)
        return None

    def close(self):
        errors = []
        for container, stream in self.videos.values():
            try:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            except Exception as exc:  # noqa: BLE001 - Preserve the existing failure/reporting boundary and exception semantics.
                errors.append(str(exc))
        self.states.close()
        self.events.close()
        if errors:
            raise OSError("Video finalization failed: " + "; ".join(errors))
