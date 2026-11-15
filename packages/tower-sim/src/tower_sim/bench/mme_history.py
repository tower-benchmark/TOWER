"""Lossless, per-episode complete-prefix history from validated exact10Hz frames."""

import base64
import hashlib
import io

import numpy as np

CAMERAS = ("right_front_camera", "left_back_camera", "left_wrist_camera", "back_wrist_camera")
SCHEMA = "mme-actual-10hz-prefix-framesamp-v1"


def resize_pad(image):
    from PIL import Image

    im = Image.fromarray(image)
    ratio = max(im.width / 224, im.height / 224)
    im = im.resize((int(im.width / ratio), int(im.height / ratio)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(im, ((224 - im.width) // 2, (224 - im.height) // 2))
    return np.asarray(canvas)


def indices(frame):
    return (
        np.arange(frame + 1, dtype=np.int32)
        if frame < 32
        else np.linspace(0, frame, 32, dtype=np.int32)
    )


def png(image):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


class PrefixHistory:
    def __init__(self, episode_id, max_frames):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Episode identity required")
        self.episode_id = episode_id
        self.max_frames = int(max_frames)
        self.frames = []
        self.closed = False

    def append(self, tick, pixels, *, validated):
        if self.closed or validated is not True:
            raise ValueError("Rejected/closed history update")
        if not isinstance(tick, int) or tick < 0 or tick % 40:
            raise ValueError("History requires the exact10Hz grid")
        if tick // 40 != len(self.frames):
            raise ValueError("Skipped or duplicate history frame")
        if len(self.frames) >= self.max_frames:
            raise ValueError("History exceeds positional embedding range")
        pixels = np.asarray(pixels)
        if pixels.shape != (4, 240, 320, 3) or pixels.dtype != np.uint8:
            raise ValueError("Wrong validated camera shape")
        ego = resize_pad(np.concatenate((pixels[0], pixels[1]), axis=1))
        raw = png(ego)
        self.frames.append((base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()))

    def request(self, tick):
        if (
            self.closed
            or not isinstance(tick, int)
            or tick % 40
            or tick // 40 != len(self.frames) - 1
        ):
            raise ValueError("Current/history timestamps differ")
        selected = indices(tick // 40)
        frames = [self.frames[int(i)] for i in selected]
        return {
            "schema": SCHEMA,
            "episode_id": self.episode_id,
            "current_frame": tick // 40,
            "time_s": tick / 400,
            "prefix_frames": len(self.frames),
            "indices": selected.tolist(),
            "ego_png": [f[0] for f in frames],
            "png_sha256": [f[1] for f in frames],
        }

    def close(self):
        self.frames.clear()
        self.closed = True


class EpisodeHistory:
    def __init__(self, episode_id, camera_names, max_frames):
        self.order = [list(camera_names).index(name) for name in CAMERAS]
        self.prefix = PrefixHistory(episode_id, max_frames)
        self.appended = 0
        self.bytes = 0

    @property
    def episode_id(self):
        return self.prefix.episode_id

    def append_validated(self, archive, tick, *, rejected, terminal):
        if rejected or terminal or tick % 40:
            return False
        if round(float(archive.times[-1]) * 400) != tick or archive._latest_pixels is None:
            raise ValueError("History requires actual validated archive at this tick")
        self.prefix.append(tick, archive._latest_pixels[self.order], validated=True)
        self.appended += 1
        self.bytes += len(self.prefix.frames[-1][0]) + 64
        return True

    def request(self, tick):
        return self.prefix.request(tick)

    def close(self):
        self.prefix.close()

    def receipt(self):
        return {
            "episode_id": self.episode_id,
            "frames_collected": self.appended,
            "lossless_history_bytes": self.bytes,
            "closed": self.prefix.closed,
            "extra_render_or_physics_calls": 0,
        }
