import os

from PIL import Image

from lanka_data.visual.animation_visual.AnimationMP4Mixin import \
    AnimationMP4Mixin
from utils_future import File, Log

log = Log("AnimationEncoder")


class AnimationEncoder(AnimationMP4Mixin):
    GIF_NAME = "Animation.gif"
    FRAME_DURATION_MS = 1200

    @staticmethod
    def _load_frames(frame_paths):
        frames = []
        size = None
        for path in frame_paths:
            frame = Image.open(path).convert("RGB")
            if size is None:
                size = frame.size
            elif frame.size != size:
                frame = frame.resize(size)
            frames.append(frame)
        return frames

    @classmethod
    def _write_gif(cls, frames, output_dir):
        gif_path = os.path.join(output_dir, cls.GIF_NAME)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=cls.FRAME_DURATION_MS,
            loop=0,
        )
        log.debug(f"Wrote {File(gif_path)}")
        return gif_path

    @classmethod
    def encode(cls, frame_paths, output_dir):
        frames = cls._load_frames(frame_paths)
        if not frames:
            raise ValueError("No frames to animate")
        gif_path = cls._write_gif(frames, output_dir)
        result = {"image_path": gif_path, "animation_path": gif_path}
        mp4_path = cls._try_encode_mp4(frames, output_dir)
        if mp4_path is not None:
            result["mp4_path"] = mp4_path
        return result
