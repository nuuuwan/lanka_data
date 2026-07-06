import os

from utils_future import Log

log = Log("AnimationMP4Mixin")


class AnimationMP4Mixin:
    MP4_NAME = "Animation.mp4"
    MP4_FPS = 1
    MP4_DPI = 100

    @classmethod
    def _try_encode_mp4(cls, frames, output_dir):
        writer_cls = cls._get_ffmpeg_writer()
        if writer_cls is None:
            return None
        return cls._write_mp4(frames, output_dir, writer_cls)

    @staticmethod
    def _get_ffmpeg_writer():
        try:
            from matplotlib.animation import FFMpegWriter
        except ImportError:
            return None
        if not FFMpegWriter.isAvailable():
            return None
        return FFMpegWriter

    @classmethod
    def _write_mp4(cls, frames, output_dir, writer_cls):
        import matplotlib.pyplot as plt

        mp4_path = os.path.join(output_dir, cls.MP4_NAME)
        width, height = frames[0].size
        fig = plt.figure(
            figsize=(width / cls.MP4_DPI, height / cls.MP4_DPI),
            dpi=cls.MP4_DPI,
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        writer = writer_cls(fps=cls.MP4_FPS)
        with writer.saving(fig, mp4_path, dpi=cls.MP4_DPI):
            for frame in frames:
                ax.imshow(frame)
                writer.grab_frame()
        plt.close(fig)
        return mp4_path
