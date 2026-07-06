import os

from lanka_data.visual.animation_visual.AnimationEncoder import \
    AnimationEncoder
from lanka_data.visual.plot.Plot import Plot
from lanka_data.visual.Visual import Visual
from utils_future import Log, timer

log = Log("AnimationVisual")


class AnimationVisual(Visual):
    def _build_frame(self, dataset):
        from lanka_data.visual.VisualFactory import VisualFactory

        year = getattr(dataset, "panel_label", None) or dataset.get_year()
        frame_command = self.command.copy(
            when_cmd=year, how_cmd=self.command.how.frame_how
        )
        frame_visual = VisualFactory.from_command_and_datasets(
            frame_command, [dataset]
        )
        return frame_visual.build()["image_path"]

    @timer
    def build(self):
        frame_paths = [
            self._build_frame(dataset) for dataset in self.datasets
        ]
        output_dir = os.path.join(Plot.DIR_OUTPUT, self.command.cmd_id)
        os.makedirs(output_dir, exist_ok=True)
        result = AnimationEncoder.encode(frame_paths, output_dir)
        result["frame_paths"] = frame_paths
        log.debug(f"Built animation with {len(frame_paths)} frames")
        return result
