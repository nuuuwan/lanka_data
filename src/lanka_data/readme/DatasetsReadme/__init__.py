from lanka_data.readme.DatasetsReadme.DatasetsReadmeConfigMixin import \
    DatasetsReadmeConfigMixin
from lanka_data.readme.DatasetsReadme.DatasetsReadmeDatasetsMixin import \
    DatasetsReadmeDatasetsMixin
from utils_future import File, Log

log = Log("DatasetsReadme")


class DatasetsReadme(
    DatasetsReadmeConfigMixin,
    DatasetsReadmeDatasetsMixin,
):
    PATH = "README.datasets.md"

    def get_lines(self):
        lines = [
            '# Datasets',
            '',
            'This document describes the available datasets and their'
            + ' structure.',
            '',
        ]

        for dataset_info in self.DATASETS:
            dataset_class = dataset_info['class']
            lines += self.get_lines_for_dataset(
                dataset_class,
                dataset_info['name'],
                dataset_info['description'],
            )

        return lines

    def build(self):
        lines = self.get_lines()
        readme_file = File(self.PATH)
        readme_file.write('\n'.join(lines))
        log.info(f'Wrote {readme_file}')