from lanka_data.datasets.dataset.custom.Census2001Dataset import \
    Census2001Dataset
from lanka_data.datasets.dataset.custom.Census2012Dataset import \
    Census2012Dataset
from lanka_data.datasets.dataset.custom.Census2024Dataset import \
    Census2024Dataset
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import \
    ElectionSummaryDataset
from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset
from utils_future import File, Log

log = Log("DatasetsReadme")


class DatasetsReadme:
    PATH = "README.datasets.md"

    DATASETS = [
        {
            'class': Census2001Dataset,
            'name': 'Census of Population and Housing 2001',
            'description': 'Demographics data from the 2001 census',
        },
        {
            'class': Census2012Dataset,
            'name': 'Census of Population and Housing 2012',
            'description': 'Demographics data from the 2012 census',
        },
        {
            'class': Census2024Dataset,
            'name': 'Census of Population and Housing 2024',
            'description': 'Demographics data from the 2024 census',
        },
        {
            'class': ElectionDataset,
            'name': 'Election Data',
            'description':
                'Detailed election results by candidate/party and region',
        },
        {
            'class': ElectionSummaryDataset,
            'name': 'Election Summary Data',
            'description':
                'Election summary statistics (Valid, Rejected, Did Not Vote)',
        },
        {
            'class': RiversDataset,
            'name': 'Rivers Data',
            'description': 'River lengths and catchment areas by region',
        },
    ]

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

    @staticmethod
    def get_lines_for_dataset(dataset_class, name, description):
        lines = [
            f'## {name}',
            '',
            description,
            '',
        ]

        lines.append('### "What" Fields')
        lines.append('')

        labels = dataset_class.get_labels()
        if labels:
            for label in sorted(labels):
                lines.append(f'- `{label}`')
        else:
            lines.append('(No labels available)')

        lines.append('')

        lines.append('### Source Data')
        lines.append('')

        sources = DatasetsReadme.get_sources_for_dataset(dataset_class)
        for source in sources:
            source_line = f'- [{source.name}]({source.url})'
            lines.append(source_line)

        lines.append('')

        return lines

    @staticmethod
    def get_sources_for_dataset(dataset_class):
        if dataset_class == ElectionSummaryDataset:
            instance = ElectionSummaryDataset([], '', '')
        elif dataset_class == ElectionDataset:
            instance = ElectionDataset([], '', '')
        elif dataset_class == RiversDataset:
            instance = RiversDataset([], '')
        else:
            instance = dataset_class([], '')

        return instance.get_sources()

    def build(self):
        lines = self.get_lines()
        readme_file = File(self.PATH)
        readme_file.write('\n'.join(lines))
        log.info(f'Wrote {readme_file}')

