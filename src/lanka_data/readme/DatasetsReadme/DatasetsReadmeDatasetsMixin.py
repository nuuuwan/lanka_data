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


class DatasetsReadmeDatasetsMixin:
    DATASET_CONSTRUCTORS = {
        Census2001Dataset: lambda: Census2001Dataset([], ''),
        Census2012Dataset: lambda: Census2012Dataset([], ''),
        Census2024Dataset: lambda: Census2024Dataset([], ''),
        ElectionDataset: lambda: ElectionDataset([], '', ''),
        ElectionSummaryDataset: lambda: ElectionSummaryDataset([], '', ''),
        RiversDataset: lambda: RiversDataset([], ''),
    }

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

        sources = \
            DatasetsReadmeDatasetsMixin.get_sources_for_dataset(dataset_class)
        for source in sources:
            source_line = f'- [{source.name}]({source.url})'
            lines.append(source_line)

        lines.append('')

        return lines

    @staticmethod
    def get_sources_for_dataset(dataset_class):
        constructor = DatasetsReadmeDatasetsMixin.DATASET_CONSTRUCTORS.get(
            dataset_class
        )
        if constructor is None:
            return []
        instance = constructor()
        return instance.get_sources()
