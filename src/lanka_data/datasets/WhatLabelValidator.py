from collections import Counter

from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import (
    ElectionSummaryDataset,
)
from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset
from lanka_data.datasets.what_label.WhatLabel import WhatLabel

from utils_future import Log

log = Log('WhatLabelValidator')


class WhatLabelValidator:
    def __init__(self):
        self.what_labels = WhatLabel.list()
        self.datasets = [
            Census2001Dataset,
            Census2012Dataset,
            Census2024Dataset,
            ElectionDataset,
            ElectionSummaryDataset,
            RiversDataset,
        ]
        self.errors = []
        self.warnings = []

    def _get_dataset_to_labels(self):
        idx = {}
        for dataset_cls in self.datasets:
            cls_name = dataset_cls.__name__
            idx[cls_name] = dataset_cls.get_labels()
        return idx

    def _check_unique_labels(self):
        all_labels = [w.label for w in self.what_labels]
        label_counts = Counter(all_labels)
        duplicates = [
            label for label, count in label_counts.items() if count > 1
        ]
        if duplicates:
            self.errors.append(
                f"Found duplicate WhatLabels: {sorted(duplicates)}"
            )

    def _check_every_label_has_dataset(self):
        unique_what_labels = set([w.label for w in self.what_labels])
        dataset_to_labels = self._get_dataset_to_labels()

        for i_what_label, what_label in enumerate(unique_what_labels, start=1):
            if 'Summary' in what_label:
                what_label = what_label.replace("Summary", "")
            matching_datasets = set()
            for dataset, labels in dataset_to_labels.items():
                if what_label in labels:
                    matching_datasets.add(dataset)
            
            if matching_datasets:
                log.debug(f"✅ {i_what_label}. '{what_label}' in {', '.join(sorted(matching_datasets))}")
            else:
                self.errors.append(f"WhatLabel '{what_label}' has no dataset")


    def _check_every_dataset_label_is_valid(self):
        dataset_to_labels = self._get_dataset_to_labels()
        valid_labels = set([w.label for w in self.what_labels])

        i_label = 0
        for dataset, labels in dataset_to_labels.items():
            for label in labels:
                i_label += 1
                if label not in valid_labels:
                    self.errors.append(f"Dataset '{dataset}' has invalid label '{label}'")
                else:
                    log.debug(f"✅ {i_label}. {dataset}.{label} is valid.")
                


    def validate(self):
        self._check_unique_labels()
        self._check_every_label_has_dataset()
        self._check_every_dataset_label_is_valid()
        log.debug('-' * 32)
        for error in self.errors:
            log.error(error)
        for warning in self.warnings:
            log.warning(warning)

