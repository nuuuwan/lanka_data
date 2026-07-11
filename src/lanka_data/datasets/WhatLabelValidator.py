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

    def _get_all_dataset_labels(self):
        dataset_labels = set()
        for dataset_cls in self.datasets:
            dataset_labels.update(dataset_cls.get_labels())
        return dataset_labels

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
        dataset_labels = self._get_all_dataset_labels()

        for what_label in unique_what_labels:
            if what_label.endswith("Summary"):
                base_label = what_label.replace("Summary", "")
                if base_label not in dataset_labels:
                    self.errors.append(
                        f"WhatLabel '{what_label}' has no dataset"
                    )
            elif what_label not in dataset_labels:
                self.errors.append(f"WhatLabel '{what_label}' has no dataset")

    def validate(self):
        self._check_unique_labels()
        self._check_every_label_has_dataset()
        return len(self.errors) == 0

    def get_report(self):
        lines = []
        lines.append("WhatLabelValidator Report")
        lines.append("=" * 40)

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")
        else:
            lines.append("\nNo errors found!")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)
