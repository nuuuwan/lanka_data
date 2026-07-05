class CensusDatasetRegistry:
    DATASET_CLASSES = []

    @classmethod
    def set_dataset_classes(cls, dataset_classes):
        cls.DATASET_CLASSES = list(dataset_classes)
