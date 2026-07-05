class ElectionDatasetRegistry:
    DATASET_CLASS = None

    @classmethod
    def set_dataset_class(cls, dataset_class):
        cls.DATASET_CLASS = dataset_class
