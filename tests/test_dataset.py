import unittest

from lanka_data import Dataset


class TestCase(unittest.TestCase):
    def test_load_from_dict(self):
        dataset = Dataset(
            source_id='source_id',
            category='category',
            sub_category='sub_category',
            scale='scale',
            unit='unit',
            frequency_name='frequency_name',
            i_subject='i_subject',
            footnotes='footnotes',
            summary_statistics='summary_statistics',
        )
        self.assertEqual(dataset.source_id, 'source_id')

    def test_load(self):
        dataset_list = Dataset.load_list()
        self.assertGreaterEqual(len(dataset_list), 1_000)

    def test_find(self):
        dataset_list = Dataset.find('GDP', limit=5)
        self.assertEqual(len(dataset_list), 5)

        first_dataset = dataset_list[0]
        self.assertEqual(first_dataset.source_id, 'adb')
        self.assertTrue(
            first_dataset.sub_category.startswith(
                'BALANCE International investment position'
            )
        )
        self.assertEqual(first_dataset.data['2000'], 33.03425)
        x, y = first_dataset.xy
        self.assertEqual(x[0], '2000')
        self.assertEqual(y[0], 33.03425)
