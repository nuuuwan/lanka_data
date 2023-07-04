import unittest

from lanka_data import Dataset, Model


def get_model():
    dataset = Dataset.find('Tourist Arrivals Monthly')[0]
    return Model(
        datasets=[dataset],
        i_y=0,
        n_window=36,
    )


class TestCase(unittest.TestCase):
    def test_init(self):
        get_model()

    def t_list(self):
        t_list = get_model().t_list
        self.assertEqual(
            t_list[-3:],
            ['2023-03-01', '2023-04-01', '2023-05-01'],
        )

    def test_n(self):
        model = get_model()
        self.assertEqual(
            model.n,
            617 - 36,
        )

    def test_x_list_list(self):
        model = get_model()
        x_list_list = model.x_list_list
        self.assertEqual(len(x_list_list), 582)
        self.assertEqual(len(x_list_list[0]), 36)
        self.assertEqual(
            x_list_list[-1][-1],
            83309,
        )

    def test_y_list(self):
        model = get_model()
        y_list = model.y_list
        self.assertAlmostEqual(
            len(y_list),
            581,
        )
        self.assertEqual(
            y_list[-1],
            83309,
        )

    def test_linear_regression(self):
        model = get_model()
        model.linear_regression
