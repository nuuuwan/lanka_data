from functools import cached_property

from sklearn.linear_model import LinearRegression
from utils import Log

from lanka_data.core.Dataset import Dataset

log = Log(__name__)


class Model:
    def __init__(
        self,
        datasets: list[Dataset],
        i_y: int,
        n_window: int,
    ):
        self.datasets = datasets
        self.i_y = i_y
        self.n_window = n_window

    @cached_property
    def t_list(self) -> list[str]:
        t_set = None
        for dataset in self.datasets:
            dataset_t_set = set(dataset.data.keys())
            if t_set:
                t_set = t_set.intersection(dataset_t_set)
            else:
                t_set = dataset_t_set
        return sorted(list(t_set))

    @cached_property
    def n(self) -> int:
        return len(self.t_list) - self.n_window

    @cached_property
    def x_list_list(self) -> list[list[float]]:
        t_list = self.t_list
        x_list_list = []
        for i in range(self.n + 1):
            x_list = []
            for dataset in self.datasets:
                for j in range(self.n_window):
                    x_list.append(dataset.data[t_list[i + j]])
            x_list_list.append(x_list)
        return x_list_list

    @cached_property
    def y_list(self) -> list[float]:
        t_list = self.t_list
        y_list = []
        dataset_y = self.datasets[self.i_y]
        for i in range(self.n):
            y_list.append(dataset_y.data[t_list[i + self.n_window]])
        return y_list

    @cached_property
    def linear_regression(self):
        x_list_list = self.x_list_list
        y_list = self.y_list

        # train model
        lr = LinearRegression()
        lr.fit(x_list_list[:-1], y_list)
        w, w0 = lr.coef_, lr.intercept_

        # test model
        t_list = self.t_list
        for i, [t, x_list, y_actual] in enumerate(
            zip(t_list[1:], x_list_list[:-1], y_list)
        ):
            y_pred = w0 + sum([x * w for x, w in zip(x_list, w)])
            log.debug(f'{t}) {y_pred:,.0f} ({y_actual:,.0f})')

        # evalute next
        last_x_list = x_list_list[-1]
        next_y_pred = w0 + sum([x * w for x, w in zip(last_x_list, w)])
        log.info(f'next) {next_y_pred:,.0f}')

        return w, w0
