import random

from utils import Log

from lanka_data import Dataset

log = Log(__name__)

MIN_N = 60


def get_common_t_list(dataset1, dataset2):
    t_set1 = set(dataset1.data.keys())
    t_set2 = set(dataset2.data.keys())
    return list(sorted(t_set1.intersection(t_set2)))


def get_pearson_correlation_coefficient(dataset1, dataset2):
    t_list = get_common_t_list(dataset1, dataset2)
    v1 = [dataset1.data[t] for t in t_list]
    v2 = [dataset2.data[t] for t in t_list]
    n = len(t_list)

    if n == 0:
        return 0, n
    sum1 = sum(v1)
    sum2 = sum(v2)
    sum1_sq = sum([x * x for x in v1])
    sum2_sq = sum([x * x for x in v2])
    p_sum = sum([x * y for x, y in zip(v1, v2)])

    num = p_sum - (sum1 * sum2 / n)
    den = ((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n)) ** 0.5
    if den == 0:
        return 0, n
    return num / den, n


def get_highly_correlated():
    dataset_list = Dataset.load_list()
    dataset_list = [
        d for d in dataset_list if d.summary_statistics['n'] >= MIN_N
    ]
    random.shuffle(dataset_list)
    n_datasets = len(dataset_list)
    log.info(f'{n_datasets=:,}')

    while True:
        i1 = random.randint(0, n_datasets - 2)
        i2 = random.randint(i1 + 1, n_datasets - 1)

        dataset1 = dataset_list[i1]
        dataset2 = dataset_list[i2]
        pcc, n = get_pearson_correlation_coefficient(dataset1, dataset2)
        if n >= MIN_N and isinstance(pcc, float) and pcc < -0.98:
            log.info(f'{pcc:.1%} {dataset1} {dataset2}')


if __name__ == '__main__':
    get_highly_correlated()
