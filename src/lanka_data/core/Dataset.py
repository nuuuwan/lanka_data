from dataclasses import dataclass
from functools import cached_property

from utils import WWW, hashx

from lanka_data.core.common import URL_DATA_REPO


def normalize_t(t: str) -> str:
    t = str(t)
    if len(t) == 4:
        return t + '-01-01'
    if len(t) == 7:
        return t + '-01'
    return t


def normalize(data: dict) -> dict:
    return dict([(normalize_t(t), v) for t, v in data.items()])


@dataclass
class Dataset:
    source_id: str
    category: str
    sub_category: str
    scale: str
    unit: str
    frequency_name: str
    i_subject: str
    footnotes: dict
    summary_statistics: dict

    @property
    def id(self) -> str:
        return f'{self.source_id}.{self.sub_category}.{self.frequency_name}'

    @property
    def short_name(self) -> str:
        return self.sub_category.split()[0] + '-' + hashx.md5(self.id)[:4]

    @property
    def url_detailed_data(self) -> str:
        return f'{URL_DATA_REPO}/sources/{self.source_id}/{self.id}.json'

    @cached_property
    def detailed_data(self) -> dict:
        print(self.url_detailed_data)
        return WWW(self.url_detailed_data).readJSON()

    @cached_property
    def cleaned_data(self) -> dict:
        return normalize(self.detailed_data['cleaned_data'])

    @cached_property
    def raw_data(self) -> dict:
        return self.detailed_data['raw_data']

    @cached_property
    def data(self) -> dict:
        return self.cleaned_data

    @cached_property
    def xy(self):
        x = list(self.data.keys())
        y = list(self.data.values())
        return x, y

    @staticmethod
    def load_from_dict(d):
        return Dataset(
            source_id=d['source_id'],
            category=d['category'],
            sub_category=d['sub_category'],
            scale=d['scale'],
            unit=d['unit'],
            frequency_name=d['frequency_name'],
            i_subject=d['i_subject'],
            footnotes=d['footnotes'],
            summary_statistics=d['summary_statistics'],
        )

    @staticmethod
    def load_list() -> list:
        url_summary = URL_DATA_REPO + '/summary.json'
        raw_data_list = WWW(url_summary).readJSON()
        unsorted_list = [Dataset.load_from_dict(d) for d in raw_data_list]
        sorted_list = sorted(
            unsorted_list,
            key=lambda dataset: dataset.id,
        )
        return sorted_list

    def isMatch(self, keywords: str) -> bool:
        keyword_list = keywords.lower().split()
        for keyword in keyword_list:
            keyword_is_in_something = False
            for haystack in [
                self.source_id,
                self.category,
                self.sub_category,
            ]:
                if keyword in haystack.lower():
                    keyword_is_in_something = True
                    break
            if not keyword_is_in_something:
                return False
        return True

    @staticmethod
    def find(keywords: str, limit=10) -> list:
        dataset_list = Dataset.load_list()
        dataset_list = [
            dataset for dataset in dataset_list if dataset.isMatch(keywords)
        ]
        return dataset_list[:limit]
