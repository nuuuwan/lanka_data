import sys
from importlib import import_module

from api.command.CommandCache import CommandCache
from api.command.CommandError import CommandError
from api.command.CommandLoaderMixin import CommandLoaderMixin
from api.command.InvalidCommandError import InvalidCommandError
from api.command.InvalidWhenError import InvalidWhenError
from api.command.InvalidWhereError import InvalidWhereError
from api.command.UnknownHowError import UnknownHowError
from api.command.UnknownWhatError import UnknownWhatError
from api.data import DataSource, Segregation, SegregationComputeMixin
from api.dataset import Dataset, DiffDataset, RegionValueDataset
from api.dataset import RegionValueDatasetTableMixin
from datasets import BarChartDrawMixin, BarChartLabelMixin, BarChartVisual
from datasets import BumpChartDataMixin, BumpChartDrawMixin, BumpChartVisual
from datasets import Census2001Dataset, Census2012Dataset, Census2024Dataset
from datasets import ColorSpec, ColorSpecCategoryMixin, ColorSpecConstants
from datasets import ColorSpecCustomMixin, ColorSpecFactory, ColorSpecHelpers
from datasets import ColorSpecHelpersMixin, Command, CommandBase, CommandHelp
from datasets import CommandRunner, DatasetFactory, Diversity, ElectionDataset
from datasets import ElectionSummaryDataset, EmptyDataset, Example
from datasets import ExampleOutputMixin, FieldNameUtils, Font, Footer, GeoData
from datasets import GeoDataLoaderMixin, GIG2Dataset, Header, HowFormatter
from datasets import JSONVisual, Label, LabelFit, Legend, MapVisual
from datasets import PieChartGridMixin, PieChartVisual, Plot, PlotVisual
from datasets import ReadMe, ReadMeExamplesItemMixin, ReadMeExamplesMixin
from datasets import ReadMeFooterMixin, ReadMeSourcesMixin, ReadMeUsageMixin
from datasets import RegionFetchMixin, RegionLoadersMixin, RegionParentMixin
from datasets import RegionParserMixin, RegionParserRadiusMixin
from datasets import RegionRawDataMixin, Regions, RegionTypeUtils, Text
from datasets import Visual, VisualFactory, WhatFormatter, Where
from datasets import WhereFormatter


def _alias_prefix(old_name, new_name):
    module = import_module(new_name)
    sys.modules[old_name] = module
    for name, child_module in list(sys.modules.items()):
        if name.startswith(new_name + "."):
            suffix = name.removeprefix(new_name)
            sys.modules[old_name + suffix] = child_module


for _old_child in [
    "command",
    "data",
    "dataset",
    "examples",
    "readme",
    "region",
    "visual",
]:
    _alias_prefix(f"lanka_data.{_old_child}", f"datasets.{_old_child}")

for _old_name, _new_name in {
    "lanka_data.command.CommandCache": "api.command.CommandCache",
    "lanka_data.command.CommandError": "api.command.CommandError",
    "lanka_data.command.CommandLoaderMixin": "api.command.CommandLoaderMixin",
    "lanka_data.command.InvalidCommandError": "api.command.InvalidCommandError",
    "lanka_data.command.InvalidWhenError": "api.command.InvalidWhenError",
    "lanka_data.command.InvalidWhereError": "api.command.InvalidWhereError",
    "lanka_data.command.UnknownHowError": "api.command.UnknownHowError",
    "lanka_data.command.UnknownWhatError": "api.command.UnknownWhatError",
    "lanka_data.data.DataSource": "api.data.DataSource",
    "lanka_data.data.Segregation": "api.data.Segregation",
    "lanka_data.dataset.Dataset": "api.dataset.Dataset",
    "lanka_data.dataset.DiffDataset": "api.dataset.DiffDataset",
    "lanka_data.dataset.RegionValueDataset": (
        "api.dataset.RegionValueDataset"
    ),
}.items():
    sys.modules[_old_name] = import_module(_new_name)
