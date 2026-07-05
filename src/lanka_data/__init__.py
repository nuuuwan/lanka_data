import sys
from importlib import import_module

_API_EXPORTS = """
CommandCache CommandError CommandLoaderMixin InvalidCommandError
InvalidWhenError InvalidWhereError UnknownHowError UnknownWhatError
DataSource Segregation SegregationComputeMixin Dataset DiffDataset
RegionValueDataset RegionValueDatasetTableMixin
""".split()
_DATASET_EXPORTS = """
BarChartDrawMixin BarChartLabelMixin BarChartVisual BumpChartDataMixin
BumpChartDrawMixin BumpChartVisual Census2001Dataset Census2012Dataset
Census2024Dataset ColorSpec ColorSpecCategoryMixin ColorSpecConstants
ColorSpecCustomMixin ColorSpecFactory ColorSpecHelpers ColorSpecHelpersMixin
Command CommandBase CommandHelp CommandRunner DatasetFactory Diversity
ElectionDataset ElectionSummaryDataset EmptyDataset Example ExampleOutputMixin
FieldNameUtils Font Footer GeoData GeoDataLoaderMixin GIG2Dataset Header
HowFormatter JSONVisual Label LabelFit Legend MapVisual PieChartGridMixin
PieChartVisual Plot PlotVisual ReadMe ReadMeExamplesItemMixin
ReadMeExamplesMixin ReadMeFooterMixin ReadMeSourcesMixin ReadMeUsageMixin
RegionFetchMixin RegionLoadersMixin RegionParentMixin RegionParserMixin
RegionParserRadiusMixin RegionRawDataMixin Regions RegionTypeUtils Text
Visual VisualFactory WhatFormatter Where WhereFormatter
""".split()


def _export_names(module_name, names):
    module = import_module(module_name)
    for name in names:
        globals()[name] = getattr(module, name)


def _alias_prefix(old_name, new_name):
    module = import_module(new_name)
    sys.modules[old_name] = module
    for name, child_module in list(sys.modules.items()):
        if name.startswith(new_name + "."):
            suffix = name.removeprefix(new_name)
            sys.modules[old_name + suffix] = child_module


_export_names("api", _API_EXPORTS)
_export_names("datasets", _DATASET_EXPORTS)

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
