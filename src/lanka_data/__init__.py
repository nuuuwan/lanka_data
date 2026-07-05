import sys
from importlib import import_module

_API_EXPORTS = """
CommandCache CommandError CommandLoaderMixin InvalidCommandError
InvalidWhenError InvalidWhereError UnknownHowError UnknownWhatError
DataSource Segregation SegregationComputeMixin Dataset DiffDataset
RegionValueDataset RegionValueDatasetTableMixin
""".split()
_DOMAIN_EXPORTS = """
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


_export_names("lanka_data.api", _API_EXPORTS)
_export_names("lanka_data.datasets", _DOMAIN_EXPORTS)

for _old_child in [
    "command",
    "data",
    "dataset",
    "region",
]:
    _alias_prefix(
        f"lanka_data.{_old_child}",
        f"lanka_data.datasets.{_old_child}",
    )

for _old_child in [
    "examples",
    "readme",
]:
    _alias_prefix(
        f"lanka_data.datasets.{_old_child}",
        f"lanka_data.{_old_child}",
    )

_alias_prefix("lanka_data.datasets.visual", "lanka_data.visual")

for _old_name, _new_name in {
    "lanka_data.command.CommandCache": "lanka_data.api.command.CommandCache",
    "lanka_data.command.CommandError": "lanka_data.api.command.CommandError",
    "lanka_data.command.CommandLoaderMixin": (
        "lanka_data.api.command.CommandLoaderMixin"
    ),
    "lanka_data.command.InvalidCommandError": (
        "lanka_data.api.command.InvalidCommandError"
    ),
    "lanka_data.command.InvalidWhenError": (
        "lanka_data.api.command.InvalidWhenError"
    ),
    "lanka_data.command.InvalidWhereError": (
        "lanka_data.api.command.InvalidWhereError"
    ),
    "lanka_data.command.UnknownHowError": (
        "lanka_data.api.command.UnknownHowError"
    ),
    "lanka_data.command.UnknownWhatError": (
        "lanka_data.api.command.UnknownWhatError"
    ),
    "lanka_data.command.fields": "lanka_data.api.command.fields",
    "lanka_data.command.fields.How": "lanka_data.api.command.fields.How",
    "lanka_data.command.fields.What": "lanka_data.api.command.fields.What",
    "lanka_data.command.fields.When": "lanka_data.api.command.fields.When",
    "lanka_data.command.fields.Where": "lanka_data.api.command.fields.Where",
    "lanka_data.data.DataSource": "lanka_data.api.data.DataSource",
    "lanka_data.data.Segregation": "lanka_data.api.data.Segregation",
    "lanka_data.dataset.Dataset": "lanka_data.api.dataset.Dataset",
    "lanka_data.dataset.DiffDataset": "lanka_data.api.dataset.DiffDataset",
    "lanka_data.dataset.RegionValueDataset": (
        "lanka_data.api.dataset.RegionValueDataset"
    ),
}.items():
    sys.modules[_old_name] = import_module(_new_name)
