from lanka_data.api.command.CommandLoaderMixin import CommandLoaderMixin
from lanka_data.api.data import (
    DataSource,
    Segregation,
    SegregationComputeMixin,
)
from lanka_data.api.dataset import Dataset, DiffDataset, RegionValueDataset
from lanka_data.api.dataset import RegionValueDatasetTableMixin
from lanka_data.datasets.command import (
    Command,
    CommandBase,
    CommandHelp,
    CommandRunner,
)
from lanka_data.datasets.data import Diversity, FieldNameUtils
from lanka_data.datasets.dataset import Census2001Dataset, Census2012Dataset
from lanka_data.datasets.dataset import Census2024Dataset, DatasetFactory
from lanka_data.datasets.dataset import (
    ElectionDataset,
    ElectionSummaryDataset,
)
from lanka_data.datasets.dataset import EmptyDataset, GIG2Dataset
from lanka_data.examples import Example, ExampleOutputMixin
from lanka_data.readme import ReadMe, ReadMeExamplesItemMixin
from lanka_data.readme import ReadMeExamplesMixin, ReadMeFooterMixin
from lanka_data.readme import ReadMeSourcesMixin, ReadMeUsageMixin
from lanka_data.datasets.region import RegionFetchMixin, RegionLoadersMixin
from lanka_data.datasets.region import RegionParentMixin, RegionParserMixin
from lanka_data.datasets.region import (
    RegionParserRadiusMixin,
    RegionRawDataMixin,
)
from lanka_data.datasets.region import Regions, RegionTypeUtils, Where
from lanka_data.visual import BarChartDrawMixin, BarChartLabelMixin
from lanka_data.visual import BarChartVisual, BumpChartDataMixin
from lanka_data.visual import BumpChartDrawMixin, BumpChartVisual
from lanka_data.visual import ColorSpec, ColorSpecCategoryMixin
from lanka_data.visual import (
    ColorSpecConstants,
    ColorSpecCustomMixin,
)
from lanka_data.visual import ColorSpecFactory, ColorSpecHelpers
from lanka_data.visual import (
    ColorSpecHelpersMixin,
    Font,
    Footer,
    GeoData,
)
from lanka_data.visual import (
    GeoDataLoaderMixin,
    Header,
    HowFormatter,
)
from lanka_data.visual import (
    JSONVisual,
    Label,
    LabelFit,
    Legend,
    MapVisual,
)
from lanka_data.visual import PieChartGridMixin, PieChartVisual, Plot
from lanka_data.visual import PlotVisual, Text, Visual, VisualFactory
from lanka_data.visual import WhatFormatter, WhereFormatter
from lanka_data.api.command.fields.CensusDatasetRegistry import (
    CensusDatasetRegistry,
)
from lanka_data.api.command.fields.ElectionDatasetRegistry import (
    ElectionDatasetRegistry,
)
from lanka_data.api.command.fields.RegionTypeRegistry import (
    RegionTypeRegistry,
)

CensusDatasetRegistry.set_dataset_classes(
    [
        Census2001Dataset,
        Census2012Dataset,
        Census2024Dataset,
    ]
)
ElectionDatasetRegistry.set_dataset_class(ElectionDataset)
RegionTypeRegistry.set_prefix_maps(RegionTypeUtils.get_prefix_maps())
