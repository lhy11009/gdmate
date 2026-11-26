"""Tools for working with ASPECT models and data."""

from gdmate.aspect_tools.fields import (
    PointFieldSampler2D,
    read_point_fields,
    sample_point_fields_2d,
)
from gdmate.aspect_tools.io import (
    parse_parameters_to_dict,
    save_parameters_from_dict,
)
from gdmate.aspect_tools.log import (
    AspectGraphicalOutput,
    AspectLogParser,
    AspectSolverOutput,
    AspectTimestep,
    AspectTimingBlock,
    AspectTimingSection,
    UnsupportedAspectVersionError,
)
from gdmate.aspect_tools.run import AspectCaseResult, run_aspect_case
from gdmate.aspect_tools.statistics import AspectStatistics

__all__ = [
    "parse_parameters_to_dict",
    "save_parameters_from_dict",
    "AspectLogParser",
    "AspectSolverOutput",
    "AspectGraphicalOutput",
    "AspectTimestep",
    "AspectTimingBlock",
    "AspectTimingSection",
    "UnsupportedAspectVersionError",
    "AspectStatistics",
    "AspectCaseResult",
    "run_aspect_case",
    "PointFieldSampler2D",
    "read_point_fields",
    "sample_point_fields_2d",
]
