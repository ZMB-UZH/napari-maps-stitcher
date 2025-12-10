try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget
from ._maps_converter_widget import MapsConverterWidget
from ._stitching_widget import StitchingWidget

__all__ = (
    "ExampleQWidget",
    "ImageThreshold",
    "MapsConverterWidget",
    "StitchingWidget",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
)
