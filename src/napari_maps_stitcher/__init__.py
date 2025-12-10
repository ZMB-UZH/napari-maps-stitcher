try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._maps_converter_widget import MapsConverterWidget
from ._stitching_widget import StitchingWidget

__all__ = (
    "MapsConverterWidget",
    "StitchingWidget",
)
