"""Base class for stitching algorithms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ngio import Roi


class StitchingAlgorithm(ABC):
    """Abstract base class for stitching algorithms."""

    @abstractmethod
    def stitch(
        self,
        input_zarr_url: str | Path,
        output_zarr_url: str | Path,
        rois: list[Roi],
        **kwargs: Any,
    ) -> None:
        """Stitch ROIs using the algorithm.

        Args:
            input_zarr_url: Path to the input OME-Zarr container.
            output_zarr_url: Path to the output OME-Zarr container.
            rois: List of ROIs to stitch.
            **kwargs: Algorithm-specific parameters.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """Get default parameters for the algorithm.

        Returns:
            Dictionary of parameter names and their default values.
        """
        ...

    @abstractmethod
    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        ...
