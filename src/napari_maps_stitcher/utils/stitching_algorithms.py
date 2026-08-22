"""Stitching algorithm implementations (multiview-stitcher, SOFIMA)."""

from pathlib import Path
from typing import Any

from ngio import Roi

from ..multiview_stitcher_utils.intensity_correction import (
    DEFAULT_DRIFT_LENGTH,
)
from .stitching_base import StitchingAlgorithm
from .stitching_multiview import stitch_rois_multiview_stitcher


class MultiviewStitcherAlgorithm(StitchingAlgorithm):
    """Multiview-stitcher based stitching algorithm."""

    def stitch(
        self,
        input_zarr_url: str | Path,
        output_zarr_url: str | Path,
        rois: list[Roi],
        **kwargs: Any,
    ) -> None:
        """Stitch ROIs using multiview-stitcher.

        Args:
            input_zarr_url: Path to the input OME-Zarr container.
            output_zarr_url: Path to the output OME-Zarr container.
            rois: List of ROIs to stitch.
            **kwargs: Algorithm parameters (resolution_path, blend,
                z_project, intensity_correction, intensity_estimator,
                intensity_drift_length).
        """
        stitch_rois_multiview_stitcher(
            input_zarr_url=input_zarr_url,
            output_zarr_url=output_zarr_url,
            rois=rois,
            resolution_path=kwargs.get("resolution_path"),
            blend=kwargs.get("blend", False),
            z_project=kwargs.get("z_project", True),
            intensity_correction=kwargs.get("intensity_correction", "offset"),
            intensity_estimator=kwargs.get(
                "intensity_estimator", "quantile"
            ),
            intensity_drift_length=kwargs.get(
                "intensity_drift_length", DEFAULT_DRIFT_LENGTH
            ),
        )

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameters for multiview-stitcher.

        Returns:
            Dictionary of parameter names and their default values.
        """
        return {
            "resolution_path": None,
            "blend": False,
            "z_project": True,
            "intensity_correction": "offset",
            "intensity_estimator": "quantile",
            "intensity_drift_length": DEFAULT_DRIFT_LENGTH,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for multiview-stitcher parameters.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {
            "resolution_path": "Resolution pyramid level to use for tile registration.\n"
            "Lower resolutions are faster but might be less precise.\n"
            "The output will always be at full resolution.",
            "blend": "Enable weighted average blending in overlapping regions.\n"
            "When enabled: Smooth transitions between tiles (slower).\n"
            "When disabled: Overlay fusion with sharp boundaries (faster).",
            "z_project": "Project z-axis for registration (internal parameter).",
            "intensity_correction": "Even out brightness differences between "
            "tiles before fusion.\n"
            "'none': no correction.\n"
            "'offset': match tile brightness only (more robust).\n"
            "'affine': match tile brightness and contrast.",
            "intensity_estimator": "How overlapping tiles are compared.\n"
            "'quantile': match intensity distributions; does not need an "
            "accurate registration.\n"
            "'pixel': compare co-located pixels; only valid when the "
            "registration is pixel-accurate.",
            "intensity_drift_length": "How far, in tiles, the intensity "
            "correction may drift before it is pulled back toward leaving the "
            "tile unchanged.\nLower values guard harder against one side of a "
            "large mosaic ending up dark.",
        }


class SofimaAlgorithm(StitchingAlgorithm):
    """SOFIMA stitching algorithm."""

    def stitch(
        self,
        input_zarr_url: str | Path,
        output_zarr_url: str | Path,
        rois: list[Roi],
        **kwargs: Any,
    ) -> None:
        """Stitch ROIs using SOFIMA.

        Args:
            input_zarr_url: Path to the input OME-Zarr container.
            output_zarr_url: Path to the output OME-Zarr container.
            rois: List of ROIs to stitch.
            **kwargs: Algorithm parameters (stride).
        """
        # Imported lazily: SOFIMA pulls in jax, whose native DLLs are fragile
        # on some platforms (e.g. Windows). Deferring the import keeps the
        # plugin and the multiview-stitcher path importable without jax.
        from .stitching_sofima import stitch_rois_sofima

        stitch_rois_sofima(
            input_zarr_url=input_zarr_url,
            output_zarr_url=output_zarr_url,
            rois=rois,
            stride=kwargs.get("stride", 20),
        )

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameters for SOFIMA.

        Returns:
            Dictionary of parameter names and their default values.
        """

        return {
            "stride": 20,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for SOFIMA parameters.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """

        return {
            "stride": (
                "Pixel stride for flow estimation; lower values improve accuracy "
                "but increase runtime."
            ),
        }

