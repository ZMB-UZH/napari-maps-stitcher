"""Core stitching dispatcher - routes to specific stitching algorithms."""
from pathlib import Path

from ngio import Roi

from .stitching_algorithms import MultiviewStitcherAlgorithm, SofimaAlgorithm
from .stitching_base import StitchingAlgorithm

# Available stitching algorithms
STITCHING_ALGORITHMS = {
    "multiview-stitcher": MultiviewStitcherAlgorithm(),
    "sofima": SofimaAlgorithm(),
}


def get_available_algorithms() -> list[str]:
    """Get list of available stitching algorithm names.

    Returns:
        List of algorithm names.
    """
    return list(STITCHING_ALGORITHMS.keys())


def get_algorithm(algorithm_name: str) -> StitchingAlgorithm:
    """Get a stitching algorithm by name.

    Args:
        algorithm_name: Name of the algorithm.

    Returns:
        StitchingAlgorithm instance.

    Raises:
        ValueError: If algorithm name is not recognized.
    """
    if algorithm_name not in STITCHING_ALGORITHMS:
        available = ", ".join(STITCHING_ALGORITHMS.keys())
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Available algorithms: {available}"
        )
    return STITCHING_ALGORITHMS[algorithm_name]


def stitch_rois(
    input_zarr_url: str | Path,
    output_zarr_url: str | Path,
    rois: list[Roi],
    algorithm: str = "multiview-stitcher",
    **kwargs,
) -> None:
    """Stitch ROIs using the specified algorithm.

    Args:
        input_zarr_url: Path to the input OME-Zarr container.
        output_zarr_url: Path to the output OME-Zarr container.
        rois: List of ROIs to stitch.
        algorithm: Name of the stitching algorithm to use.
            Default: "multiview-stitcher".
        **kwargs: Algorithm-specific parameters.

    Raises:
        ValueError: If algorithm name is not recognized.
    """
    stitcher = get_algorithm(algorithm)
    stitcher.stitch(input_zarr_url, output_zarr_url, rois, **kwargs)
