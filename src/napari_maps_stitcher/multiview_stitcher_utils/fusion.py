import numpy as np


def overlay_fusion(
    transformed_views,
):
    """
    Simple fusion by overlaying the transformed views.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views

    Returns
    -------
    ndarray
        Maximum of input views at each pixel
    """
    output = np.zeros_like(transformed_views[0])
    for view in transformed_views:
        mask = view > 0
        output[mask] = view[mask]
    return output
