import numpy as np


def overlay_fusion(
    transformed_views,
):
    """
    Simple fusion by overlaying the transformed views.

    Later views win where several views cover the same pixel.

    ``multiview_stitcher.fusion`` resamples each view with ``cval=np.nan``, so
    ``NaN`` marks the pixels a view does not cover. Testing for that is what
    tells "outside the view" apart from "genuinely dark" -- a brightness test
    would drop black pixels and punch holes into the mosaic.

    Parameters
    ----------
    transformed_views : list of ndarrays
        transformed input views, ``NaN`` outside each view

    Returns
    -------
    ndarray
        Overlay of input views, 0 where no view covers the pixel
    """
    output = np.zeros_like(transformed_views[0])
    for view in transformed_views:
        mask = ~np.isnan(view)
        output[mask] = view[mask]
    return output
