"""Smoke tests that the plugin widgets instantiate inside a napari viewer.

These use napari's ``make_napari_viewer`` fixture (a real Qt viewer), so they
exercise widget construction, signal wiring, and the napari manifest without
needing any MAPS data on disk.
"""

from napari_maps_stitcher._maps_converter_widget import MapsConverterWidget
from napari_maps_stitcher._stitching_widget import StitchingWidget


def test_converter_widget_constructs(make_napari_viewer):
    """The converter widget builds with disabled, placeholder dropdowns."""
    viewer = make_napari_viewer()
    widget = MapsConverterWidget(viewer)

    # Dropdowns start disabled until a valid project folder is selected.
    assert not widget.layer_combo.isEnabled()
    assert not widget.acquisition_combo.isEnabled()
    assert widget.convert_button.text() == "Convert to OME-Zarr"


def test_stitching_widget_constructs(make_napari_viewer):
    """The stitching widget exposes both algorithms and default options."""
    viewer = make_napari_viewer()
    widget = StitchingWidget(viewer)

    algorithms = [
        widget.algorithm_combo.itemData(i)
        for i in range(widget.algorithm_combo.count())
    ]
    assert algorithms == ["multiview-stitcher", "sofima"]
    # Advanced options are collapsed by default.
    assert not widget.advanced_widget.isVisible()


def test_stitching_widget_algorithm_toggles_options(make_napari_viewer):
    """Switching to SOFIMA swaps which advanced options are shown."""
    viewer = make_napari_viewer()
    widget = StitchingWidget(viewer)

    sofima_index = next(
        i
        for i in range(widget.algorithm_combo.count())
        if widget.algorithm_combo.itemData(i) == "sofima"
    )
    widget.algorithm_combo.setCurrentIndex(sofima_index)

    # SOFIMA shows the stride input and hides the resolution selector.
    assert widget.stride_input.isVisibleTo(widget.advanced_widget)
    assert not widget.resolution_combo.isVisibleTo(widget.advanced_widget)
