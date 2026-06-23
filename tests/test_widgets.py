"""Smoke tests that the plugin widgets construct and wire up correctly.

These use ``pytest-qt``'s ``qtbot`` (which only guarantees a ``QApplication``)
together with a lightweight mock viewer. The widgets only store ``self.viewer``
at construction time and build their Qt children, so a real napari viewer is not
needed -- and avoiding it keeps the tests off the offscreen-OpenGL path that is a
common source of fatal crashes on headless CI.
"""

from unittest.mock import MagicMock

from napari_maps_stitcher._maps_converter_widget import MapsConverterWidget
from napari_maps_stitcher._stitching_widget import StitchingWidget


def test_converter_widget_constructs(qtbot):
    """The converter widget builds with disabled, placeholder dropdowns."""
    widget = MapsConverterWidget(MagicMock())
    qtbot.addWidget(widget)

    # Dropdowns start disabled until a valid project folder is selected.
    assert not widget.layer_combo.isEnabled()
    assert not widget.acquisition_combo.isEnabled()
    assert widget.convert_button.text() == "Convert to OME-Zarr"


def test_stitching_widget_constructs(qtbot):
    """The stitching widget exposes both algorithms and default options."""
    widget = StitchingWidget(MagicMock())
    qtbot.addWidget(widget)

    algorithms = [
        widget.algorithm_combo.itemData(i)
        for i in range(widget.algorithm_combo.count())
    ]
    assert algorithms == ["multiview-stitcher", "sofima"]
    # Advanced options are collapsed by default.
    assert not widget.advanced_widget.isVisible()


def test_stitching_widget_algorithm_toggles_options(qtbot):
    """Switching to SOFIMA swaps which advanced options are shown."""
    widget = StitchingWidget(MagicMock())
    qtbot.addWidget(widget)

    sofima_index = next(
        i
        for i in range(widget.algorithm_combo.count())
        if widget.algorithm_combo.itemData(i) == "sofima"
    )
    widget.algorithm_combo.setCurrentIndex(sofima_index)

    # SOFIMA shows the stride input and hides the resolution selector.
    assert widget.stride_input.isVisibleTo(widget.advanced_widget)
    assert not widget.resolution_combo.isVisibleTo(widget.advanced_widget)
