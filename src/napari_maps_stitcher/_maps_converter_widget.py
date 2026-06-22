"""QWidget for MAPS to OME-Zarr conversion.

Provides a user interface for converting MAPS projects to OME-Zarr format.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

from dask.diagnostics import ProgressBar
from maps_omezarr_converter import (
    MapsAcquisitionModel,
    convert_maps_to_omezarr,
)
from ome_zarr_converters_tools import OverwriteMode

# Sentinel dropdown entries for batch conversion.
ALL_LAYERS = "(All layers)"
ALL_ACQUISITIONS = "(All acquisitions in layer)"

# Namespace of the FEI/Thermo Fisher MAPS project XML.
_MAPS_PROJECT_NS = (
    "{http://schemas.datacontract.org/2004/07/Fei.Applications.Perseus.Project}"
)


def _list_acquisitions_from_xml(project: Path) -> list[tuple[str, str]]:
    """Parse ``MapsProject.xml`` for the project's real tile acquisitions.

    Mirrors the converter's own discovery: an acquisition is a ``displayName``
    under ``LayersData`` whose XML node also carries a ``columns`` and ``rows``
    grid (derived layers such as stitched images or line scans have neither and
    are skipped). Uses the standard library so the dropdown can be populated
    without depending on the converter's private parsing internals.
    """
    xml_path = project / "MapsProject.xml"
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    root = ET.parse(xml_path).getroot()
    # ElementTree has no parent pointers; build a child -> parent map.
    parents = {child: parent for parent in root.iter() for child in parent}

    acquisitions: list[tuple[str, str]] = []
    for display_name in root.iter(_MAPS_PROJECT_NS + "displayName"):
        text = display_name.text
        if not text or "LayersData" not in text:
            continue
        parent = parents.get(display_name)
        if parent is None:
            continue
        has_grid = (
            parent.find(_MAPS_PROJECT_NS + "columns") is not None
            and parent.find(_MAPS_PROJECT_NS + "rows") is not None
        )
        if not has_grid:
            continue
        parts = text.split("\\")
        if len(parts) < 3:
            continue
        acquisitions.append((parts[1], "\\".join(parts[2:])))
    return acquisitions


def _list_acquisitions_from_filesystem(project: Path) -> list[tuple[str, str]]:
    """Fallback discovery: scan ``LayersData/<layer>/<acquisition>`` folders.

    A folder is treated as a tile acquisition if it contains both MAPS tile
    TIFFs (``Tile_*.tif``) and a position list (``*--posList.txt``).
    """
    result: list[tuple[str, str]] = []
    layers_dir = project / "LayersData"
    if not layers_dir.is_dir():
        return result
    for layer_dir in sorted(p for p in layers_dir.iterdir() if p.is_dir()):
        for acq_dir in sorted(p for p in layer_dir.iterdir() if p.is_dir()):
            has_tiles = any(acq_dir.glob("Tile_*.tif"))
            has_poslist = any(acq_dir.glob("*--posList.txt"))
            if has_tiles and has_poslist:
                result.append((layer_dir.name, acq_dir.name))
    return result


def list_maps_acquisitions(project_path: str) -> list[tuple[str, str]]:
    """List the stitchable tile acquisitions in a MAPS project.

    Returns a list of ``(layer, acquisition_name)`` pairs, where ``layer`` is a
    folder under ``LayersData`` and ``acquisition_name`` a folder under it. The
    project's ``MapsProject.xml`` is the source of truth; if it is missing or
    unparsable, falls back to a filesystem scan.

    Args:
        project_path: Path to the MAPS project folder.

    Returns:
        List of ``(layer, acquisition_name)`` tuples (possibly empty).
    """
    project = Path(project_path)
    try:
        return _list_acquisitions_from_xml(project)
    except (OSError, ET.ParseError):
        # Missing/malformed XML: fall back to a filesystem scan.
        return _list_acquisitions_from_filesystem(project)


class ConversionWorker(QObject):
    """Worker object to run conversion in a separate thread.

    Signals:
        finished: Emitted when conversion completes (success or failure).
        error: Emitted on conversion error with error message.
        success: Emitted on successful conversion with the list of output zarr
            paths (one per converted acquisition).
    """

    finished = Signal()
    error = Signal(str)
    success = Signal(list)  # list[str] of zarr paths

    def __init__(
        self,
        output_dir: str,
        project_path: str,
        layer: str | None,
        acquisition_name: str | None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.project_path = project_path
        self.layer = layer
        self.acquisition_name = acquisition_name

    def run(self) -> None:
        """Run the conversion in a background thread."""
        try:
            with ProgressBar():
                results = convert_maps_to_omezarr(
                    zarr_dir=self.output_dir,
                    acquisitions=[
                        MapsAcquisitionModel(
                            project_path=self.project_path,
                            layer=self.layer,
                            acquisition_name=self.acquisition_name,
                        )
                    ],
                    overwrite=OverwriteMode.OVERWRITE,
                )

            # Collect the output zarr paths from the conversion result.
            zarr_paths = [
                update["zarr_url"]
                for result in results
                for update in result["image_list_updates"]
            ]

            if zarr_paths:
                self.success.emit(zarr_paths)
            else:
                self.error.emit("Conversion produced no output images.")

        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MapsConverterWidget(QWidget):
    """Widget for converting MAPS projects to OME-Zarr format.

    Provides folder selection, layer selection, and conversion functionality.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.thread = None
        self._acquisitions: list[tuple[str, str]] = []

        self.setLayout(QVBoxLayout())

        # Instructional text
        instructions = QLabel(
            "Convert MAPS microscopy data (tiled & 2D) to OME-Zarr format:\n"
            "1. Select your MAPS project folder\n"
            "2. Choose layer and acquisition to convert\n"
            "3. Select an output folder\n"
            "4. Click 'Convert to OME-Zarr'"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("QLabel { color: #888; margin-bottom: 10px; }")
        self.layout().addWidget(instructions)

        # Path selection row
        path_layout = QHBoxLayout()
        path_label = QLabel("MAPS project folder:")
        path_label.setToolTip(
            "Select the MAPS project folder containing the Project.mapsxml file."
        )
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(
            "Enter or browse for folder path..."
        )
        self.path_input.setToolTip(
            "Select the MAPS project folder containing the Project.mapsxml file."
        )
        self.path_input.textChanged.connect(self._on_path_changed)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setToolTip("Browse for MAPS project folder")
        self.browse_button.clicked.connect(self._on_browse_clicked)

        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)

        # Layer selection row
        layer_layout = QHBoxLayout()
        layer_label = QLabel("Layer:")
        layer_label.setToolTip(
            "Select which layer from the MAPS project to convert.\n"
            "Available options will populate after selecting a valid project folder."
        )
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("(Select a folder first)")
        self.layer_combo.setEnabled(False)
        self.layer_combo.setToolTip(
            "Select which layer from the MAPS project to convert.\n"
            "Available options will populate after selecting a valid project folder.\n"
            f"Choose '{ALL_LAYERS}' to convert every acquisition in the project."
        )
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(self.layer_combo)

        # Acquisition selection row
        acquisition_layout = QHBoxLayout()
        acquisition_label = QLabel("Acquisition:")
        acquisition_label.setToolTip(
            "Select which acquisition (tile set) within the layer to convert.\n"
            f"Choose '{ALL_ACQUISITIONS}' to convert every acquisition in the layer."
        )
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItem("(Select a folder first)")
        self.acquisition_combo.setEnabled(False)
        self.acquisition_combo.setToolTip(
            "Select which acquisition (tile set) within the layer to convert.\n"
            f"Choose '{ALL_ACQUISITIONS}' to convert every acquisition in the layer."
        )

        acquisition_layout.addWidget(acquisition_label)
        acquisition_layout.addWidget(self.acquisition_combo)

        # Output folder selection row
        output_layout = QHBoxLayout()
        output_label = QLabel("Output folder:")
        output_label.setToolTip(
            "Select the folder where the converted OME-Zarr file will be saved.\n"
            "The output will be a .zarr directory containing the converted data."
        )
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText(
            "Enter or browse for output folder..."
        )
        self.output_input.setToolTip(
            "Select the folder where the converted OME-Zarr file will be saved.\n"
            "The output will be a .zarr directory containing the converted data."
        )
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.setToolTip("Browse for output folder")
        self.output_browse_button.clicked.connect(
            self._on_output_browse_clicked
        )

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(self.output_browse_button)

        # Action row
        action_layout = QHBoxLayout()
        self.convert_button = QPushButton("Convert to OME-Zarr")
        self.convert_button.setToolTip(
            "Start the conversion process.\n"
            "This will convert the selected MAPS acquisition to OME-Zarr format.\n"
            "After conversion, the ROI Stitching widget will open automatically."
        )
        self.convert_button.clicked.connect(self._on_convert_clicked)

        action_layout.addWidget(self.convert_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)

        # Add to main layout
        self.layout().addLayout(path_layout)
        self.layout().addLayout(layer_layout)
        self.layout().addLayout(acquisition_layout)
        self.layout().addLayout(output_layout)
        self.layout().addLayout(action_layout)
        self.layout().addWidget(self.progress_bar)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        """Enable or disable all input widgets.

        Args:
            enabled: Whether to enable or disable the inputs.
        """
        self.path_input.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.layer_combo.setEnabled(enabled)
        self.acquisition_combo.setEnabled(enabled)
        self.output_input.setEnabled(enabled)
        self.output_browse_button.setEnabled(enabled)
        self.convert_button.setEnabled(enabled)

    def _on_path_changed(self, path: str) -> None:
        """Update layer list when path is changed (typed or pasted)."""
        if path and Path(path).exists():
            self._update_layer_list(path)
        else:
            # Clear layer/acquisition lists if path is invalid
            self.layer_combo.clear()
            self.layer_combo.addItem("(Select a folder first)")
            self.layer_combo.setEnabled(False)
            self._clear_acquisition_combo()

    def _on_browse_clicked(self) -> None:
        """Open file dialog to select MAPS project folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select MAPS Project Folder",
            str(Path.home()),
            QFileDialog.ShowDirsOnly,
        )
        if folder:
            self.path_input.setText(folder)
            self._update_layer_list(folder)

    def _on_output_browse_clicked(self) -> None:
        """Open file dialog to select output folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(Path.home()),
            QFileDialog.ShowDirsOnly,
        )
        if folder:
            self.output_input.setText(folder)

    def _clear_acquisition_combo(self) -> None:
        """Reset the acquisition dropdown to its disabled placeholder state."""
        self.acquisition_combo.clear()
        self.acquisition_combo.addItem("(Select a folder first)")
        self.acquisition_combo.setEnabled(False)

    def _update_layer_list(self, project_folder: str) -> None:
        """Discover acquisitions from the project and populate the dropdowns."""
        self._acquisitions = list_maps_acquisitions(project_folder)

        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()

        if not self._acquisitions:
            self.layer_combo.addItem("(No acquisitions found)")
            self.layer_combo.setEnabled(False)
            self.layer_combo.blockSignals(False)
            self._clear_acquisition_combo()
            return

        layers = sorted({layer for layer, _ in self._acquisitions})
        self.layer_combo.addItems(layers)
        if len(layers) > 1:
            self.layer_combo.addItem(ALL_LAYERS)
        self.layer_combo.setEnabled(True)
        self.layer_combo.blockSignals(False)

        # Populate acquisitions for the initially selected layer.
        self._on_layer_changed(self.layer_combo.currentText())

    def _on_layer_changed(self, layer: str) -> None:
        """Populate the acquisition dropdown for the selected layer."""
        if not hasattr(self, "acquisition_combo"):
            return

        self.acquisition_combo.clear()

        if layer == ALL_LAYERS:
            # Converting all layers: every acquisition is included.
            self.acquisition_combo.addItem(ALL_ACQUISITIONS)
            self.acquisition_combo.setEnabled(False)
            return

        if not layer or layer.startswith("("):
            # No valid layer selected.
            self.acquisition_combo.addItem("(Select a layer first)")
            self.acquisition_combo.setEnabled(False)
            return

        names = sorted(name for lyr, name in self._acquisitions if lyr == layer)
        self.acquisition_combo.addItems(names)
        if len(names) > 1:
            self.acquisition_combo.addItem(ALL_ACQUISITIONS)
        self.acquisition_combo.setEnabled(True)

    def _on_convert_clicked(self) -> None:
        """Convert MAPS project to OME-Zarr format."""
        project_path = self.path_input.text()
        if not project_path:
            print("Please select a project folder first.")
            return

        output_dir = self.output_input.text()
        if not output_dir:
            print("Please select an output folder.")
            return

        if not self.layer_combo.isEnabled():
            print("Please select a valid layer.")
            return

        layer_text = self.layer_combo.currentText()
        acq_text = self.acquisition_combo.currentText()

        if layer_text == ALL_LAYERS:
            # Convert every acquisition in every layer.
            layer = None
            acquisition_name = None
        elif layer_text.startswith("("):
            print("Please select a valid layer.")
            return
        else:
            layer = layer_text
            if acq_text == ALL_ACQUISITIONS:
                acquisition_name = None
            elif acq_text.startswith("("):
                print("Please select a valid acquisition.")
                return
            else:
                acquisition_name = acq_text

        if layer is None:
            print("Converting all acquisitions in all layers...")
        elif acquisition_name is None:
            print(f"Converting all acquisitions in layer '{layer}'...")
        else:
            print(f"Converting {acquisition_name} (layer: {layer})...")

        # Disable all widget inputs during conversion
        self._set_inputs_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Create worker and thread
        self.thread = QThread()
        self.worker = ConversionWorker(
            output_dir=output_dir,
            project_path=project_path,
            layer=layer,
            acquisition_name=acquisition_name,
        )
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.success.connect(self._on_conversion_success)
        self.worker.error.connect(self._on_conversion_error)

        # Start the thread
        self.thread.start()

    def _on_conversion_success(self, zarr_paths: list) -> None:
        """Handle successful conversion.

        ``zarr_paths`` may contain multiple outputs when a whole layer or the
        whole project was converted; for now only the first one is loaded.
        """
        zarr_path = zarr_paths[0]
        if len(zarr_paths) > 1:
            print(
                f"Conversion complete: {len(zarr_paths)} acquisitions converted. "
                f"Loading the first one ({zarr_path})."
            )
        else:
            print(f"Conversion complete: {zarr_path}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        # Open stitching widget and populate with zarr path
        print("Opening ROI Stitching widget...")
        try:
            from ._stitching_widget import StitchingWidget

            # Get or create the stitching widget
            stitching_widget = None
            for dock_widget in self.viewer.window.dock_widgets.values():
                # dock_widget can be tuple (widget, container) or just widget
                widget = (
                    dock_widget[0]
                    if isinstance(dock_widget, tuple)
                    else dock_widget
                )
                if isinstance(widget, StitchingWidget):
                    stitching_widget = widget
                    break

            if stitching_widget is None:
                # Create new stitching widget if it doesn't exist

                stitching_widget = StitchingWidget(self.viewer)

                # Find the converter widget's dock widget
                from qtpy.QtCore import Qt
                from qtpy.QtWidgets import QDockWidget

                converter_dock = None
                # Iterate through all QDockWidgets to find the one containing self
                for qt_dock in self.viewer.window._qt_window.findChildren(
                    QDockWidget
                ):
                    if qt_dock.widget() is self:
                        converter_dock = qt_dock
                        break

                # Add the stitching widget
                added_dock = self.viewer.window.add_dock_widget(
                    stitching_widget, name="ROI Stitching", area="right"
                )

                # Position it below the converter widget if found
                if converter_dock is not None and added_dock is not None:
                    try:
                        # Split: put added_dock below converter_dock
                        self.viewer.window._qt_window.splitDockWidget(
                            converter_dock, added_dock, Qt.Vertical
                        )

                        # Ensure widgets are visible
                        converter_dock.show()
                        added_dock.show()
                        converter_dock.raise_()

                    except (RuntimeError, ValueError) as e:
                        print(
                            f"Could not position stitching widget below converter: {e}"
                        )

            # Set the zarr path in the stitching widget
            stitching_widget.zarr_input.setText(zarr_path)

            # Trigger the load operation
            stitching_widget._on_load_clicked()

            print("Stitching widget opened and file loaded")
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
            print(f"Failed to open stitching widget: {e}")

    def _on_conversion_error(self, error_msg: str) -> None:
        """Handle conversion error."""
        print(f"Conversion failed: {error_msg}")
        self.progress_bar.setVisible(False)

    def _on_conversion_finished(self) -> None:
        """Re-enable widget inputs after conversion."""
        self._set_inputs_enabled(True)

    def closeEvent(self, event):
        """Clean up threads when widget is closed."""
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)
