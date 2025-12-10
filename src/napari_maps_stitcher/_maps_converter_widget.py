"""QWidget for MAPS to OME-Zarr conversion.

Provides a user interface for converting MAPS projects to OME-Zarr format.
"""

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
from maps_omezarr_converter.wrappers import convert_maps_to_omezarr


class ConversionWorker(QObject):
    """Worker object to run conversion in a separate thread.

    Signals:
        finished: Emitted when conversion completes (success or failure).
        error: Emitted on conversion error with error message.
        success: Emitted on successful conversion with output zarr path.
    """

    finished = Signal()
    error = Signal(str)
    success = Signal(str)  # zarr_path

    def __init__(
        self,
        output_dir: str,
        project_path: str,
        acquisition_name: str,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.project_path = project_path
        self.acquisition_name = acquisition_name

    def run(self) -> None:
        """Run the conversion in a background thread."""
        try:
            with ProgressBar():
                convert_maps_to_omezarr(
                    zarr_dir=self.output_dir,
                    project_path=self.project_path,
                    acquisition_names=[self.acquisition_name],
                    overwrite=True,
                )

            # Build zarr path (spaces replaced with underscores)
            zarr_filename = self.acquisition_name.replace(" ", "_") + ".zarr"
            zarr_path = Path(self.output_dir) / zarr_filename

            if zarr_path.exists():
                self.success.emit(str(zarr_path))
            else:
                self.error.emit(f"Expected zarr file not found at {zarr_path}")

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

        self.setLayout(QVBoxLayout())

        # Instructional text
        instructions = QLabel(
            "Convert MAPS microscopy data (tiled & 2D) to OME-Zarr format:\n"
            "1. Select your MAPS project folder\n"
            "2. Choose the layer to convert\n"
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
            "Available options will populate after selecting a valid project folder."
        )

        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(self.layer_combo)

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
        self.output_input.setEnabled(enabled)
        self.output_browse_button.setEnabled(enabled)
        self.convert_button.setEnabled(enabled)

    def _on_path_changed(self, path: str) -> None:
        """Update layer list when path is changed (typed or pasted)."""
        if path and Path(path).exists():
            self._update_layer_list(path)
        else:
            # Clear layer list if path is invalid
            self.layer_combo.clear()
            self.layer_combo.addItem("(Select a folder first)")
            self.layer_combo.setEnabled(False)

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

    def _update_layer_list(self, project_folder: str) -> None:
        """Scan LayersData/Layer folder and populate dropdown."""
        self.layer_combo.clear()
        layers_path = Path(project_folder) / "LayersData" / "Layer"

        if not layers_path.exists() or not layers_path.is_dir():
            self.layer_combo.addItem("(No LayersData/Layer folder found)")
            self.layer_combo.setEnabled(False)
            return

        # Get all subdirectories
        layer_folders = [f.name for f in layers_path.iterdir() if f.is_dir()]

        if not layer_folders:
            self.layer_combo.addItem("(No layer folders found)")
            self.layer_combo.setEnabled(False)
            return

        # Populate dropdown with found folders
        self.layer_combo.addItems(sorted(layer_folders))
        self.layer_combo.setEnabled(True)

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

        acquisition_name = self.layer_combo.currentText()
        if not self.layer_combo.isEnabled() or acquisition_name.startswith(
            "("
        ):
            print("Please select a valid layer.")
            return

        print(f"Converting {acquisition_name}...")

        # Disable all widget inputs during conversion
        self._set_inputs_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Create worker and thread
        self.thread = QThread()
        self.worker = ConversionWorker(
            output_dir=output_dir,
            project_path=project_path,
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

    def _on_conversion_success(self, zarr_path: str) -> None:
        """Handle successful conversion."""
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
