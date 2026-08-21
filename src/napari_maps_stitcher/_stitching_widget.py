"""QWidget for stitching ROIs from OME-Zarr files.

Provides a user interface for stitching regions of interest.
"""

from typing import TYPE_CHECKING

import napari
import numpy as np
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

from .utils.omezarr_utils import get_levels_paths_dict


class StitchingWorker(QObject):
    """Worker for running stitching in a background thread.

    Signals:
        finished: Emitted when stitching completes (success or failure).
        error: Emitted on stitching error with error message.
        success: Emitted on successful completion with list of output paths.
        progress: Emitted after each ROI completion with (current, total) counts.
        roi_completed: Emitted after each individual ROI is stitched with the output path.
    """

    finished = Signal()
    error = Signal(str)
    success = Signal(list)
    progress = Signal(int, int)  # current, total
    roi_completed = Signal(str)  # output_path

    def __init__(
        self,
        zarr_path: str,
        roi_layer,
        get_shapes_func,
        algorithm: str,
        resolution_path: str | None,
        blend: bool,
        stride: int | None,
        output_format: str,
    ):
        """Initialize the stitching worker.

        Args:
            zarr_path: Path to the input OME-Zarr file.
            roi_layer: Napari shapes layer containing ROIs.
            get_shapes_func: Function to extract shapes from the layer.
            algorithm: Name of the stitching algorithm to use.
            resolution_path: Resolution path for registration (algorithm-specific).
            blend: Whether to use blending for fusion (algorithm-specific).
            stride: Stride parameter for SOFIMA algorithm.
            output_format: Output format ('zarr' or 'tiff').
        """
        super().__init__()
        self.zarr_path = zarr_path
        self.roi_layer = roi_layer
        self.get_shapes_func = get_shapes_func
        self.algorithm = algorithm
        self.resolution_path = resolution_path
        self.blend = blend
        self.stride = stride
        self.output_format = output_format

    def run(self):
        """Execute the stitching process."""
        try:
            # Import here to access the module for patching
            from .utils import roi_stitching

            # Prepare stitching data
            shapes, output_paths = roi_stitching.prepare_roi_stitching(
                self.zarr_path,
                self.roi_layer,
                self.get_shapes_func,
                self.output_format,
            )

            total_rois = len(shapes)
            print(f"\nStarting stitching of {total_rois} ROI(s)...")

            # Emit initial progress to show "Stitching ROI 1/N"
            self.progress.emit(0, total_rois)

            # Stitch each ROI
            completed_paths = []
            for i, (shape, output_path) in enumerate(
                zip(shapes, output_paths, strict=True)
            ):
                print(f"\nProcessing ROI {i + 1}/{total_rois}:")
                final_path = roi_stitching.stitch_single_roi(
                    self.zarr_path,
                    output_path,
                    shape,
                    self.algorithm,
                    self.resolution_path,
                    self.blend,
                    self.stride,
                    self.output_format,
                )
                completed_paths.append(final_path)
                # Emit progress and individual ROI completion
                self.progress.emit(i + 1, total_rois)
                self.roi_completed.emit(str(final_path))

            print(f"\nCompleted stitching {total_rois} ROI(s)")
            self.success.emit(completed_paths)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class StitchingWidget(QWidget):
    """Widget for stitching ROIs from OME-Zarr files.

    Provides zarr file selection and stitching functionality.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.stitching_thread = None
        self.stitching_worker = None

        self.setLayout(QVBoxLayout())

        # Instructional text
        instructions = QLabel(
            "Stitch regions of interest.\n"
            "1. Select an OME-Zarr file and click 'Load OME-Zarr'\n"
            "   (The MAPS to OME-Zarr converter plugin automatically does this)\n"
            "2. Draw regions of interest (ROIs) in the 'ROI_layer'\n"
            "3. Click 'Stitch ROIs' to process and stitch your regions\n"
            "   The outputs will be saved next to the original .zarr\n"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "QLabel { color: #888; margin-bottom: 10px; }"
        )
        self.layout().addWidget(instructions)

        # Zarr path selection row
        zarr_layout = QHBoxLayout()
        zarr_label = QLabel("OME-Zarr file:")
        zarr_label.setToolTip(
            "Select the OME-Zarr file to load for stitching.\n"
            "This should be a .zarr directory created with the MAPS OME-Zarr converter."
        )
        self.zarr_input = QLineEdit()
        self.zarr_input.setPlaceholderText("Enter or browse for .zarr file...")
        self.zarr_input.setToolTip(
            "Select the OME-Zarr file to load for stitching.\n"
            "This should be a .zarr directory created with the MAPS OME-Zarr converter."
        )
        self.zarr_browse_button = QPushButton("Browse...")
        self.zarr_browse_button.setToolTip("Browse for OME-Zarr file")
        self.zarr_browse_button.clicked.connect(self._on_zarr_browse_clicked)

        zarr_layout.addWidget(zarr_label)
        zarr_layout.addWidget(self.zarr_input)
        zarr_layout.addWidget(self.zarr_browse_button)

        self.layout().addLayout(zarr_layout)

        # Load button row
        load_layout = QHBoxLayout()
        self.load_button = QPushButton("Load OME-Zarr")
        self.load_button.setToolTip(
            "Load the selected OME-Zarr file into napari viewer.\n"
            "This will also create an 'ROI_layer' for drawing regions of interest."
        )
        self.load_button.clicked.connect(self._on_load_clicked)
        load_layout.addWidget(self.load_button)

        self.layout().addLayout(load_layout)

        # Algorithm selection row
        algorithm_layout = QHBoxLayout()
        algorithm_label = QLabel("Stitching algorithm:")
        algorithm_label.setToolTip("Select the stitching algorithm to use.")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem(
            "Multiview-Stitcher", "multiview-stitcher"
        )
        self.algorithm_combo.addItem("SOFIMA (experimental)", "sofima")
        # Add more algorithms as they become available
        self.algorithm_combo.setToolTip(
            "Select the stitching algorithm to use."
        )
        self.algorithm_combo.currentIndexChanged.connect(
            self._on_algorithm_changed
        )
        algorithm_layout.addWidget(algorithm_label)
        algorithm_layout.addWidget(self.algorithm_combo)
        self.layout().addLayout(algorithm_layout)

        # Output format selection row
        output_format_layout = QHBoxLayout()
        output_format_label = QLabel("Output format:")
        output_format_label.setToolTip(
            "Select the output file format for stitched ROIs.\n"
            "OME-Zarr: Multi-resolution, efficient for large images.\n"
            "OME-TIFF: Standard format, compatible with most software."
        )
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("OME-Zarr (.zarr)", "zarr")
        self.output_format_combo.addItem("OME-TIFF (.ome.tif)", "tiff")
        self.output_format_combo.setToolTip(
            "Select the output file format for stitched ROIs.\n"
            "OME-Zarr: Multi-resolution, efficient for large images.\n"
            "OME-TIFF: Standard format, compatible with most software."
        )
        output_format_layout.addWidget(output_format_label)
        output_format_layout.addWidget(self.output_format_combo)
        self.layout().addLayout(output_format_layout)

        # Advanced Options toggle button
        self.advanced_toggle = QPushButton("▶ Advanced Options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolTip(
            "Show/hide advanced stitching options."
        )
        self.advanced_toggle.clicked.connect(self._toggle_advanced_options)
        self.layout().addWidget(self.advanced_toggle)

        # Advanced Options container (hidden by default)
        self.advanced_widget = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(20, 0, 0, 0)  # Indent the options

        # Resolution path option
        resolution_layout = QHBoxLayout()
        self.resolution_label = QLabel("Resolution for registration:")
        self.resolution_label.setToolTip(
            "Select which resolution pyramid level to use for tile registration.\n"
            "Lower resolutions are faster but might be less precise.\n"
            "The output will always be at full resolution."
        )
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("Auto (lowest resolution)", None)
        self.resolution_combo.setToolTip(
            "Select which resolution pyramid level to use for tile registration.\n"
            "Lower resolutions are faster but might be less precise.\n"
            "The output will always be at full resolution."
        )
        resolution_layout.addWidget(self.resolution_label)
        resolution_layout.addWidget(self.resolution_combo)
        advanced_layout.addLayout(resolution_layout)

        # Blend option (multiview-stitcher only)
        blend_layout = QHBoxLayout()
        self.blend_label = QLabel("Blend overlapping tiles:")
        self.blend_label.setToolTip(
            "Enable weighted average blending in overlapping regions.\n"
            "When enabled: Smooth transitions between tiles.\n"
            "When disabled: Overlay fusion with sharp boundaries."
        )
        self.blend_checkbox = QCheckBox()
        self.blend_checkbox.setChecked(False)
        self.blend_checkbox.setToolTip(
            "Enable weighted average blending in overlapping regions.\n"
            "When enabled: Smooth transitions between tiles.\n"
            "When disabled: Overlay fusion with sharp boundaries."
        )
        blend_layout.addWidget(self.blend_label)
        blend_layout.addWidget(self.blend_checkbox)
        advanced_layout.addLayout(blend_layout)

        # SOFIMA stride option
        stride_layout = QHBoxLayout()
        self.stride_label = QLabel("SOFIMA stride (px):")
        self.stride_label.setToolTip(
            "Pixel stride for SOFIMA flow estimation; lower is slower but more accurate."
        )
        self.stride_input = QSpinBox()
        self.stride_input.setRange(1, 512)
        self.stride_input.setValue(20)
        self.stride_input.setToolTip(
            "Pixel stride for SOFIMA flow estimation; lower is slower but more accurate."
        )
        stride_layout.addWidget(self.stride_label)
        stride_layout.addWidget(self.stride_input)
        advanced_layout.addLayout(stride_layout)

        self.advanced_widget.setLayout(advanced_layout)
        self.advanced_widget.setVisible(False)  # Hidden by default
        self.layout().addWidget(self.advanced_widget)

        # Initialize advanced option visibility based on default algorithm
        self._on_algorithm_changed(self.algorithm_combo.currentIndex())

        # Stitch button row
        stitch_layout = QHBoxLayout()
        self.stitch_button = QPushButton("Stitch ROIs")
        self.stitch_button.setToolTip(
            "Start stitching all ROIs drawn in the 'ROI_layer'.\n"
            "Each ROI will be processed separately and saved as a new OME-Zarr file."
        )
        self.stitch_button.clicked.connect(self._on_stitch_clicked)
        stitch_layout.addWidget(self.stitch_button)

        self.layout().addLayout(stitch_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout().addWidget(self.progress_bar)

    def _on_zarr_browse_clicked(self) -> None:
        """Open file dialog to select a zarr file."""
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Select OME-Zarr file",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if selected_path:
            # Ensure it's a .zarr directory
            if selected_path.endswith(".zarr"):
                self.zarr_input.setText(selected_path)
            else:
                print("Warning: Selected directory does not end with .zarr")
                self.zarr_input.setText(selected_path)

    def _toggle_advanced_options(self) -> None:
        """Toggle visibility of advanced options."""
        is_visible = self.advanced_toggle.isChecked()
        self.advanced_widget.setVisible(is_visible)
        # Update button text to show expand/collapse state
        if is_visible:
            self.advanced_toggle.setText("▼ Advanced Options")
        else:
            self.advanced_toggle.setText("▶ Advanced Options")

    def _on_load_clicked(self) -> None:
        """Load the selected zarr file into napari."""
        zarr_path = self.zarr_input.text()
        if not zarr_path:
            print("Please select a zarr file first.")
            return

        try:
            print(f"Loading {zarr_path}...")

            # Track the number of layers before opening
            num_layers_before = len(self.viewer.layers)

            # Open the zarr file
            self.viewer.open(zarr_path, plugin="napari-ome-zarr")

            # Get the zarr file name without extension
            from pathlib import Path

            zarr_name = Path(zarr_path).stem

            # Rename any newly added layers to use the zarr name
            for i in range(num_layers_before, len(self.viewer.layers)):
                layer = self.viewer.layers[i]
                # If there's only one new layer, use the zarr name directly
                # If multiple layers, append the original suffix
                if len(self.viewer.layers) - num_layers_before == 1:
                    layer.name = zarr_name
                else:
                    # Keep any suffix from the original name (e.g., channel info)
                    original_name = layer.name
                    layer.name = f"{zarr_name}_{original_name}"

            print(f"Loaded {zarr_path} into napari viewer")

            # Add ROI shapes layer if it doesn't exist
            if "ROI_layer" not in [layer.name for layer in self.viewer.layers]:
                # Inherit the image layer's units. Without this the shapes
                # layer defaults to "pixel" while the OME-Zarr image is in
                # micrometer, and napari drops units for rendering.
                image_layer = self.viewer.layers[num_layers_before]
                shapes_layer = self.viewer.add_shapes(
                    name="ROI_layer",
                    units=image_layer.units[-2:],
                    axis_labels=image_layer.axis_labels[-2:],
                )
                shapes_layer.mode = "add_rectangle"
                print("Added shapes layer: ROI_layer")

            # Populate resolution options
            self._populate_resolution_options(zarr_path)

            # Change button text to indicate reload option
            self.load_button.setText("Reload OME-Zarr")

        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            print(f"Failed to load zarr file: {e}")

    def _populate_resolution_options(self, zarr_path: str) -> None:
        """Populate the resolution dropdown with available levels.

        Args:
            zarr_path: Path to the OME-Zarr file.
        """
        # Clear existing items except the first (Auto)
        self.resolution_combo.clear()
        self.resolution_combo.addItem("Auto (lowest resolution)", None)

        # Get available resolution levels
        levels_dict = get_levels_paths_dict(zarr_path)

        # Add each resolution level to the dropdown
        for label, path in levels_dict.items():
            self.resolution_combo.addItem(f"{label} (level {path})", path)

    def _get_shapes_from_layer(
        self, shapes_layer: napari.layers.Shapes
    ) -> list[np.ndarray]:
        """Extract all shapes from a napari Shapes layer.

        Args:
            shapes_layer: A napari Shapes layer containing ROI definitions.

        Returns:
            List of numpy arrays representing the shape coordinates,
            scaled by the layer's scaling.
        """
        shapes = []
        if isinstance(shapes_layer, napari.layers.Shapes):
            for shape in shapes_layer.data:
                if len(shape) >= 2:  # Ensure it's a valid shape
                    # Scale coordinates by the layer's scale
                    shapes.append(shape * shapes_layer.scale)
        return shapes

    def _on_stitch_clicked(self) -> None:
        """Stitch the defined ROIs."""
        zarr_path = self.zarr_input.text()
        if not zarr_path:
            print("Please select a zarr file first.")
            return

        # Find the ROI_layer
        roi_layer = None
        for layer in self.viewer.layers:
            if layer.name == "ROI_layer":
                roi_layer = layer
                break

        if roi_layer is None:
            print("ROI_layer not found. Please load a zarr file first.")
            return

        # Get selected algorithm
        algorithm = self.algorithm_combo.currentData()

        # Get selected resolution path (multiview only)
        resolution_path = (
            self.resolution_combo.currentData()
            if algorithm == "multiview-stitcher"
            else None
        )

        # Get blend option (multiview only)
        blend = (
            self.blend_checkbox.isChecked()
            if algorithm == "multiview-stitcher"
            else False
        )

        # Get stride (SOFIMA only)
        stride = self.stride_input.value() if algorithm == "sofima" else None

        # Get output format
        output_format = self.output_format_combo.currentData()

        # Disable all inputs during stitching
        self._set_inputs_enabled(False)

        # Create worker and thread
        self.stitching_worker = StitchingWorker(
            zarr_path,
            roi_layer,
            self._get_shapes_from_layer,
            algorithm,
            resolution_path,
            blend,
            stride,
            output_format,
        )
        self.stitching_thread = QThread()
        self.stitching_worker.moveToThread(self.stitching_thread)

        # Connect signals
        self.stitching_thread.started.connect(self.stitching_worker.run)
        self.stitching_worker.finished.connect(self.stitching_thread.quit)
        self.stitching_worker.finished.connect(self._on_stitching_finished)
        self.stitching_worker.success.connect(self._on_stitching_success)
        self.stitching_worker.error.connect(self._on_stitching_error)
        self.stitching_worker.progress.connect(self._on_stitching_progress)
        self.stitching_worker.roi_completed.connect(self._on_roi_completed)

        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Start the thread
        print("Starting stitching in background...")
        self.stitching_thread.start()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        """Enable or disable all input widgets.

        Args:
            enabled: Whether to enable or disable the inputs.
        """
        self.zarr_input.setEnabled(enabled)
        self.zarr_browse_button.setEnabled(enabled)
        self.load_button.setEnabled(enabled)
        self.algorithm_combo.setEnabled(enabled)
        self.output_format_combo.setEnabled(enabled)
        self.advanced_toggle.setEnabled(enabled)
        self.resolution_combo.setEnabled(enabled)
        self.blend_checkbox.setEnabled(enabled)
        self.stride_input.setEnabled(enabled)
        self.stitch_button.setEnabled(enabled)

    def _on_algorithm_changed(self, _index: int) -> None:
        """Handle algorithm selection change.

        Update advanced options based on selected algorithm.
        """
        algorithm = self.algorithm_combo.currentData()
        is_sofima = algorithm == "sofima"

        # Visibility: show multiview options only for multiview, stride only for SOFIMA
        # Visibility per algorithm
        self.resolution_label.setVisible(not is_sofima)
        self.resolution_combo.setVisible(not is_sofima)
        self.blend_label.setVisible(not is_sofima)
        self.blend_checkbox.setVisible(not is_sofima)

        self.stride_label.setVisible(is_sofima)
        self.stride_input.setVisible(is_sofima)

        # Enablement
        self.resolution_combo.setEnabled(not is_sofima)
        self.blend_checkbox.setEnabled(not is_sofima)
        self.stride_input.setEnabled(is_sofima)

        # Tooltips
        if not is_sofima:
            self.resolution_combo.setToolTip(
                "Select which resolution pyramid level to use for tile registration.\n"
                "Lower resolutions are faster but might be less precise.\n"
                "The output will always be at full resolution."
            )
            self.blend_checkbox.setToolTip(
                "Enable weighted average blending in overlapping regions.\n"
                "When enabled: Smooth transitions between tiles (slower).\n"
                "When disabled: Overlay fusion with sharp boundaries (faster)."
            )
        else:
            self.resolution_combo.setToolTip(
                "SOFIMA computes alignment internally; resolution selection is not used."
            )
            self.blend_checkbox.setToolTip("Blend is not used by SOFIMA.")
            self.stride_input.setToolTip(
                "Pixel stride for SOFIMA flow estimation; lower is slower but more accurate."
            )

    def _on_stitching_finished(self) -> None:
        """Re-enable inputs after stitching completes."""
        self._set_inputs_enabled(True)
        self.progress_bar.setVisible(False)

    def _on_stitching_success(self, output_paths: list) -> None:
        """Handle successful stitching completion.

        Args:
            output_paths: List of paths to the generated stitched zarr files.
        """
        print(f"\nStitching complete! Generated {len(output_paths)} file(s).")

    def _on_roi_completed(self, output_path: str) -> None:
        """Handle completion of individual ROI stitching.

        Args:
            output_path: Path to the generated stitched file (.zarr or .ome.tif).
        """
        try:
            # Track the number of layers before opening
            num_layers_before = len(self.viewer.layers)

            # Open the file - use napari-ome-zarr only for zarr files
            from pathlib import Path

            path_obj = Path(output_path)

            if path_obj.suffix == ".zarr":
                self.viewer.open(output_path, plugin="napari-ome-zarr")
            else:
                # For TIFF files, use custom reader to preserve pixel size
                self._load_ome_tiff_with_metadata(output_path)

            # Get the file name without extension
            file_name = path_obj.stem

            # Rename any newly added layers to use the file name
            for i in range(num_layers_before, len(self.viewer.layers)):
                layer = self.viewer.layers[i]
                # If there's only one new layer, use the file name directly
                # If multiple layers, append the original suffix
                if len(self.viewer.layers) - num_layers_before == 1:
                    layer.name = file_name
                else:
                    # Keep any suffix from the original name (e.g., channel info)
                    original_name = layer.name
                    layer.name = f"{file_name}_{original_name}"

            print(f"Loaded stitched ROI into napari: {output_path}")
        except Exception as e:
            print(f"Warning: Failed to load {output_path} into napari: {e}")

    def _load_ome_tiff_with_metadata(self, tiff_path: str) -> None:
        """Load OME-TIFF file with proper metadata including pixel size.

        Args:
            tiff_path: Path to the OME-TIFF file.
        """
        from tifffile import TiffFile

        with TiffFile(tiff_path) as tif:
            data = tif.asarray()
            scale = [1.0] * data.ndim

            # Parse OME-XML metadata if available
            if tif.ome_metadata:
                try:
                    from xml.etree import ElementTree as ET

                    root = ET.fromstring(tif.ome_metadata)

                    # Try common OME namespace versions
                    for year in ["2016-06", "2015-01", "2013-06"]:
                        ns = {
                            "ome": f"http://www.openmicroscopy.org/Schemas/OME/{year}"
                        }
                        pixels = root.find(".//ome:Pixels", ns)
                        if pixels is not None:
                            break

                    if pixels is not None:
                        # Get physical sizes and apply to last dimensions (Y, X)
                        if phys_x := pixels.get("PhysicalSizeX"):
                            scale[-1] = float(phys_x)
                        if phys_y := pixels.get("PhysicalSizeY"):
                            scale[-2] = float(phys_y)

                        # Handle Z if present
                        if (
                            phys_z := pixels.get("PhysicalSizeZ")
                        ) and data.ndim >= 3:
                            axes = tif.series[0].axes if tif.series else ""
                            if "Z" in axes:
                                scale[axes.index("Z")] = float(phys_z)

                except Exception as e:
                    print(f"Note: Could not parse OME-XML metadata: {e}")

            # Add layer with scale
            self.viewer.add_image(data, name=None, scale=tuple(scale))

    def _on_stitching_progress(self, current: int, total: int) -> None:
        """Update progress bar.

        Args:
            current: Number of ROIs completed.
            total: Total number of ROIs.
        """
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        # Show current ROI being processed (1-indexed)
        roi_num = current + 1 if current < total else total
        self.progress_bar.setFormat(f"Stitching ROI {roi_num}/{total}")

    def _on_stitching_error(self, error_message: str) -> None:
        """Handle stitching error.

        Args:
            error_message: The error message.
        """
        print(f"Stitching failed: {error_message}")

    def closeEvent(self, event):
        """Clean up threads when widget is closed."""
        if (
            self.stitching_thread is not None
            and self.stitching_thread.isRunning()
        ):
            self.stitching_thread.quit()
            self.stitching_thread.wait()
        super().closeEvent(event)
