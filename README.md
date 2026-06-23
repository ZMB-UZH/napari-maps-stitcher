# napari-maps-stitcher

A napari plugin to convert tiled, 2D MAPS data to OME-Zarr and to stitch regions of interest.

----------------------------------

## Installation

```
pip install git+https://github.com/ZMB-UZH/napari-maps-stitcher.git
```

If napari is not already installed, you can install `napari-maps-stitcher` with napari and Qt via:

```
pip install "napari[all]"
pip install git+https://github.com/ZMB-UZH/napari-maps-stitcher.git
```

## Instructions

The plugin contributes two widgets, found in napari under
`Plugins → Maps Stitcher`:

- **MAPS to OME-Zarr Conversion** — convert a MAPS project to OME-Zarr.
- **ROI Stitching** — stitch regions of interest from a converted OME-Zarr.

A typical workflow runs the converter first and then stitches; the converter
opens the stitching widget and loads its output automatically when it finishes.

### 1. Convert MAPS data to OME-Zarr

Open the **MAPS to OME-Zarr Conversion** widget and:

1. **Select your MAPS project folder** (the folder containing
   `MapsProject.xml`) by typing/pasting the path or using *Browse…*.
2. **Choose the layer and acquisition** to convert. The dropdowns populate from
   the project's `MapsProject.xml`. To batch-convert, pick *(All acquisitions in
   layer)* or *(All layers)* — when several acquisitions are produced, only the
   first is loaded into napari.
3. **Select an output folder** where the `.zarr` will be written.
4. Click **Convert to OME-Zarr**.

When conversion finishes, the **ROI Stitching** widget opens automatically with
the converted OME-Zarr already loaded.

### 2. Stitch regions of interest

In the **ROI Stitching** widget:

1. **Load an OME-Zarr file** — set automatically after conversion, or select one
   manually and click *Load OME-Zarr*. This adds the image and an empty
   `ROI_layer` to the viewer.
2. **Draw regions of interest** in the `ROI_layer`. Each ROI is
   stitched independently. A region covering the whole mosaic stitches
   everything.
3. (Optional) Pick a **stitching algorithm** and **output format** (OME-Zarr or
   OME-TIFF), and adjust **Advanced Options**:
   - *Multiview-Stitcher* (default): choose the registration resolution level
     (lower is faster) and whether to blend overlapping tiles.
   - *SOFIMA* (experimental): set the flow-estimation stride.
4. Click **Stitch ROIs**. Each stitched ROI is saved next to the input `.zarr`
   and loaded back into the viewer.

## Contributing

Contributions are very welcome. Please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-maps-stitcher" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ZMB-UZH/napari-maps-stitcher/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
