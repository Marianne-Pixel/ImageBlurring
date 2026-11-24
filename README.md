# Person Anonymization (Equirectangular Panos, Fisheye & Regular Images)

Tool for automatically obscuring people in images using YOLO11
segmentation and a selectable blur effect (Gaussian blur or pixelation).
Supports regular, barely distorted photos, equirectangular panoramas and
fisheye images (circular image area). By default, the processed images are
saved to a freely selectable location and the originals remain unchanged.
Optionally, the originals can be overwritten directly.

## Installation

Python 3.12 recommended.

```bash
cd personen_anonymisierer
pip install -r requirements.txt
```

The tool automatically selects the fastest available compute backend: an
NVIDIA GPU with CUDA (half precision), otherwise OpenVINO on the CPU (about
2.5 times faster than PyTorch, package `openvino` in `requirements.txt`),
otherwise PyTorch on the CPU. On the very first start without a GPU, the
model is converted once for this purpose, which takes a few seconds.

To use an NVIDIA GPU, PyTorch with CUDA must be installed (on Windows, `pip`
installs the CPU version by default), see
[ANLEITUNG.md](ANLEITUNG.md#nvidia-gpu-nutzen).

## Start

```bash
python main.py
```

As an exe: `pyinstaller ImageBlurring_YOLO11.spec` creates the folder
`dist\ImageBlurring_YOLO11\` (distribute the entire folder). For details see
[ANLEITUNG.md](ANLEITUNG.md#exe-erstellen-pyinstaller).

## Usage

On small screens, simply shrink the window: scroll bars appear (mouse wheel
works too), and the "Start" button always stays visible at the bottom.

1. Click **Durchsuchen…** (next to "Ordner") and select the folder containing the
   source images. The folder is searched (including subfolders) for `.jpg`,
   `.jpeg` and `.png` files.
2. Click **Durchsuchen…** (next to "Speicherort") and select the folder where
   the processed images should be saved. The subfolder structure of the
   source folder is recreated at the output location. Alternatively, enable
   the checkbox **"Bilder im Quellordner ersetzen"** (default: off): the
   original images are then replaced and no output location is needed.
3. Select the desired images in the list (multiple selection with
   `Ctrl`/`Shift` click, or "Alle auswählen"). The "Sortieren nach" option orders the
   list by name or image size. When a single image is selected, a small
   preview appears next to the list.
4. Under **"Verwisch-Einstellungen"**, choose the method (**Gaussian Blur** or
   **Pixelieren/Mosaik**) and set the strength with the slider. The preview on
   the right shows the effect live on a small, fictitious sample image.
5. Under **"Belichtungskorrektur"**, enable the correction and set the two
   independent sliders **"Schatten aufhellen"** and **"Lichter abdunkeln"**
   (0–100 each): 0 = no effect, 100 = maximum effect. The sliders act
   independently of each other.
6. Under **"Bildtyp"**, choose **Equirectangular** (default),
   **Fisheye** or **Normale Perspektive** via radio button.
7. Under **"Metadaten (nur JPEG)"**, specify whether EXIF/XMP/color profile
   are preserved (default: yes, required for 360° display) and whether the
   GPS position should be removed (default: no).
8. Click **Start**. A confirmation prompt appears showing the number of
   selected images ("Es werden X Bild(er) bearbeitet."), since files with the
   same name at the output location will be overwritten ("Ja" = start,
   "Nein"/"Abbrechen" = abort).
9. Progress is shown at the bottom, followed by a time estimate (elapsed
   time, remaining time, expected finish); at the end a summary appears
   (blurred areas/images, number of brightened/darkened images, including
   any errors per image).

## How It Works

### Person Detection
Both image types use the same YOLO11 **segmentation** (not just a bounding
box) via `PersonDetector.detect_masks_normal()` in `detector.py`. This
provides the actual silhouette of each detected person, so that only this
irregular shape is blurred — not a rectangle or oval that would needlessly
blur a lot of background as well.

### Regular Images
`processing.build_mask_from_polygons()` creates a mask from the detected
silhouettes with only a few pixels of safety margin;
`processing.blur_mask()` applies the selected method (**Gaussian blur** or
**Pixelate/Mosaic**, with strength adjustable in the GUI) to it, with a soft
transition at the edge. By default, the strength is adapted to the size of
each person, so that even close, large people are reliably made
unrecognizable.

### Equirectangular Images (Theta Z1)
As before: split the panorama into overlapping tiles, segment each tile,
project the silhouettes back into the panorama coordinate system, then blur
using the resulting mask.

### Fisheye Images
Analogous to panoramas: cut the image circle into 9 overlapping, undistorted
tiles, segment all tiles at 0°/90°, and additionally segment tiles with hits
at 180°/270° (for cameras mounted rotated), and project the silhouettes
exactly back into the fisheye image. The field of view (default 200°) and
the projection function (default equidistant) can be adjusted at the top of
`detector.py`.

### Exposure Correction
Tone curve on the L channel (lightness) in the LAB color space: shadows are
lifted, very bright areas are compressed. Two **independent** sliders in the
GUI — **"Schatten aufhellen"** and **"Lichter abdunkeln"** (0–100 each) —
control the two effects separately. A safety check prevents the image from
becoming too bright overall as a result.

### Final Statistics
After processing, the GUI shows a summary: number of blurred areas and
number of affected images, as well as (if enabled) how many images were
brightened or darkened.

For details see [ANLEITUNG.md](ANLEITUNG.md).

### Metadata
For JPEG, EXIF, XMP (including the 360° information of the Theta Z1) and the
color profile are preserved. Embedded thumbnails are always removed, as they
would show the people unblurred. Optionally, the GPS position can be
removed.

The image geometry remains unchanged: portrait photos that the camera stores
in landscape orientation with the EXIF tag "Orientation" are also saved in
landscape with the same tag. This way, programs such as RiSCAN get the same
geometry as with the original.

## Note

By default, the original files are **not modified**. The processed images
are saved under the selected output location (with the same relative
subfolder structure as in the source folder). If files with the same name
already exist there, they will be overwritten.

With **"Bilder im Quellordner ersetzen"**, the originals are replaced
(cannot be undone). Each image is first written to a temporary file and
replaces the original only after it has been saved without errors. Images
without a detected person remain untouched.
