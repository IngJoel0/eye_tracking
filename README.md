# Eye Tracking — Segmented Eyes & Pupil Logger

This repository contains a Python script that captures video from a webcam, locates each eye using
MediaPipe Face Mesh, extracts a rectangular crop for each eye, detects the pupil inside each crop
using classical OpenCV image processing, and logs the pupil coordinates to a CSV file in real time.

Key highlights:
- Real-time detection of left and right pupils with a mirrored camera view (horizontal flip).
- Side-by-side static window that shows a resized crop for each eye so both panels remain
	constant in size regardless of the subject's distance to the camera.
- Robust pupil segmentation using CLAHE, denoising, morphological operators and contour analysis.
- Per-frame CSV logging with timestamp, detection flags and global (x,y) coordinates for each eye.

Requirements
-----------
- Python 3.8+ recommended
- OpenCV
- MediaPipe
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

Usage
-----
Run the script from a terminal. Example options:

```bash
python eye_traking.py --camera 0 --tile_w 360 --tile_h 240
```

Command-line options:
- `--camera`: Camera index (0 for integrated, 1/2 for external devices).
- `--width`, `--height`: Capture resolution.
- `--tile_w`, `--tile_h`: Size of each eye panel in the static window.
- `--expand`: Extra margin (pixels) around the eye when computing the ROI.

Output
------
- A static window titled "Segmented eyes (static window) - ESC to exit" showing the
	left and right eye crops side-by-side.
- A CSV file (`eye_log.csv`) that receives one row per frame with the following columns:
	`timestamp,left_detected,right_detected,left_x,left_y,right_x,right_y`.

Notes
-----
- The script uses a mirrored view (horizontal flip) to provide a natural, mirror-like preview.
- Detection quality depends on camera resolution, lighting conditions, and the subject's head pose.
- This project is intended as a straightforward, classical-computer-vision pupil tracker
	and can be extended with calibration, smoothing, or more advanced pupil-estimation models.

If you want any adjustments (different CSV filename, additional overlays, or calibration),
let me know and I will update the code.
