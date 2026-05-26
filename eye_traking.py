#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static window with both eyes segmented (rectangular crop), camera flip,
pupil detection (OpenCV) and real-time coordinates.

Requirements:
        pip install -r requirements.txt

Usage:
        python eye_traking.py --camera 0 --tile_w 360 --tile_h 240

Features:
- Uses MediaPipe Face Mesh to locate eyes.
- Extracts a rectangular crop (frame) around each eye (no black background).
- Detects the pupil inside each ROI using OpenCV and shows its coordinates
    both in the global frame and inside each panel of the static window.
- The window is always the same size (2 side-by-side panels) independent of distance.
- Applies a horizontal flip to the camera (mirror effect).
- Shows a border around each panel and the (x,y) values in real time.
- Press ESC to exit.
"""

import cv2
import numpy as np
import argparse
import time
import csv as csv_module
import os
import mediapipe as mp


# Landmark indices for the eye contours (MediaPipe FaceMesh with refine_landmarks=True)
LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
def write_csv_row(timestamp, left_detected, right_detected,
                  left_x, left_y, right_x, right_y,
                  filename='eye_log.csv'):
    """
    Append a row to a CSV file with headers:
    timestamp,left_detected,right_detected,left_x,left_y,right_x,right_y
    """
    write_header = not os.path.exists(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        w = csv_module.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "left_detected", "right_detected",
                "left_x", "left_y", "right_x", "right_y"
            ])

        # Fill missing coordinates with 0
        lx = 0 if left_x is None else left_x
        ly = 0 if left_y is None else left_y
        rx = 0 if right_x is None else right_x
        ry = 0 if right_y is None else right_y

        # Timestamp with milliseconds
        w.writerow([
            f"{timestamp:.3f}",  # seconds with 3 decimals
            1 if left_detected else 0,
            1 if right_detected else 0,
            lx, ly, rx, ry
        ])


def landmarks_to_points(landmarks, image_shape):
    h, w = image_shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def compute_eye_roi_bbox(eye_points, frame_shape, expand=5):
    """Devuelve una caja (x0,y0,x1,y1) alrededor del ojo con margen 'expand' de píxeles."""
    if not eye_points:
        return None
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    x0, y0 = max(min(xs) - expand, 0), max(min(ys) - expand, 0)
    x1, y1 = min(max(xs) + expand, frame_shape[1]-1), min(max(ys) + expand, frame_shape[0]-1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def detect_pupil(roi_bgr, roi_mask=None):
    """
    Detecta la pupila en el ROI BGR (opcionalmente restringido por roi_mask).
    Devuelve:
      - centro_roi: (cx, cy) en coordenadas del ROI o None
      - radio: radio estimado de la pupila o None
    """
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    #roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    if roi_mask is None:
        roi_mask = np.ones(roi_gray.shape, dtype=np.uint8) * 255

    masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)

    blurred = cv2.GaussianBlur(masked, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    _, bin_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=roi_mask)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 10:
        return None, None

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    return (int(cx), int(cy)), int(radius)


def build_eye_mask(frame_shape, eye_points):
    """Polygonal mask of the eye to improve detection (not for visualization)."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if len(eye_points) >= 3:
        cv2.fillConvexPoly(mask, np.array(eye_points, dtype=np.int32), 255)
    return mask

t0 = time.time()   # <-- tiempo inicial (referencia)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0=internal; 1,2=external)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--tile_w", type=int, default=360, help="Width of each eye panel in the window")
    ap.add_argument("--tile_h", type=int, default=240, help="Height of each eye panel in the window")
    ap.add_argument("--expand", type=int, default=6, help="Extra margin around the eye (px)")
    args = ap.parse_args()

    #cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Verify the camera index with --camera.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Static window (two side-by-side panels)
    canvas_w = args.tile_w * 2
    canvas_h = args.tile_h
    window_name = "Segmented eyes (static window) - ESC to exit"

    last_print = 0.0

    # Ensure output folder exists and prepare dated CSV filename (one file per run)
    os.makedirs("coords_ET", exist_ok=True)
    csv_filename = os.path.join("coords_ET", f"ET_{time.strftime('%d_%m_%y')}.csv")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            if res.multi_face_landmarks:
                face_landmarks = res.multi_face_landmarks[0].landmark
                all_pts = landmarks_to_points(face_landmarks, frame.shape)

                left_eye_pts = [all_pts[i] for i in LEFT_EYE_IDX if i < len(all_pts)]
                right_eye_pts = [all_pts[i] for i in RIGHT_EYE_IDX if i < len(all_pts)]

                left_mask_global = build_eye_mask(frame.shape, left_eye_pts)
                right_mask_global = build_eye_mask(frame.shape, right_eye_pts)

                left_bbox = compute_eye_roi_bbox(left_eye_pts, frame.shape, expand=args.expand)
                right_bbox = compute_eye_roi_bbox(right_eye_pts, frame.shape, expand=args.expand)

                # --- ACUMULADORES POR FRAME (inicializa en "no detectado") ---
                left_detected = False
                right_detected = False
                left_gx = left_gy = None
                right_gx = right_gy = None
                # -------------------------------------------------------------

                for side, bbox, mask_global, tile_x0 in (
                    ("Left", left_bbox, left_mask_global, 0),
                    ("Right", right_bbox, right_mask_global, args.tile_w),
                ):
                    if bbox is None:
                        cv2.putText(canvas, f"{side}: not detected", (tile_x0 + 10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(canvas, (tile_x0, 0), (tile_x0 + args.tile_w - 1, args.tile_h - 1), (80, 80, 80), 2)
                        continue

                    x0, y0, x1, y1 = bbox
                    roi = frame[y0:y1, x0:x1]
                    h_roi, w_roi = roi.shape[:2]
                    mask_roi = mask_global[y0:y1, x0:x1]

                    pupil_center_roi, pupil_radius = detect_pupil(roi, mask_roi)

                    roi_resized = cv2.resize(roi, (args.tile_w, args.tile_h), interpolation=cv2.INTER_CUBIC)
                    canvas[0:args.tile_h, tile_x0:tile_x0 + args.tile_w] = roi_resized
                    cv2.rectangle(canvas, (tile_x0, 0), (tile_x0 + args.tile_w - 1, args.tile_h - 1), (0, 255, 0), 2)

                    if pupil_center_roi is not None and pupil_radius is not None:
                        sx = args.tile_w / float(w_roi)
                        sy = args.tile_h / float(h_roi)
                        cx_panel = int(pupil_center_roi[0] * sx) + tile_x0
                        cy_panel = int(pupil_center_roi[1] * sy)
                        r_panel = max(2, int(pupil_radius * (sx + sy) / 2.0))
                        # cv2.circle(canvas, (cx_panel, cy_panel), r_panel, (0, 255, 255), 2)
                        cv2.circle(canvas, (cx_panel, cy_panel), 2, (0, 0, 255), -1)

                        gx = x0 + pupil_center_roi[0]
                        gy = y0 + pupil_center_roi[1]
                        # cv2.putText(canvas, f"{side}: ({gx}, {gy})", (tile_x0 + 10, args.tile_h - 15),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                        # --- STORE IN ACCUMULATORS BY SIDE ---
                        if side == "Left":
                            left_detected = True
                            left_gx, left_gy = gx, gy
                        else:  # "Right"
                            right_detected = True
                            right_gx, right_gy = gx, gy
                        # --------------------------------------

                        now = time.time()
                        if now - last_print > 0.2:
                            print(f"{side} -> x:{gx}  y:{gy}")
                            last_print = now
                    else:
                        cv2.putText(canvas, f"{side}: no pupil", (tile_x0 + 10, args.tile_h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # === call CSV once per frame ===
                write_csv_row(
                    timestamp=time.time() - t0,   # relative time since start
                    left_detected=left_detected,
                    right_detected=right_detected,
                    left_x=left_gx, left_y=left_gy,
                    right_x=right_gx, right_y=right_gy,
                    filename=csv_filename
                )
                # =================================

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break


    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
