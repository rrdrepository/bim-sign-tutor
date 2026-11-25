"""
hand_tracker.py - Find hands and their 21 landmarks in each webcam frame.

We use MediaPipe's *Hand Landmarker* (the "Tasks" API). It's a small pretrained
neural network: give it an image, it returns 21 3D points per hand (wrist,
knuckles, fingertips...). We don't train anything here - our own model
(Stage 2) learns from these points, not from raw pixels.

Changes from the Draw in the Air version:
  * tracks up to 2 hands (many signs use both)
  * also returns a z (depth) value per point
  * returns a list of hands instead of just one
"""

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Model file -------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)

# --- Landmark indices (see MediaPipe's hand landmark diagram) ---------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8          # MCP = the knuckle where finger meets palm
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_TIP = 16
PINKY_MCP, PINKY_TIP = 17, 20

# Pairs of landmarks to connect when drawing the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]


def ensure_model() -> None:
    """Download the hand landmark model once, if it isn't there yet."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading hand model to {MODEL_PATH} ...")
    # Download to a temp name first, then rename: if the download is cut off,
    # we don't leave a broken model file that would crash every later run.
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SystemExit(
            f"Could not download the hand model ({e}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )
    print("Done.")


@dataclass
class HandResult:
    """One detected hand."""
    landmarks: list[tuple[int, int]]            # 21 (x, y) pixel points, for drawing
    points3d: list[tuple[float, float, float]]  # 21 (x, y, z) points, for the model
    handedness: str                             # "Left" or "Right"
    score: float                                # how sure MediaPipe is about handedness


class HandTracker:
    """Thin wrapper around MediaPipe's HandLandmarker."""

    def __init__(self, max_hands: int = 2,
                 detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.5):
        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            # VIDEO mode tracks hands between frames instead of searching the
            # whole image every time -> faster and steadier.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def process(self, frame_bgr) -> list[HandResult]:
        """Return all detected hands in the frame (possibly an empty list)."""
        h, w = frame_bgr.shape[:2]

        # OpenCV gives BGR images; MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps (in milliseconds).
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        result = self._landmarker.detect_for_video(mp_image, ts)

        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            # x and y come back normalised to 0..1 of the image width/height.
            # We convert to pixels so x and y use the same unit (otherwise a
            # 640x480 frame would squash shapes). MediaPipe's z uses roughly
            # the same scale as x, so we multiply it by the width too.
            pts3d = [(p.x * w, p.y * h, p.z * w) for p in lms]
            pts2d = [(int(x), int(y)) for x, y, _ in pts3d]

            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score
            hands.append(HandResult(pts2d, pts3d, label, score))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    # Allows `with HandTracker() as tracker:` so it's always closed properly.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_hand(image, landmarks: list[tuple[int, int]],
              line_colour=(200, 200, 200), point_colour=(0, 128, 255)) -> None:
    """Draw the hand skeleton onto `image` (in place) for visual feedback."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, landmarks[a], landmarks[b], line_colour, 2)
    for x, y in landmarks:
        cv2.circle(image, (x, y), 3, point_colour, -1)
