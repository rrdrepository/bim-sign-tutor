"""
ui.py - Small drawing and camera helpers shared by all the scripts.
"""

import cv2
import numpy as np

from hand_tracker import HAND_CONNECTIONS

FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(index: int = 0, width: int = 640, height: int = 480):
    """Open the webcam, or exit with a helpful message."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. Is another app (Zoom, Teams, a "
            "browser tab) using it, or does Python need camera permission?"
        )
    return cap


def put_text(image, text, org, scale=0.6, colour=(255, 255, 255), thickness=1):
    """Text with a black drop shadow, so it's readable on any background."""
    x, y = org
    cv2.putText(image, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def window_closed(name: str) -> bool:
    """True if the user clicked the window's X button."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def draw_normalised_hand(image, norm_pts, top_left, size=160, colour=(0, 255, 0),
                         background=True):
    """Draw a normalised hand (from features.normalise_hand) in a small box.

    This is a learning aid: move your hand around and closer/further from the
    camera and this little hand should barely change. That's exactly what the
    model will see.
    """
    x0, y0 = top_left
    if background:     # False = draw on top of something already in the box
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), (40, 40, 40), -1)
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), (120, 120, 120), 1)
    if norm_pts is None:
        return
    # Normalised hands span roughly -2..2 palm-lengths; fit that into the box,
    # with the wrist placed near the bottom centre.
    scale = size / 4.5
    pts = [(int(x0 + size / 2 + p[0] * scale), int(y0 + size * 0.8 + p[1] * scale))
           for p in np.asarray(norm_pts)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], colour, 2)
