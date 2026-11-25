"""
features.py - Turn hand landmarks into numbers a classifier can learn from.

THE KEY IDEA OF THE WHOLE PROJECT
---------------------------------
A sign's meaning comes from the *shape* of the hand, not where the hand is
on screen or how close it is to the camera. Raw pixel coordinates mix all
three together. So before learning, we normalise:

  1. Translate: make every point relative to the wrist
     -> moving your hand around the screen no longer changes the numbers.
  2. Scale: divide by the palm size (wrist -> middle-finger knuckle)
     -> moving closer to / further from the camera no longer changes them.

What's left is (mostly) pure hand shape. That's why a simple model with a few
hundred samples per sign can work well: we've removed the variation it would
otherwise need thousands of examples to learn to ignore.

We don't normalise rotation on purpose: in sign language, which way the hand
is tilted or facing can change the meaning.

Feature layout (126 numbers per frame):
  [ right hand: 21 points x (x, y, z) = 63 ][ left hand: 63 ]
A missing hand is all zeros. Using fixed slots for each hand means one-handed
and two-handed signs fit the same format.
"""

import numpy as np

from hand_tracker import MIDDLE_MCP, WRIST

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3     # x, y, z for each point
NUM_FEATURES = FEATURES_PER_HAND * 2      # right-hand slot + left-hand slot


def normalise_hand(points3d) -> np.ndarray:
    """21 (x, y, z) points -> array of shape (21, 3), wrist-centred, palm-scaled."""
    pts = np.asarray(points3d, dtype=np.float32)
    pts = pts - pts[WRIST]                          # 1. translate
    palm_size = np.linalg.norm(pts[MIDDLE_MCP, :2]) # use x, y only: z is noisier
    if palm_size < 1e-6:                            # degenerate detection
        palm_size = 1.0
    return pts / palm_size                          # 2. scale


def slot_hands(hands) -> dict:
    """Put each detected hand in a "Right" or "Left" slot (None = no hand).

    MediaPipe sometimes labels both hands "Right" for a frame; when a slot is
    taken we use the other one, so we never silently drop a hand.
    Values are the raw (x, y, z) point lists.
    """
    slots = {"Right": None, "Left": None}
    # Most confident hands first, so they get their "true" slot.
    for hand in sorted(hands, key=lambda h: -h.score):
        preferred = hand.handedness if hand.handedness in slots else "Right"
        other = "Left" if preferred == "Right" else "Right"
        if slots[preferred] is None:
            slots[preferred] = hand.points3d
        elif slots[other] is None:
            slots[other] = hand.points3d
    return slots


def frame_features(hands) -> np.ndarray:
    """List of HandResult -> feature vector of length NUM_FEATURES."""
    slots = slot_hands(hands)
    parts = []
    for name in ("Right", "Left"):
        pts = slots[name]
        parts.append(np.zeros(FEATURES_PER_HAND, np.float32) if pts is None
                     else normalise_hand(pts).flatten())
    return np.concatenate(parts)


def feature_column_names() -> list[str]:
    """Readable column names for the CSV, e.g. 'R_p8_x' = right index tip x."""
    names = []
    for side in ("R", "L"):
        for i in range(NUM_LANDMARKS):
            for axis in ("x", "y", "z"):
                names.append(f"{side}_p{i}_{axis}")
    return names
