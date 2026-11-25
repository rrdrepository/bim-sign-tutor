"""
sequences.py - Stage 5: turn a short clip of hand movement into features.

Single letters (Stages 1-4) are one frame: a handshape. Signs that MOVE -
J, Z, and most word signs - are a handshape *changing over time* plus a
*path* through the air. So a sample is now a clip of ~1.5 seconds.

What we save when recording (raw, so we can change the features later
without re-recording):
    array of shape (frames, 2, 21, 3)
                     |      |   |   +- x, y, z (pixels; z on the x scale)
                     |      |   +----- 21 landmarks
                     |      +--------- slot 0 = Right hand, slot 1 = Left hand
                     +---------------- however many frames the camera gave us
    Missing hands are NaN ("not a number").

What the model sees (sequence_features):
    1. Fill gaps: if a hand drops out for a frame or two, reuse its last
       position instead of a hole.
    2. Resample to SEQ_LEN frames: a sign done fast and the same sign done
       slowly line up frame by frame.
    3. Per frame and hand: the normalised handshape (same as features.py).
    4. Per frame and hand: the wrist's movement since the START of the clip,
       divided by palm size -> the path, at any distance from the camera.
    5. The gap between the two wrists (for two-handed signs).
"""

import time

import numpy as np

from features import FEATURES_PER_HAND, normalise_hand, slot_hands
from hand_tracker import MIDDLE_MCP, WRIST

SEQ_LEN = 20                    # frames after resampling
RECORD_SECONDS = 1.5            # length of each captured clip
PER_FRAME = FEATURES_PER_HAND * 2 + 3 + 3 + 3   # shapes + 2 paths + wrist gap = 135
NUM_SEQ_FEATURES = SEQ_LEN * PER_FRAME


# --- Raw capture ---------------------------------------------------------------
def raw_frame(hands) -> np.ndarray:
    """One frame of hands -> array (2, 21, 3), NaN where a hand is missing."""
    out = np.full((2, 21, 3), np.nan, dtype=np.float32)
    slots = slot_hands(hands)
    for i, name in enumerate(("Right", "Left")):
        if slots[name] is not None:
            out[i] = np.asarray(slots[name], dtype=np.float32)
    return out


# --- Feature extraction ------------------------------------------------------------
def _fill_gaps(seq: np.ndarray) -> np.ndarray:
    """Forward-fill, then back-fill, missing frames for each hand slot."""
    seq = seq.copy()
    for s in range(2):
        present = ~np.isnan(seq[:, s, 0, 0])
        if not present.any():
            continue                                 # this hand never appeared
        last = None
        for t in range(len(seq)):
            if present[t]:
                last = seq[t, s]
            elif last is not None:
                seq[t, s] = last
        first = np.argmax(present)
        seq[:first, s] = seq[first, s]
    return seq


def _resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Stretch or squeeze the clip in time to exactly n frames (linear interpolation)."""
    old_t = np.linspace(0, 1, len(seq))
    new_t = np.linspace(0, 1, n)
    flat = seq.reshape(len(seq), -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new_t, old_t, flat[:, j])
    return out.reshape((n,) + seq.shape[1:])


def sequence_features(raw: np.ndarray) -> np.ndarray:
    """Raw clip (frames, 2, 21, 3) -> flat feature vector of NUM_SEQ_FEATURES."""
    if len(raw) < 2:
        return np.zeros(NUM_SEQ_FEATURES, np.float32)
    seq = _fill_gaps(raw)
    has_hand = [not np.isnan(seq[0, s, 0, 0]) for s in range(2)]
    seq = np.nan_to_num(seq)                         # never-seen hand -> zeros
    seq = _resample(seq, SEQ_LEN)

    # One palm size for the whole clip (median = ignores odd frames).
    palms = [np.linalg.norm(seq[t, s, MIDDLE_MCP, :2] - seq[t, s, WRIST, :2])
             for t in range(SEQ_LEN) for s in range(2) if has_hand[s]]
    palm = float(np.median(palms)) if palms else 1.0
    palm = palm if palm > 1e-6 else 1.0

    frames = []
    for t in range(SEQ_LEN):
        parts = []
        for s in range(2):                           # 3. handshapes
            parts.append(normalise_hand(seq[t, s]).flatten() if has_hand[s]
                         else np.zeros(FEATURES_PER_HAND, np.float32))
        for s in range(2):                           # 4. paths
            parts.append((seq[t, s, WRIST] - seq[0, s, WRIST]) / palm if has_hand[s]
                         else np.zeros(3, np.float32))
        if all(has_hand):                            # 5. gap between wrists
            parts.append((seq[t, 0, WRIST] - seq[t, 1, WRIST]) / palm)
        else:
            parts.append(np.zeros(3, np.float32))
        frames.append(np.concatenate(parts))
    return np.concatenate(frames).astype(np.float32)


# --- Data augmentation (training only) -----------------------------------------
def augment(raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A slightly altered copy of a recorded clip.

    You'll record maybe 20-30 clips per sign - small for a model with 2700
    inputs. Augmentation makes realistic variations so the model learns what
    matters (the movement) and ignores what doesn't:
      * trim a little off the start/end  -> you started a bit early or late
      * rotate a few degrees              -> camera or body slightly tilted
      * small random jitter               -> tracking noise
    """
    n = len(raw)
    start = rng.integers(0, max(1, int(n * 0.1)) + 1)
    end = n - rng.integers(0, max(1, int(n * 0.1)) + 1)
    out = raw[start:max(end, start + 2)].copy()

    angle = np.radians(rng.uniform(-8, 8))
    c, s = np.cos(angle), np.sin(angle)
    valid = ~np.isnan(out[..., 0])
    centre = np.nanmean(out[..., :2][valid], axis=0) if valid.any() else np.zeros(2)
    xy = out[..., :2] - centre
    out[..., 0] = xy[..., 0] * c - xy[..., 1] * s + centre[0]
    out[..., 1] = xy[..., 0] * s + xy[..., 1] * c + centre[1]
    out += rng.normal(0, 1.0, out.shape).astype(np.float32)   # NaN stays NaN
    return out


# --- Countdown + record helper (used by the recorder, live demo and tutor) ------
class ClipCapture:
    """A tiny state machine: idle -> countdown -> recording -> done.

    Call start(), then update(hands) every frame. When update() returns a
    clip (numpy array), the recording has finished.
    """

    def __init__(self, countdown_s: float = 1.5, record_s: float = RECORD_SECONDS):
        self.countdown_s, self.record_s = countdown_s, record_s
        self.state = "idle"
        self.frames: list[np.ndarray] = []
        self.t0 = 0.0

    def start(self, now: float | None = None) -> None:
        self.state, self.frames = "countdown", []
        self.t0 = time.monotonic() if now is None else now

    def update(self, hands, now: float | None = None):
        now = time.monotonic() if now is None else now
        if self.state == "countdown" and now - self.t0 >= self.countdown_s:
            self.state, self.t0 = "recording", now
        if self.state == "recording":
            self.frames.append(raw_frame(hands))
            if now - self.t0 >= self.record_s:
                self.state = "idle"
                return np.stack(self.frames)
        return None

    def countdown_left(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, self.countdown_s - (now - self.t0))

    def progress(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return min(1.0, (now - self.t0) / self.record_s) if self.state == "recording" else 0.0
