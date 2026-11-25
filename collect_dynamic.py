"""
collect_dynamic.py - Stage 5: record clips of signs that move.

    python collect_dynamic.py

Each SPACE press records a ROUND of 5 clips automatically:
    "get ready" countdown -> RECORD (1.5 s, red border) -> short rest -> next
Do the whole sign inside the red-border time. Between clips, drop your
hands or relax, so every clip starts the same way.

Clips are saved one file each in data/sequences/<SIGN>/ as raw hand points.

Controls (click the video window first):
    1-9        choose a sign from SIGNS below
    0          "_none": not signing - still hands, scratching your face,
               random movement. Teaches the model when NOT to guess.
    SPACE      record a round of 5 clips
    BACKSPACE  delete the last clip
    ESC        quit (or the window's X)

Edit SIGNS to choose your own signs (up to 9). Use the BIM signs you are
learning, checked against a trusted source. Names must be valid folder names.
"""

import os
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, INDEX_TIP, draw_hand
from sequences import ClipCapture
from ui import open_camera, put_text, window_closed

SIGNS = ["J", "Z", "TERIMA_KASIH", "MAAF", "TOLONG"]   # edit me (max 9)
NONE_LABEL = "_none"
CLIPS_PER_ROUND = 5
REST_S = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(HERE, "data", "sequences")
WINDOW = "BIM Tutor - Record Moving Signs"


def count_clips() -> dict[str, int]:
    counts = {}
    for label in SIGNS + [NONE_LABEL]:
        folder = os.path.join(SEQ_DIR, label)
        counts[label] = len([f for f in os.listdir(folder) if f.endswith(".npy")]) \
            if os.path.isdir(folder) else 0
    return counts


def save_clip(label: str, clip: np.ndarray) -> str:
    folder = os.path.join(SEQ_DIR, label)
    os.makedirs(folder, exist_ok=True)
    # Timestamp names sort in recording order (train_dynamic.py relies on that).
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.npy"
    path = os.path.join(folder, name)
    np.save(path, clip.astype(np.float32))
    return path


def main() -> None:
    cap = open_camera()
    capture = ClipCapture()
    label = None
    saved_paths: list[str] = []          # for undo
    clips_left = 0                       # clips remaining in the current round
    rest_until = 0.0
    trail: list[tuple[int, int]] = []    # fingertip path drawn while recording
    counts = count_clips()
    message, message_until = "", 0.0

    cv2.namedWindow(WINDOW)
    with HandTracker(max_hands=2) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            now = time.monotonic()

            # --- Round logic ---------------------------------------------------
            if clips_left and capture.state == "idle" and now >= rest_until:
                capture.start(now)
                trail = []
            clip = capture.update(hands, now) if capture.state != "idle" else None
            if capture.state == "recording" and hands:
                trail.append(hands[0].landmarks[INDEX_TIP])
            if clip is not None:
                valid = (~np.isnan(clip[:, :, 0, 0])).any(axis=1).mean()
                if valid < 0.5:
                    message, message_until = "No hands seen in that clip - not saved", now + 2
                else:
                    saved_paths.append(save_clip(label, clip))
                    counts = count_clips()
                clips_left -= 1
                rest_until = now + REST_S
                if clips_left == 0:
                    message, message_until = f"Round done: {counts[label]} clips of {label}", now + 2

            # --- Draw ------------------------------------------------------------
            display = frame.copy()
            h, w = display.shape[:2]
            for hand in hands:
                draw_hand(display, hand.landmarks)
            for a, b in zip(trail, trail[1:]):
                cv2.line(display, a, b, (255, 0, 255), 3)

            put_text(display, f"Sign: {label or '(press 1-9 or 0)'}", (10, 30), 0.8, (255, 255, 0), 2)
            if capture.state == "countdown":
                put_text(display, f"Get ready... {capture.countdown_left(now):.1f}   "
                                  f"(clip {CLIPS_PER_ROUND - clips_left + 1}/{CLIPS_PER_ROUND})",
                         (10, 65), 0.7, (0, 200, 255), 2)
            elif capture.state == "recording":
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                put_text(display, "SIGN NOW!", (10, 65), 0.9, (0, 0, 255), 2)
                cv2.rectangle(display, (10, 80), (10 + int(300 * capture.progress(now)), 92),
                              (0, 0, 255), -1)
            elif clips_left:
                put_text(display, "Rest - relax your hands", (10, 65), 0.7, (200, 200, 200), 2)
            elif now < message_until:
                put_text(display, message, (10, 65), 0.7, (0, 255, 0), 2)
            else:
                put_text(display, "1-9/0 sign | SPACE record 5 | BACKSPACE undo | ESC quit",
                         (10, 65), 0.5)

            # Sign list with clip counts (green = 25+, a good amount to start)
            for i, name in enumerate(SIGNS + [NONE_LABEL]):
                key = "0" if name == NONE_LABEL else str(i + 1)
                n = counts[name]
                colour = (255, 255, 0) if name == label else (0, 200, 0) if n >= 25 \
                    else (0, 200, 255) if n else (150, 150, 150)
                put_text(display, f"{key}  {name}: {n}", (w - 230, 30 + i * 24), 0.5, colour)
            cv2.imshow(WINDOW, display)

            # --- Keyboard ----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_closed(WINDOW):
                break
            if clips_left or capture.state != "idle":
                continue                                       # busy recording
            if ord("1") <= key <= ord("9") and key - ord("1") < len(SIGNS):
                label = SIGNS[key - ord("1")]
            elif key == ord("0"):
                label = NONE_LABEL
            elif key == ord(" "):
                if label is None:
                    message, message_until = "Choose a sign first (1-9 or 0)", now + 2
                else:
                    clips_left, rest_until = CLIPS_PER_ROUND, now
            elif key == 8 and saved_paths:                     # BACKSPACE
                path = saved_paths.pop()
                os.remove(path)
                counts = count_clips()
                message, message_until = f"Deleted last clip of {os.path.basename(os.path.dirname(path))}", now + 2

    cap.release()
    cv2.destroyAllWindows()
    print("Clips per sign:")
    for name, n in count_clips().items():
        print(f"  {name:>14}: {n}")


if __name__ == "__main__":
    main()
