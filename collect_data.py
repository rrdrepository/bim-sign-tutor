"""
collect_data.py - Stage 1: record your own training samples.

How it works: pick a letter, make its handshape, press SPACE. After a short
countdown the script records a burst of samples (one every couple of frames)
and saves them to data/samples.csv. Each sample is one row: the label plus
126 normalised landmark numbers (see features.py).

Controls (click the video window first so it receives your key presses):
    A-Z        choose which letter you're recording
    0          choose "_none": relaxed hand, hands moving between signs,
               random gestures. This teaches the model to say "that's not
               a letter" instead of always guessing one.
    SPACE      record a burst of samples for the chosen label
    BACKSPACE  undo the last burst (if you made a mistake)
    ESC        save and quit (also: the window's X button)

Note: Q is a letter now, so quitting is ESC.
"""

import csv
import os
import time

import cv2
import numpy as np

from features import NUM_FEATURES, feature_column_names, frame_features, normalise_hand
from hand_tracker import HandTracker, draw_hand
from ui import draw_normalised_hand, open_camera, put_text, window_closed

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "samples.csv")
WINDOW = "BIM Tutor - Data Collection"

SAMPLES_PER_BURST = 60    # samples recorded per SPACE press
FRAME_STEP = 2            # keep every 2nd frame -> more variety, less duplication
COUNTDOWN_S = 1.5         # time to get your hand into position
NONE_LABEL = "_none"


# --- Saving and loading ------------------------------------------------------
def load_samples() -> list[list]:
    """Read existing rows so new sessions add to old data instead of replacing it."""
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                      # skip header
        return [row for row in reader if row]


def save_samples(rows: list[list]) -> None:
    """Rewrite the whole CSV. Small files (a few MB) so this is instant.
    Writing to a temp file then renaming means a crash can't corrupt the data."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    tmp = DATA_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "num_hands"] + feature_column_names())
        writer.writerows(rows)
    os.replace(tmp, DATA_PATH)


def count_labels(rows) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


# --- Drawing the interface ---------------------------------------------------
def draw_label_summary(image, counts: dict[str, int], current: str | None) -> None:
    """Bottom bar: every letter with its sample count. Green = enough data."""
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, h - 52), (w, h), (30, 30, 30), -1)
    labels = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [NONE_LABEL]
    cell = w / 14                                  # 2 rows of 14 cells
    for i, lab in enumerate(labels):
        n = counts.get(lab, 0)
        colour = (0, 200, 0) if n >= 200 else (0, 200, 255) if n > 0 else (130, 130, 130)
        if lab == current:
            colour = (255, 255, 0)
        x = int((i % 14) * cell) + 4
        y = h - 32 if i < 14 else h - 8
        short = "--" if lab == NONE_LABEL else lab
        cv2.putText(image, f"{short}:{n}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, colour, 1, cv2.LINE_AA)


def main() -> None:
    rows = load_samples()
    counts = count_labels(rows)
    print(f"Loaded {len(rows)} existing samples from {DATA_PATH}")

    cap = open_camera()
    current_label: str | None = None
    burst_sizes: list[int] = []        # for undo
    state = "idle"                     # idle -> countdown -> recording -> idle
    countdown_end = 0.0
    burst_rows: list[list] = []
    frame_no = 0
    message, message_until = "", 0.0

    cv2.namedWindow(WINDOW)
    with HandTracker(max_hands=2) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break
            # Mirror so it behaves like a mirror. MediaPipe's Left/Right labels
            # assume a mirrored image, so this also keeps handedness correct.
            frame = cv2.flip(frame, 1)
            frame_no += 1

            hands = tracker.process(frame)
            now = time.monotonic()

            # --- Recording state machine -----------------------------------
            if state == "countdown" and now >= countdown_end:
                state, burst_rows = "recording", []
            if state == "recording" and hands and frame_no % FRAME_STEP == 0:
                feats = frame_features(hands)
                burst_rows.append([current_label, len(hands)]
                                  + [f"{v:.5f}" for v in feats])
                if len(burst_rows) >= SAMPLES_PER_BURST:
                    rows.extend(burst_rows)
                    burst_sizes.append(len(burst_rows))
                    counts = count_labels(rows)
                    save_samples(rows)               # save after every burst
                    state = "idle"
                    message, message_until = f"Saved {len(burst_rows)} x {current_label}", now + 2

            # --- Draw ----------------------------------------------------------
            display = frame.copy()
            for hand in hands:
                draw_hand(display, hand.landmarks)
                put_text(display, hand.handedness, (hand.landmarks[0][0] - 20,
                                                    hand.landmarks[0][1] + 25), 0.5)

            label_text = current_label or "(press a letter)"
            put_text(display, f"Label: {label_text}", (10, 30), 0.8, (255, 255, 0), 2)
            put_text(display, f"Hands: {len(hands)}   Samples for label: "
                              f"{counts.get(current_label, 0)}", (10, 58))

            if state == "countdown":
                put_text(display, f"Get ready... {countdown_end - now:.1f}",
                         (10, 90), 0.8, (0, 200, 255), 2)
            elif state == "recording":
                put_text(display, f"RECORDING {len(burst_rows)}/{SAMPLES_PER_BURST}"
                                  " - vary angle & distance slightly",
                         (10, 90), 0.6, (0, 0, 255), 2)
                if not hands:
                    put_text(display, "No hand visible - paused", (10, 115), 0.6, (0, 0, 255))
            elif now < message_until:
                put_text(display, message, (10, 90), 0.7, (0, 255, 0), 2)
            else:
                put_text(display, "A-Z/0 label | SPACE record | BACKSPACE undo | ESC quit",
                         (10, 90), 0.5)

            # The model's-eye view of the first hand (top-right corner).
            if hands:
                norm = normalise_hand(hands[0].points3d)
                draw_normalised_hand(display, norm, (display.shape[1] - 170, 10))
                put_text(display, "model view", (display.shape[1] - 160, 185), 0.45)

            draw_label_summary(display, counts, current_label)
            cv2.imshow(WINDOW, display)

            # --- Keyboard --------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_closed(WINDOW):           # ESC or X button
                break
            if state != "idle":
                continue                                     # ignore keys mid-burst
            if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                current_label = chr(key).upper()
            elif key == ord("0"):
                current_label = NONE_LABEL
            elif key == ord(" "):
                if current_label is None:
                    message, message_until = "Choose a label first (A-Z or 0)", now + 2
                else:
                    state, countdown_end = "countdown", now + COUNTDOWN_S
            elif key == 8 and burst_sizes:                   # BACKSPACE = undo
                n = burst_sizes.pop()
                undone = rows[-n][0]
                del rows[-n:]
                counts = count_labels(rows)
                save_samples(rows)
                message, message_until = f"Undid last burst ({n} x {undone})", now + 2

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(rows)} samples to {DATA_PATH}")
    for label in sorted(counts):
        print(f"  {label:>6}: {counts[label]}")


if __name__ == "__main__":
    main()
