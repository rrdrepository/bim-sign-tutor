"""
dynamic_live.py - Stage 5: test your moving-sign model live.

    python dynamic_live.py

Press SPACE, wait for the countdown, do one sign while the border is red,
and see what the model thinks (top 3 with confidence).

We use "press, then sign" rather than watching all the time, because with
movement it's hard to know when a sign starts and ends. A fixed recording
window makes that problem disappear and keeps this stage simple.

The DynamicClassifier class is what the tutor will use for moving signs.
"""

import os
import time

import cv2
import joblib
import numpy as np

from hand_tracker import HandTracker, INDEX_TIP, draw_hand
from sequences import NUM_SEQ_FEATURES, ClipCapture, sequence_features
from ui import open_camera, put_text, window_closed

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "dynamic_model.joblib")
WINDOW = "BIM Tutor - Moving Signs"


class DynamicClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise SystemExit(f"No model at {model_path}. Run train_dynamic.py first.")
        bundle = joblib.load(model_path)
        if bundle["num_features"] != NUM_SEQ_FEATURES:
            raise SystemExit("Moving-sign model uses an old feature format. "
                             "Run train_dynamic.py again.")
        self.model = bundle["model"]
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.classes = [str(c) for c in self.model.classes_]
        self.info = f"{bundle['model_type']}, {bundle['test_accuracy']:.0%} test acc."

    def classify(self, clip: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Raw clip -> top k (label, probability)."""
        probs = self.model.predict_proba(sequence_features(clip).reshape(1, -1))[0]
        order = np.argsort(probs)[::-1][:k]
        return [(self.classes[i], float(probs[i])) for i in order]


def main() -> None:
    clf = DynamicClassifier()
    print(f"Loaded: {clf.info}  Signs: {', '.join(clf.classes)}")
    cap = open_camera()
    capture = ClipCapture()
    result: list[tuple[str, float]] = []
    trail: list[tuple[int, int]] = []

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

            if capture.state != "idle":
                clip = capture.update(hands, now)
                if capture.state == "recording" and hands:
                    trail.append(hands[0].landmarks[INDEX_TIP])
                if clip is not None:
                    result = clf.classify(clip)

            display = frame.copy()
            h, w = display.shape[:2]
            for hand in hands:
                draw_hand(display, hand.landmarks)
            for a, b in zip(trail, trail[1:]):
                cv2.line(display, a, b, (255, 0, 255), 3)

            if capture.state == "countdown":
                put_text(display, f"Get ready... {capture.countdown_left(now):.1f}",
                         (10, 40), 1.0, (0, 200, 255), 2)
            elif capture.state == "recording":
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                put_text(display, "SIGN NOW!", (10, 40), 1.0, (0, 0, 255), 2)
            elif result:
                label, p = result[0]
                good = label != "_none" and p >= 0.5
                put_text(display, label if good else "not sure", (10, 60), 1.6,
                         (0, 255, 0) if good else (150, 150, 150), 3)
                for i, (lab, prob) in enumerate(result):
                    put_text(display, f"{lab}: {prob:.0%}", (10, 100 + i * 28), 0.7)
            put_text(display, "SPACE = record a sign | ESC quit", (10, h - 15), 0.55)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_closed(WINDOW):
                break
            if key == ord(" ") and capture.state == "idle":
                capture.start(now)
                trail, result = [], []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
