"""
recognizer.py - Stage 3: recognise signs live from the webcam.

    python recognizer.py

Two parts:
  * Recognizer   - loads the trained model and turns hands into a STABLE
                   prediction. The tutor game (Stage 4) will reuse this class.
  * main()       - a live demo: fingerspell and watch the letters appear as text.

WHY "STABLE"?
The model predicts every frame (~30 times a second). While your hand moves
from one sign to the next it passes through shapes that briefly look like
other letters, so raw predictions flicker: A A A S A N N N... We only accept a
letter when it wins most of the recent frames AND the model is confident.
That's the same idea as "debouncing" a physical button.

Demo controls (click the video window first):
    SPACE      add a space          BACKSPACE  delete last character
    C          clear text           ESC        quit (or the window's X)
"""

import os
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import joblib
import numpy as np

from features import NUM_FEATURES, frame_features
from hand_tracker import HandTracker, draw_hand
from ui import open_camera, put_text, window_closed

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "sign_model.joblib")
NONE_LABEL = "_none"
WINDOW = "BIM Tutor - Live Recognition"


@dataclass
class Prediction:
    top: list[tuple[str, float]]   # best 3 (label, probability) for THIS frame
    stable: str | None             # letter that has been held steadily, or None


class Recognizer:
    def __init__(self, model_path: str = MODEL_PATH,
                 window: int = 10, min_votes: int = 7, min_confidence: float = 0.6):
        """
        window          how many recent frames to look at (10 frames ~ 1/3 s)
        min_votes       how many of those must agree on the same letter
        min_confidence  average model probability needed for that letter
        Raise min_votes / min_confidence if wrong letters slip through;
        lower them if correct letters take too long to register.
        """
        if not os.path.exists(model_path):
            raise SystemExit(f"No model at {model_path}. Run train_model.py first.")
        bundle = joblib.load(model_path)
        if bundle["num_features"] != NUM_FEATURES:
            raise SystemExit("Model was trained with a different feature format. "
                             "Run train_model.py again.")
        self.model = bundle["model"]
        # A random forest normally uses all CPU cores, but starting those
        # threads for a single frame costs more time than it saves.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.classes = [str(c) for c in self.model.classes_]
        self.model_info = f"{bundle['model_type']}, {bundle['test_accuracy']:.0%} test acc."

        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self._history: deque[tuple[str, float]] = deque(maxlen=window)

    def update(self, hands) -> Prediction:
        """Feed one frame's hands; get this frame's top 3 + the stable letter."""
        if not hands:
            # No hand = definitely not signing. Counts as a "none" vote so a
            # letter can't stay stuck on screen after you drop your hand.
            self._history.append((NONE_LABEL, 1.0))
            return Prediction([], None)

        feats = frame_features(hands).reshape(1, -1)    # model expects a 2D table
        probs = self.model.predict_proba(feats)[0]
        order = np.argsort(probs)[::-1][:3]
        top = [(self.classes[i], float(probs[i])) for i in order]
        self._history.append(top[0])
        return Prediction(top, self._stable_label())

    def _stable_label(self) -> str | None:
        votes = Counter(label for label, _ in self._history)
        label, n = votes.most_common(1)[0]
        if label == NONE_LABEL or n < self.min_votes:
            return None
        avg_conf = np.mean([p for lab, p in self._history if lab == label])
        return label if avg_conf >= self.min_confidence else None

    def reset(self) -> None:
        self._history.clear()


# --- Live demo ---------------------------------------------------------------------
def draw_top3(image, top, x, y):
    """Bars showing the model's top 3 guesses for the current frame."""
    for i, (label, p) in enumerate(top):
        yy = y + i * 28
        name = "none" if label == NONE_LABEL else label
        cv2.rectangle(image, (x + 55, yy - 16), (x + 55 + 150, yy + 2), (60, 60, 60), -1)
        cv2.rectangle(image, (x + 55, yy - 16), (x + 55 + int(150 * p), yy + 2),
                      (0, 200, 0) if i == 0 else (0, 140, 255), -1)
        put_text(image, f"{name}", (x, yy), 0.6)
        put_text(image, f"{p:.0%}", (x + 212, yy), 0.5)


def main() -> None:
    recognizer = Recognizer()
    print(f"Loaded model: {recognizer.model_info}")
    cap = open_camera()
    text = ""
    last_added: str | None = None     # prevents one held sign typing "AAAA"

    cv2.namedWindow(WINDOW)
    with HandTracker(max_hands=2) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            pred = recognizer.update(hands)

            # Type a letter when it becomes stable. To type the same letter
            # twice (e.g. "OO"), relax your hand or drop it briefly in between.
            if pred.stable and pred.stable != last_added:
                text += pred.stable
                last_added = pred.stable
            elif pred.stable is None and not (pred.top and pred.top[0][0] == last_added):
                last_added = None

            # --- Draw ------------------------------------------------------
            display = frame.copy()
            for hand in hands:
                draw_hand(display, hand.landmarks)
            h, w = display.shape[:2]

            big = pred.stable or "-"
            put_text(display, big, (20, 90), 3.0, (0, 255, 0) if pred.stable else (150, 150, 150), 5)
            draw_top3(display, pred.top, w - 250, 30)

            cv2.rectangle(display, (0, h - 60), (w, h), (30, 30, 30), -1)
            put_text(display, text[-38:] + "_", (12, h - 22), 0.9, (255, 255, 255), 2)
            put_text(display, "SPACE space | BACKSPACE delete | C clear | ESC quit",
                     (12, h - 70), 0.45)
            cv2.imshow(WINDOW, display)

            # --- Keyboard ----------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_closed(WINDOW):
                break
            elif key == ord(" "):
                text += " "
            elif key == 8:                            # BACKSPACE
                text = text[:-1]
            elif key in (ord("c"), ord("C")):         # C = clear
                text = ""

    cap.release()
    cv2.destroyAllWindows()
    if text.strip():
        print(f"You spelled: {text}")


if __name__ == "__main__":
    main()
