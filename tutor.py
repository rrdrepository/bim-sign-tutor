"""
tutor.py - Stage 4: the BIM fingerspelling tutor game.

    python tutor.py

Modes (pick from the menu):
  1  LEARN  - go through the letters one by one. The reference panel shows the
              target handshape (from YOUR recordings) with your live hand drawn
              over it, so you can see what's different.
  2  QUIZ   - 10 random letters, 10 seconds each. Faster = more points, and a
              streak of correct answers multiplies your score. Letters you
              struggle with come up more often.
  3  SPELL  - fingerspell whole words, letter by letter.

Your progress (accuracy and speed per letter) is saved to data/progress.json,
so the menu can tell you which letters to practise.

Controls:  menu: 1 / 2 / 3, ESC quits
           in a mode: H = hint (show/hide reference), N = skip,
                      B = back (learn mode), ESC = back to menu
"""

import csv
import json
import os
import random
import time

import cv2
import numpy as np

from collect_data import DATA_PATH
from features import FEATURES_PER_HAND, NUM_FEATURES, normalise_hand
from hand_tracker import HandTracker, draw_hand
from recognizer import NONE_LABEL, Recognizer
from ui import draw_normalised_hand, open_camera, put_text, window_closed

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(HERE, "data", "progress.json")
WINDOW = "BIM Tutor"
DISPLAY_W, DISPLAY_H = 960, 720   # we track on 640x480 but draw bigger, for readability

HOLD_FRAMES = 12         # frames (~0.4 s) the correct sign must be held
TARGET_MIN_PROB = 0.40   # min confidence for the target letter to count in a frame
SHOW_MODEL_GUESS = True  # show "model sees: X 45%" - handy for finding weak letters
QUIZ_ROUNDS = 10
QUIZ_TIME_S = 10.0
SPELL_WORDS = 5
SPELL_TIME_PER_LETTER_S = 6.0
PAUSE_AFTER_RESULT_S = 1.0

# Short Malay and English words. Only words whose letters your model knows
# are used, so this list works even if you haven't recorded every letter.
WORDS = [
    "SAYA", "KAMU", "MAKAN", "MINUM", "RUMAH", "BUKU", "BOLA", "IBU", "AYAH",
    "ADIK", "KAKAK", "AIR", "NASI", "KUCING", "IKAN", "BAIK", "TOLONG", "MAAF",
    "CAT", "DOG", "SUN", "TREE", "HELLO", "BOOK", "FISH", "HAPPY", "WATER",
]

GREEN, RED, YELLOW, WHITE, GREY = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255), (150, 150, 150)


# --- Reference handshapes ------------------------------------------------------
def load_reference_shapes() -> dict[str, np.ndarray]:
    """Average handshape per letter, computed from your recorded samples.

    Each sample is already normalised (wrist at 0, palm size 1), so averaging
    them gives a clean 'typical' version of how YOU make each sign. We use the
    right-hand slot, or the left slot if the right is empty.
    Note: it's only as correct as your recordings - check them against a
    trusted BIM source.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    if not os.path.exists(DATA_PATH):
        return {}
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 2 + NUM_FEATURES:
                continue
            feats = np.asarray(row[2:], dtype=np.float32)
            hand = feats[:FEATURES_PER_HAND]
            if not hand.any():
                hand = feats[FEATURES_PER_HAND:]
            sums[row[0]] = sums.get(row[0], 0) + hand
            counts[row[0]] = counts.get(row[0], 0) + 1
    return {lab: (sums[lab] / counts[lab]).reshape(21, 3) for lab in sums}


# --- Progress tracking ---------------------------------------------------------
class Progress:
    """Remembers how you do on each letter, across sessions."""

    def __init__(self, path: str = PROGRESS_PATH):
        self.path = path
        self.stats: dict[str, dict] = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.stats = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.stats = {}            # corrupted file: start fresh

    def record(self, letter: str, correct: bool, seconds: float) -> None:
        s = self.stats.setdefault(letter, {"tries": 0, "correct": 0, "time": 0.0})
        s["tries"] += 1
        if correct:
            s["correct"] += 1
            s["time"] += seconds
        self.save()

    def accuracy(self, letter: str) -> float | None:
        s = self.stats.get(letter)
        return s["correct"] / s["tries"] if s and s["tries"] else None

    def avg_time(self, letter: str) -> float | None:
        s = self.stats.get(letter)
        return s["time"] / s["correct"] if s and s["correct"] else None

    def weight(self, letter: str) -> float:
        """How often the quiz should pick this letter. Weak or slow letters
        come up more - a simple version of 'spaced repetition'."""
        acc = self.accuracy(letter)
        if acc is None:
            return 2.0                      # never tried: worth practising
        slow = (self.avg_time(letter) or QUIZ_TIME_S) / QUIZ_TIME_S
        return 1.0 + 3.0 * (1.0 - acc) + 2.0 * slow

    def weakest(self, letters, n: int = 5) -> list[str]:
        """Letters that need work: below 90% accuracy, or slow (>4 s on average)."""
        weak = [l for l in letters
                if self.accuracy(l) is not None
                and (self.accuracy(l) < 0.9 or (self.avg_time(l) or 0) > 4.0)]
        return sorted(weak, key=lambda l: -self.weight(l))[:n]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.stats, f, indent=1)
        os.replace(tmp, self.path)


# --- The game --------------------------------------------------------------------
class Tutor:
    def __init__(self, letters, refs, progress: Progress):
        self.letters = letters
        self.refs = refs
        self.progress = progress
        known = set(letters)
        self.words = [w for w in WORDS if set(w) <= known]
        self.state = "menu"
        self.message, self.message_colour, self.message_until = "", WHITE, 0.0
        self.summary_lines: list[str] = []

    # ----- shared "sign the target letter" logic -----------------------------------
    def _set_target(self, letter: str, now: float) -> None:
        self.target = letter
        self.target_start = now
        self.hold = 0
        self.wrong_label, self.wrong_since = None, None
        self.hint_used = False

    def _check_target(self, pred) -> bool:
        """True once the target sign has been held steadily for HOLD_FRAMES.

        The tutor knows which letter it asked for, so it can be more forgiving
        than open-ended recognition: a frame counts if the target is the
        model's TOP guess with at least TARGET_MIN_PROB confidence. Letters
        with close look-alikes (A vs S/E/T, for example) often win with only
        40-55%, which never passed the recognizer's stricter 60% + voting rule.
        The hold bar still requires ~0.4 s of consistent frames, so a random
        flicker can't pass.

        `blocked` stops a sign you're still holding from instantly counting for
        the NEXT target too - needed for double letters like the OO in BOOK.
        You have to relax or change your hand first.
        """
        top_label = pred.top[0][0] if pred.top else None
        top_prob = pred.top[0][1] if pred.top else 0.0
        if self.blocked and top_label != self.blocked:
            self.blocked = None
        if (top_label == self.target and top_prob >= TARGET_MIN_PROB
                and self.blocked != self.target):
            self.hold += 1
        else:
            self.hold = max(0, self.hold - 2)   # decay rather than reset: forgives a flicker
        return self.hold >= HOLD_FRAMES

    def _track_wrong(self, pred, now: float) -> None:
        """Remember if you've been showing a different letter for a while."""
        wrong = pred.stable if pred.stable not in (None, self.target) else None
        if wrong != self.wrong_label:
            self.wrong_label, self.wrong_since = wrong, now

    def _say(self, text: str, colour=WHITE, seconds: float = 1.5) -> None:
        self.message, self.message_colour = text, colour
        self.message_until = time.monotonic() + seconds

    # ----- starting modes ---------------------------------------------------------
    def start_learn(self, now):
        self.state, self.index, self.show_ref = "learn", 0, True
        self.blocked, self.pause_until = None, 0.0
        self._set_target(self.letters[0], now)

    def start_quiz(self, now):
        self.state, self.round, self.score, self.streak = "quiz", 0, 0, 0
        self.correct_count, self.missed, self.show_ref = 0, [], False
        self.blocked, self.pause_until = None, 0.0
        self._next_quiz_letter(now)

    def start_spell(self, now):
        if not self.words:
            self._say("No words fit your recorded letters yet - record more letters!", RED, 3)
            return
        self.state, self.word_no, self.score = "spell", 0, 0
        self.words_done, self.missed, self.show_ref = 0, [], False
        self.word_queue = random.sample(self.words, min(SPELL_WORDS, len(self.words)))
        self.blocked, self.pause_until = None, 0.0
        self._next_word(now)

    # ----- per-mode helpers -------------------------------------------------------
    def _next_quiz_letter(self, now):
        previous = getattr(self, "target", None)
        choices = [l for l in self.letters if l != previous] or self.letters
        letter = random.choices(choices, weights=[self.progress.weight(l) for l in choices])[0]
        self.round += 1
        self._set_target(letter, now)

    def _next_word(self, now):
        self.word = self.word_queue[self.word_no]
        self.pos = 0
        self.word_start = now
        self.word_time = SPELL_TIME_PER_LETTER_S * len(self.word) + 4
        self._set_target(self.word[0], now)

    def _finish(self, title: str, lines: list[str]):
        weak = self.progress.weakest(self.letters)
        self.summary_lines = [title, *lines,
                              "Practise next: " + (", ".join(weak) if weak else "-"),
                              "", "Press SPACE for the menu"]
        self.state = "summary"

    # ----- per-frame update -----------------------------------------------------------
    def update(self, pred, now: float) -> None:
        if self.state in ("menu", "summary") or now < self.pause_until:
            return
        self._track_wrong(pred, now)
        correct = self._check_target(pred)
        elapsed = now - self.target_start

        if self.state == "learn":
            if correct:
                self.progress.record(self.target, True, elapsed)
                self.blocked = self.target
                self._say("Bagus! Well done!", GREEN)
                self.pause_until = now + PAUSE_AFTER_RESULT_S
                self.index = (self.index + 1) % len(self.letters)
                self._set_target(self.letters[self.index], now + PAUSE_AFTER_RESULT_S)

        elif self.state == "quiz":
            time_left = QUIZ_TIME_S - elapsed
            if correct:
                self.streak += 1
                points = 100 + int(10 * time_left)
                points = int(points * (1 + 0.1 * min(self.streak - 1, 5)))  # streak bonus, max x1.5
                if self.hint_used:
                    points //= 2
                self.score += points
                self.correct_count += 1
                self.progress.record(self.target, True, elapsed)
                self.blocked = self.target
                self._say(f"Correct! +{points}" + (f"  streak x{self.streak}" if self.streak > 1 else ""), GREEN)
            elif time_left <= 0:
                self.streak = 0
                self.missed.append(self.target)
                self.progress.record(self.target, False, QUIZ_TIME_S)
                self._say(f"Time's up - that was {self.target}", RED)
            else:
                return
            self.pause_until = now + PAUSE_AFTER_RESULT_S
            if self.round >= QUIZ_ROUNDS:
                self._finish("QUIZ COMPLETE", [
                    f"Score: {self.score}",
                    f"Correct: {self.correct_count}/{QUIZ_ROUNDS}",
                    "Missed: " + (", ".join(self.missed) if self.missed else "none!")])
            else:
                self._next_quiz_letter(now + PAUSE_AFTER_RESULT_S)

        elif self.state == "spell":
            word_left = self.word_time - (now - self.word_start)
            if correct:
                self.progress.record(self.target, True, elapsed)
                self.blocked = self.target
                self.score += 50
                self.pos += 1
                if self.pos < len(self.word):
                    self._set_target(self.word[self.pos], now)
                    return
                bonus = 100 + int(10 * max(word_left, 0))
                self.score += bonus
                self.words_done += 1
                self._say(f"{self.word}! +{bonus} bonus", GREEN)
            elif word_left <= 0:
                self.missed.append(self.target)
                self.progress.record(self.target, False, elapsed)
                self._say(f"Time's up on {self.target} (word: {self.word})", RED, 2)
            else:
                return
            self.pause_until = now + PAUSE_AFTER_RESULT_S
            self.word_no += 1
            if self.word_no >= len(self.word_queue):
                self._finish("SPELLING COMPLETE", [
                    f"Score: {self.score}",
                    f"Words: {self.words_done}/{len(self.word_queue)}",
                    "Stuck on: " + (", ".join(self.missed) if self.missed else "nothing!")])
            else:
                self.word_start = now + PAUSE_AFTER_RESULT_S
                self._next_word(now + PAUSE_AFTER_RESULT_S)

    # ----- keyboard ------------------------------------------------------------------
    def handle_key(self, key: int, now: float) -> bool:
        """Return False to quit the program."""
        if self.state == "menu":
            if key == 27:
                return False
            if key == ord("1"):
                self.start_learn(now)
            elif key == ord("2"):
                self.start_quiz(now)
            elif key == ord("3"):
                self.start_spell(now)
        elif self.state == "summary":
            if key in (ord(" "), 13, 27):
                self.state = "menu"
        else:
            if key == 27:
                self.state = "menu"
            elif key in (ord("h"), ord("H")):
                self.show_ref = not self.show_ref
                if self.show_ref:
                    self.hint_used = True
            elif key in (ord("n"), ord("N")) and self.state == "learn":
                self.index = (self.index + 1) % len(self.letters)
                self._set_target(self.letters[self.index], now)
            elif key in (ord("b"), ord("B")) and self.state == "learn":
                self.index = (self.index - 1) % len(self.letters)
                self._set_target(self.letters[self.index], now)
            elif key in (ord("n"), ord("N")) and self.state == "quiz":
                self.target_start = now - QUIZ_TIME_S        # skip = let the timer run out
        return True

    # ----- drawing -------------------------------------------------------------------
    def draw(self, img, pred, hands, now: float) -> None:
        if self.state == "menu":
            self._draw_menu(img)
        elif self.state == "summary":
            self._draw_summary(img)
        else:
            self._draw_play(img, pred, hands, now)
        if now < self.message_until:
            put_text(img, self.message, (30, DISPLAY_H - 30), 1.0, self.message_colour, 2)

    def _panel(self, img, x0, y0, x1, y1, alpha=0.6):
        """Semi-transparent dark box, so text is readable over the video."""
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = (roi * (1 - alpha)).astype(np.uint8)

    def _draw_menu(self, img):
        self._panel(img, 140, 120, 820, 600, 0.75)
        put_text(img, "BIM FINGERSPELLING TUTOR", (190, 190), 1.3, YELLOW, 3)
        put_text(img, "1   Learn  - letter by letter, with reference", (200, 280), 0.9)
        put_text(img, "2   Quiz   - 10 letters against the clock", (200, 330), 0.9)
        put_text(img, f"3   Spell  - fingerspell words ({len(self.words)} available)", (200, 380), 0.9)
        weak = self.progress.weakest(self.letters)
        if weak:
            put_text(img, "Practise next: " + ", ".join(weak), (200, 450), 0.8, YELLOW)
        put_text(img, f"Letters in your model: {''.join(self.letters)}", (200, 500), 0.6, GREY)
        put_text(img, "ESC to quit", (200, 560), 0.7, GREY)

    def _draw_summary(self, img):
        self._panel(img, 140, 120, 820, 600, 0.75)
        for i, line in enumerate(self.summary_lines):
            colour, scale = (YELLOW, 1.3) if i == 0 else (WHITE, 0.9)
            put_text(img, line, (200, 200 + i * 55), scale, colour, 2 if i == 0 else 1)

    def _draw_play(self, img, pred, hands, now):
        self._panel(img, 0, 0, DISPLAY_W, 70)
        mode = self.state.upper()
        put_text(img, mode, (20, 45), 1.0, YELLOW, 2)

        # Top bar: mode-specific info
        if self.state == "learn":
            put_text(img, f"Letter {self.index + 1}/{len(self.letters)}", (180, 45), 0.8)
            put_text(img, "H hint | N next | B back | ESC menu", (520, 45), 0.6, GREY)
        elif self.state == "quiz":
            left = max(0.0, QUIZ_TIME_S - (now - self.target_start)) if now >= self.pause_until else QUIZ_TIME_S
            put_text(img, f"Round {self.round}/{QUIZ_ROUNDS}   Score {self.score}   Streak {self.streak}",
                     (160, 45), 0.8)
            put_text(img, f"{left:4.1f}s", (820, 45), 1.0, RED if left < 3 else WHITE, 2)
        elif self.state == "spell":
            left = max(0.0, self.word_time - (now - self.word_start))
            put_text(img, f"Word {self.word_no + 1}/{len(self.word_queue)}   Score {self.score}",
                     (160, 45), 0.8)
            put_text(img, f"{left:4.1f}s", (820, 45), 1.0, RED if left < 4 else WHITE, 2)

        # Target: big letter (or the word with progress) on the left
        self._panel(img, 20, 90, 300, 380)
        if self.state == "spell":
            x = 35
            for i, ch in enumerate(self.word):
                colour = GREEN if i < self.pos else YELLOW if i == self.pos else GREY
                put_text(img, ch, (x, 160), 1.2, colour, 3)
                x += 34
            put_text(img, "Sign:", (35, 215), 0.7, GREY)
            put_text(img, self.target, (110, 340), 4.5, YELLOW, 8)
        else:
            put_text(img, "Sign:", (35, 130), 0.7, GREY)
            put_text(img, self.target, (90, 320), 6.0, YELLOW, 10)

        # Hold bar: fills while you hold the correct sign
        frac = min(self.hold / HOLD_FRAMES, 1.0)
        cv2.rectangle(img, (35, 355), (285, 368), (60, 60, 60), -1)
        cv2.rectangle(img, (35, 355), (35 + int(250 * frac), 368), GREEN, -1)

        # Reference panel (right): target shape + your live hand on top
        if self.show_ref:
            box = 240
            x0, y0 = DISPLAY_W - box - 20, 90
            ref = self.refs.get(self.target)
            draw_normalised_hand(img, ref, (x0, y0), box, WHITE)
            if hands:
                # Your live hand, drawn on top without clearing the box.
                live = normalise_hand(hands[0].points3d)
                draw_normalised_hand(img, live, (x0, y0), box, YELLOW, background=False)
            put_text(img, "white = target, yellow = you", (x0, y0 + box + 25), 0.55)
            if ref is None:
                put_text(img, "no recordings", (x0 + 50, y0 + box // 2), 0.6, GREY)
        else:
            put_text(img, "H = show reference", (DISPLAY_W - 250, 110), 0.6, GREY)

        # Feedback line
        if now >= self.message_until and now >= self.pause_until:
            if not hands:
                fb, colour = "Show your hand to the camera", GREY
            elif self.blocked == self.target:
                # Same letter twice in a row (e.g. the OO in BOOK).
                fb, colour = f"Relax your hand, then sign {self.target} again", YELLOW
            elif self.hold > 0:
                fb, colour = "Hold it...", GREEN
            elif self.wrong_label and now - self.wrong_since > 0.8:
                fb, colour = f"That looks like {self.wrong_label} - compare with the reference (H)", YELLOW
            else:
                fb, colour = "", WHITE
            if fb:
                put_text(img, fb, (30, DISPLAY_H - 30), 0.9, colour, 2)

        # What the model thinks right now (top 3), bottom-right corner.
        if SHOW_MODEL_GUESS and pred.top:
            guesses = "  ".join(f"{'none' if l == NONE_LABEL else l} {p:.0%}"
                                for l, p in pred.top)
            put_text(img, f"model sees: {guesses}", (DISPLAY_W - 360, DISPLAY_H - 80), 0.55, GREY)


def main() -> None:
    recognizer = Recognizer()
    letters = sorted(c for c in recognizer.classes if c != NONE_LABEL)
    if not letters:
        raise SystemExit("Your model doesn't know any letters yet. Record and train first.")
    print(f"Model: {recognizer.model_info}  Letters: {''.join(letters)}")
    tutor = Tutor(letters, load_reference_shapes(), Progress())

    cap = open_camera()
    cv2.namedWindow(WINDOW)
    with HandTracker(max_hands=2) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            hands = tracker.process(frame)
            pred = recognizer.update(hands)
            now = time.monotonic()
            tutor.update(pred, now)

            # Draw on an enlarged copy; scale the landmark pixels to match.
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            sx, sy = DISPLAY_W / w, DISPLAY_H / h
            for hand in hands:
                draw_hand(display, [(int(x * sx), int(y * sy)) for x, y in hand.landmarks])
            tutor.draw(display, pred, hands, now)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if window_closed(WINDOW):
                break
            if key != 255 and not tutor.handle_key(key, now):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
