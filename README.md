# BIM Sign Tutor: learn fingerspelling with your webcam

A small Python app that helps you learn the **Bahasa Isyarat Malaysia (BIM)** fingerspelling alphabet. You sign a letter in front of your webcam, and it tells you whether you got it right.

It started as a "draw in the air" guessing game. Once I had hand tracking working, I realised the same idea could do something more useful: help people learn sign language, so more of us can talk to Deaf people in their first language.

> **Still a work in progress.** The model learns from recordings of *your own* hands, so it is only as correct as the way you sign. Please check every sign against a trusted BIM source, and if you can, with Deaf signers. This is a learning aid, not a translator.

## What it does

**Letters (one handshape)**

- **Record your own samples:** pick a letter, make the sign, press SPACE. It saves 60 samples in about 4 seconds.
- **Train a model:** a few seconds on a normal laptop, no GPU needed. It tells you which letters it struggles with and which ones it mixes up.
- **Live recognition:** fingerspell in front of the camera and watch the letters appear as text.
- **Tutor game** with three modes:
  - **Learn:** go through the alphabet one letter at a time. A small panel shows the target handshape with your live hand drawn over it, so you can see what's off.
  - **Quiz:** 10 random letters against the clock. Faster answers and streaks score more, and letters you get wrong come up more often.
  - **Spell:** fingerspell short Malay and English words like SAYA, MAKAN, RUMAH or BOOK.
- **Progress tracking:** it remembers how you do on each letter and tells you what to practise next.

**Signs that move**

- Record 1.5 second clips of signs like J, Z or TERIMA_KASIH, then train and test them the same way. The tutor game doesn't use these yet.

## How it works

1. **MediaPipe** finds 21 points on each hand (wrist, knuckles, fingertips) in every webcam frame.
2. The points are **moved so the wrist is at zero** and **scaled by palm size**, so it doesn't matter where your hand is on screen or how far you sit from the camera. What's left is basically just the shape of your hand, which is what a sign is.
3. A small **scikit-learn** model (a random forest or a small neural network, whichever scores better) learns from those numbers. Training on points instead of images means a few hundred samples per letter is enough.
4. When testing, it holds back whole recordings the model has never seen. Neighbouring frames are almost identical, so a random split would make the scores look better than they really are.
5. For moving signs, each clip is resampled to 20 frames and also stores how far the wrist travelled. That's how it tells **J** from **Z**: they share a handshape but trace a different path.

## How to run it

You need **Python 3.10 or newer** (tested on 3.14) and a webcam.

```bash
git clone https://github.com/rrdrepository/bim-sign-tutor.git
cd bim-sign-tutor
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows you can also just double-click **`start.bat`**. It activates the environment and gives you a menu.

Then, in this order:

| Step | Command | What to do |
|---|---|---|
| 1 | `python collect_data.py` | Press a letter key, make the sign, press SPACE. Do 3 or 4 rounds per letter. Press **0** to record "not a sign" (relaxed hands, random movement). |
| 2 | `python train_model.py` | Trains the model and shows accuracy per letter. |
| 3 | `python recognizer.py` | Live fingerspelling to text. |
| 4 | `python tutor.py` | The tutor game. |

For signs that move: `collect_dynamic.py`, then `train_dynamic.py`, then `dynamic_live.py`. Edit the `SIGNS` list at the top of `collect_dynamic.py` to pick your own signs.

The first run downloads MediaPipe's hand model (about 7 MB) into `models/`.

## Tips

- **Lighting matters most.** Face a window or a lamp, not away from one. A plain background helps.
- Keep your hand about an arm's length from the camera, with your palm towards it.
- While recording, **slowly tilt your hand and move a bit closer and further away**. Varied samples work much better than 60 identical ones.
- If two letters keep getting mixed up, record more of both, ideally in a different session or room.
- To repeat a letter (like the OO in BOOK), relax your hand briefly between the two.

## Files

| File | What it does |
|---|---|
| `hand_tracker.py` | Wraps MediaPipe hand tracking |
| `features.py` | Turns hand points into numbers the model can learn from |
| `collect_data.py` | Records letter samples |
| `train_model.py` | Trains and tests the letter model |
| `recognizer.py` | Live recognition, and the `Recognizer` class the tutor uses |
| `tutor.py` | The tutor game |
| `sequences.py` | Features for moving signs |
| `collect_dynamic.py` / `train_dynamic.py` / `dynamic_live.py` | Record, train and test moving signs |
| `ui.py` | Camera and drawing helpers |
| `start.bat` | Windows menu launcher |

Your recordings (`data/`) and trained models (`models/`) are not in this repo. They're recordings of your own hands, so they stay on your computer.

## What's next

- Moving signs inside the tutor game
- Recognising moving signs continuously, without pressing SPACE first
- Testing with more signers, so the model works on hands other than mine
- Checking against the BIM-SSD dataset from UNIMAS

## Built with

Python, OpenCV, [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), NumPy, scikit-learn and Matplotlib.
