"""
train_dynamic.py - Stage 5: train a model on your recorded moving-sign clips.

    python train_dynamic.py

Same idea as train_model.py, with two differences:
  * Test split: for each sign, your NEWEST 20% of clips are held out as the
    test (clips are separate performances, so this is a fair test).
  * Augmentation: each training clip also gets several slightly altered
    copies (see sequences.augment), because you have far fewer clips than
    you had frames for the letters.
"""

import glob
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from collect_dynamic import SEQ_DIR
from sequences import NUM_SEQ_FEATURES, SEQ_LEN, augment, sequence_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "dynamic_model.joblib")
AUGMENT_COPIES = 8
TEST_FRACTION = 0.2
MIN_CLIPS = 8


def load_clips():
    """Return {label: [clip, clip, ...]} in recording order."""
    data = {}
    for folder in sorted(glob.glob(os.path.join(SEQ_DIR, "*"))):
        files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        if files:
            data[os.path.basename(folder)] = [np.load(f) for f in files]
    return data


def to_xy(clips_by_label, rng=None, copies=0):
    X, y = [], []
    for label, clips in clips_by_label.items():
        for clip in clips:
            X.append(sequence_features(clip))
            y.append(label)
            for _ in range(copies):
                X.append(sequence_features(augment(clip, rng)))
                y.append(label)
    return np.asarray(X), np.asarray(y)


def candidate_models():
    return {
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "Neural net (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, max_iter=1000,
                          early_stopping=True, random_state=0)),
    }


def main() -> None:
    data = load_clips()
    if len(data) < 2:
        raise SystemExit(f"Need clips of at least 2 signs in {SEQ_DIR}. "
                         "Run collect_dynamic.py first.")
    print("Clips per sign: " + ", ".join(f"{k}:{len(v)}" for k, v in data.items()))
    few = [k for k, v in data.items() if len(v) < MIN_CLIPS]
    if few:
        print(f"Warning: fewer than {MIN_CLIPS} clips for {', '.join(few)} - "
              "results for these will be unreliable.")

    # Newest clips of each sign -> test set.
    train, test = {}, {}
    for label, clips in data.items():
        n_test = max(1, int(round(len(clips) * TEST_FRACTION))) if len(clips) >= 3 else 0
        train[label] = clips[:len(clips) - n_test]
        test[label] = clips[len(clips) - n_test:]

    rng = np.random.default_rng(0)
    X_train, y_train = to_xy(train, rng, AUGMENT_COPIES)
    X_test, y_test = to_xy(test)
    print(f"Train: {len(y_train)} examples (incl. augmented copies)   Test: {len(y_test)} real clips")

    results = {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        if len(y_test):
            pred = model.predict(X_test)
            results[name] = (accuracy_score(y_test, pred), pred)
            print(f"  {name:<18} test accuracy: {results[name][0]:.1%}")
        else:
            results[name] = (0.0, np.array([]))
    best = max(results, key=lambda n: results[n][0])
    acc, pred = results[best]
    print(f"\nBest: {best} ({acc:.1%})")

    if len(y_test):
        print("\nPer sign (on your newest, unseen clips):")
        for label in data:
            mask = y_test == label
            if mask.any():
                a = (pred[mask] == label).mean()
                print(f"  {label:>14}: {a:6.1%}  ({mask.sum()} clips)"
                      + ("   <-- record more" if a < 0.8 else ""))
        mix = Counter((t, p) for t, p in zip(y_test, pred) if t != p)
        if mix:
            print("\nMix-ups (true -> predicted): "
                  + ", ".join(f"{t}->{p} x{n}" for (t, p), n in mix.most_common(6)))

    # Retrain on everything (with augmentation) and save.
    X_all, y_all = to_xy(data, np.random.default_rng(1), AUGMENT_COPIES)
    final = candidate_models()[best]
    final.fit(X_all, y_all)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final, "labels": sorted(data), "num_features": NUM_SEQ_FEATURES,
                 "seq_len": SEQ_LEN, "model_type": best, "test_accuracy": acc}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
