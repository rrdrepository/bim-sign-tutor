"""
train_model.py - Stage 2: train a classifier on your recorded samples.

    python train_model.py

What it does:
  1. Loads data/samples.csv (from collect_data.py).
  2. Splits it into TRAIN and TEST data *by burst* (see below).
  3. Trains two kinds of model and compares them on the test data.
  4. Prints accuracy per letter and which letters get confused.
  5. Retrains the best one on ALL the data and saves it to models/sign_model.joblib.

WHY SPLIT BY BURST, NOT RANDOMLY?
Each SPACE press records 60 frames in about 4 seconds, so neighbouring frames
are almost identical. If we split frames randomly, nearly every test frame has
a near-twin in the training data, and the model scores ~100% just by
remembering - that's called "data leakage". Real use is a NEW attempt at the
sign, so we hold out whole bursts the model has never seen. The score is lower
but honest.
"""

import csv
import os
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from collect_data import DATA_PATH, SAMPLES_PER_BURST
from features import NUM_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "sign_model.joblib")
CONFUSION_PNG = os.path.join(HERE, "models", "confusion_matrix.png")
GOOD_ENOUGH = 0.90      # letters below this test accuracy need more/better data


# --- 1. Load ---------------------------------------------------------------
def load_dataset():
    """Return features X, labels y, and a burst id for every row.

    collect_data.py always saves exactly SAMPLES_PER_BURST rows per burst, in
    order, and undo removes whole bursts - so row_index // 60 tells us which
    burst each row came from.
    """
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"No data found at {DATA_PATH}. Run collect_data.py first.")
    X, y = [], []
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                  # header
        for row in reader:
            if len(row) != 2 + NUM_FEATURES:
                continue                              # skip damaged rows
            y.append(row[0])
            X.append([float(v) for v in row[2:]])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    bursts = np.arange(len(y)) // SAMPLES_PER_BURST
    return X, y, bursts


# --- 2. Split ----------------------------------------------------------------
def split_by_burst(y, bursts):
    """Boolean mask: True = test row.

    For each label, the LAST burst you recorded becomes test data.
    If a label has only one burst, we use the last 30% of that burst instead
    (less honest, so we warn about it).
    """
    test = np.zeros(len(y), dtype=bool)
    weak_labels = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        label_bursts = np.unique(bursts[idx])
        if len(label_bursts) >= 2:
            test[idx[bursts[idx] == label_bursts[-1]]] = True
        else:
            cut = int(len(idx) * 0.7)
            test[idx[cut:]] = True
            weak_labels.append(label)
    return test, weak_labels


# --- 3. Models -----------------------------------------------------------------
def candidate_models():
    """Two different kinds of model, so you can see which suits the data.

    Random forest: hundreds of small decision trees that vote. Robust, needs no
      tuning, and copes with features on different scales.
    MLP (multi-layer perceptron): a small neural network. Often a bit more
      accurate on landmark data. Neural nets train best when every feature has
      a similar range, so we put a StandardScaler in front of it.
    """
    return {
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
        "Neural net (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                          max_iter=800, early_stopping=True, random_state=0)),
    }


# --- 4. Reporting --------------------------------------------------------------
def report(y_true, y_pred, labels):
    """Per-letter accuracy and the most common mix-ups."""
    print("\nAccuracy per label (on unseen bursts):")
    per_label = {}
    for lab in labels:
        mask = y_true == lab
        if mask.any():
            per_label[lab] = float((y_pred[mask] == lab).mean())
    for lab, acc in sorted(per_label.items(), key=lambda kv: kv[1]):
        flag = "" if acc >= GOOD_ENOUGH else "   <-- record more / check this sign"
        print(f"  {lab:>6}: {acc:6.1%}{flag}")

    mixups = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    if mixups:
        print("\nMost common mix-ups (true -> predicted):")
        for (t, p), n in mixups.most_common(8):
            print(f"  {t} -> {p}: {n} frames")
    return per_label


def save_confusion_png(y_true, y_pred, labels):
    """A picture of every mix-up. Rows = what you signed, columns = what the
    model said. A perfect model has numbers only on the diagonal."""
    try:
        import matplotlib
        matplotlib.use("Agg")                  # draw to file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Model predicted")
    ax.set_ylabel("You signed")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=120)
    plt.close(fig)
    print(f"\nConfusion matrix picture saved to {CONFUSION_PNG}")


# --- Main ------------------------------------------------------------------------
def main() -> None:
    X, y, bursts = load_dataset()
    labels = sorted(str(lab) for lab in np.unique(y))
    counts = Counter(y)
    print(f"Loaded {len(y)} samples, {len(labels)} labels:")
    print("  " + ", ".join(f"{lab}:{counts[lab]}" for lab in labels))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 different labels to train. Record more letters.")

    test, weak = split_by_burst(y, bursts)
    print(f"\nTrain: {(~test).sum()} samples   Test: {test.sum()} samples (held-out bursts)")
    if weak:
        print(f"Note: only one burst for {', '.join(weak)} - their test score is less "
              "reliable. Record a second burst of each for an honest score.")

    # Train each candidate on the training part, score on the unseen part.
    results = {}
    for name, model in candidate_models().items():
        model.fit(X[~test], y[~test])
        pred = model.predict(X[test])
        results[name] = (accuracy_score(y[test], pred), pred)
        print(f"  {name:<18} test accuracy: {results[name][0]:.1%}")

    best_name = max(results, key=lambda n: results[n][0])
    best_acc, best_pred = results[best_name]
    print(f"\nBest: {best_name} ({best_acc:.1%})")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    report(y[test], best_pred, labels)
    save_confusion_png(y[test], best_pred, labels)

    # Now that we know which model type works best, retrain it on ALL the
    # data - the held-out bursts are useful training examples too.
    final = candidate_models()[best_name]
    final.fit(X, y)
    joblib.dump({"model": final, "labels": labels, "num_features": NUM_FEATURES,
                 "model_type": best_name, "test_accuracy": best_acc}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
