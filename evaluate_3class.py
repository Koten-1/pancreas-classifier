import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from utils.image_processor import preprocess_image
from utils.classifier import _extract_deep_features

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
MODEL_PATH = Path("models/classifier_3class.pkl")

# Only these 3 classes
CLASSES = ["pdac", "ipmn", "chronic_pancreatitis"]
LABEL_NAMES = ["PDAC", "IPMN", "CP"]

# ── Load all images into memory ─────────────────────────────────────────────
print("Loading images...\n")
X, y, filenames = [], [], []

for idx, cls in enumerate(CLASSES):
    cls_dir = DATA_DIR / cls
    images  = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                and not f.name.startswith("aug_")]

    print(f"  {cls}: {len(images)} original images")

    for img_path in images:
        try:
            img  = Image.open(img_path).convert("RGB")
            arr  = preprocess_image(img)
            feat = _extract_deep_features(arr)
            X.append(feat)
            y.append(idx)
            filenames.append(img_path.name)
        except Exception as e:
            print(f"    Skip {img_path.name}: {e}")

X = np.array(X)
y = np.array(y)
print(f"\nTotal: {len(X)} images across {len(CLASSES)} classes")

# ── Train / test split (80/20) ──────────────────────────────────────────────
X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
    X, y, filenames,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Train fresh on training split ───────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
])
pipe.fit(X_train, y_train)

# ── Evaluate on held-out test set ───────────────────────────────────────────
y_pred = pipe.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"HELD-OUT TEST ACCURACY: {acc:.1%}")
print(f"{'='*50}")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

print("CONFUSION MATRIX:")
print("Rows = Actual, Columns = Predicted")
print(f"{'':>6} {'PDAC':>8} {'IPMN':>8} {'CP':>8}")
cm = confusion_matrix(y_test, y_pred)
for i, row in enumerate(cm):
    print(f"{LABEL_NAMES[i]:>6} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

print("\nPER CLASS ACCURACY:")
for i, name in enumerate(LABEL_NAMES):
    mask    = y_test == i
    correct = (y_pred[mask] == y_test[mask]).sum()
    total   = mask.sum()
    print(f"  {name}: {correct}/{total} ({correct/total:.0%})")

print("\nWRONG PREDICTIONS:")
for i, (true, pred, fname) in enumerate(zip(y_test, y_pred, f_test)):
    if true != pred:
        print(f"  {fname:45s} true={LABEL_NAMES[true]:6s} pred={LABEL_NAMES[pred]}")