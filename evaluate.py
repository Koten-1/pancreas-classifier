import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from utils.image_processor import preprocess_image
from utils.classifier import _extract_deep_features

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/classifier_binary.pkl")
CLASSES = ["pdac", "ipmn", "chronic_pancreatitis", "normal_pancreas", "non_pancreas"]

# ── Load model ─────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model       = data["model"]
label_map   = data["label_map"]
class_names = data["class_names"]

print(f"Model classes: {class_names}")
print(f"Label map: {label_map}\n")

# ── Load all images and predict ────────────────────────────────────────────
all_true  = []
all_pred  = []
all_files = []

for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        continue

    images = list(cls_dir.glob("*.jpg")) + \
             list(cls_dir.glob("*.jpeg")) + \
             list(cls_dir.glob("*.png"))

    print(f"Testing {cls}: {len(images)} images")

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            arr = preprocess_image(img)
            feats = _extract_deep_features(arr)
            pred_idx = model.predict([feats])[0]
            pred_label = class_names[pred_idx]

            all_true.append(cls)
            all_pred.append(pred_label)
            all_files.append(img_path.name)
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")

# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_true, all_pred))

# ── Per class accuracy ─────────────────────────────────────────────────────
print("PER CLASS ACCURACY")
print("="*50)
for cls in set(all_true):
    indices = [i for i, t in enumerate(all_true) if t == cls]
    correct = sum(1 for i in indices if all_pred[i] == all_true[i])
    total   = len(indices)
    pct     = correct / total * 100 if total > 0 else 0
    print(f"  {cls:30s} {correct}/{total}  ({pct:.0f}%)")

# ── Wrong predictions ──────────────────────────────────────────────────────
print("\nWRONG PREDICTIONS")
print("="*50)
for i, (t, p, f) in enumerate(zip(all_true, all_pred, all_files)):
    if t != p:
        print(f"  {f:40s} true={t:25s} pred={p}")