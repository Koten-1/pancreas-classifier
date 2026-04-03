"""
Pancreas CT Classifier
- Organ gate: is this even a pancreas/abdominal CT?
- Binary MVP: PDAC vs Non-PDAC
- 3-class V2: PDAC / IPMN / Chronic Pancreatitis

Uses transfer learning (MobileNetV2 features) + scikit-learn classifier.
Works well on 30–200 images per class — no GPU needed.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

from utils.image_processor import preprocess_image, extract_visual_features


# ── Class definitions ─────────────────────────────────────────────────────────
BINARY_CLASSES = {
    "pdac":                  "PDAC",
    "non_pdac":              "Non-PDAC",
    "ipmn":                  "Non-PDAC",
    "chronic_pancreatitis":  "Non-PDAC",
    "normal_pancreas":       "Non-PDAC",
}

THREE_CLASS = {
    "pdac":                  "PDAC",
    "ipmn":                  "IPMN",
    "chronic_pancreatitis":  "CP",
    "normal_pancreas":       "Non-PDAC",
}

NON_PANCREAS_FOLDERS = {"non_pancreas", "chest", "brain", "knee", "xray", "mri_brain"}

MODEL_PATH_BINARY = "models/classifier_binary.pkl"
MODEL_PATH_3CLASS  = "models/classifier_3class.pkl"
ORGAN_GATE_PATH    = "models/organ_gate.pkl"
FEATURE_EXTRACTOR  = None   # lazy-loaded MobileNetV2


# ── Feature extraction ────────────────────────────────────────────────────────
def _get_feature_extractor():
    global FEATURE_EXTRACTOR
    if FEATURE_EXTRACTOR is None:
        try:
            import tensorflow as tf
            base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling="avg",
                weights="imagenet",
            )
            base.trainable = False
            FEATURE_EXTRACTOR = base
        except ImportError:
            pass   # fall back to handcrafted features
    return FEATURE_EXTRACTOR


def _extract_deep_features(img_array: np.ndarray) -> np.ndarray:
    """MobileNetV2 feature vector (1280-d). Falls back to HOG if TF unavailable."""
    extractor = _get_feature_extractor()
    if extractor is not None:
        import tensorflow as tf
        inp = np.expand_dims(img_array, 0)
        inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
        feats = extractor(inp, training=False).numpy().flatten()
        return feats
    else:
        return _extract_handcrafted_features(img_array)


def _extract_handcrafted_features(img_array: np.ndarray) -> np.ndarray:
    """
    Fallback feature set when TensorFlow is not installed.
    Uses intensity histograms + texture statistics across image patches.
    ~200-d vector, good enough for prototype with sklearn.
    """
    gray = img_array.mean(axis=-1)   # (224, 224)

    features = []

    # Global histogram (32 bins)
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    features.extend((hist / (hist.sum() + 1e-6)).tolist())

    # Divide into 4×4 grid — local stats per patch
    patch_h, patch_w = 56, 56
    for i in range(4):
        for j in range(4):
            patch = gray[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            features.extend([
                float(patch.mean()),
                float(patch.std()),
                float(np.percentile(patch, 25)),
                float(np.percentile(patch, 75)),
                float((patch < 0.2).mean()),   # dark fraction
                float((patch > 0.7).mean()),   # bright fraction (calcifications)
            ])

    # Gradient-based texture over full image
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    features.extend([
        float(np.abs(gy).mean()),
        float(np.abs(gx).mean()),
        float(np.abs(gy).std()),
        float(np.abs(gx).std()),
    ])

    return np.array(features, dtype=np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_dataset(data_dir: str, class_map: dict, organ_gate_mode: bool = False):
    """
    Load images from subdirectories.
    Returns (X, y, class_names).
    """
    data_dir = Path(data_dir)
    X, y = [], []
    class_names = []
    label_map = {}

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower()

        if organ_gate_mode:
            label = "non_pancreas" if folder_lower in NON_PANCREAS_FOLDERS else "pancreas"
        else:
            label = class_map.get(folder_lower)
            if label is None:
                continue

        if label not in label_map:
            label_map[label] = len(label_map)
            class_names.append(label)

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png")) + list(folder.glob("*.webp"))

        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                arr = preprocess_image(img)
                feats = _extract_deep_features(arr)
                X.append(feats)
                y.append(label_map[label])
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

    if not X:
        return None, None, None, None

    return np.array(X), np.array(y), class_names, label_map


# ── Classifier class ──────────────────────────────────────────────────────────
class PancreasClassifier:
    def __init__(self, binary: bool = True):
        self.binary = binary
        self.clf = None
        self.organ_gate = None
        self.label_map = {}
        self.class_names = []

    def train(self, data_dir: str) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import accuracy_score

        Path("models").mkdir(exist_ok=True)

        # ── Organ gate ────────────────────────────────────────────────────────
        print("Training organ gate...")
        X_org, y_org, cn_org, lm_org = _load_dataset(data_dir, {}, organ_gate_mode=True)

        if X_org is not None and len(np.unique(y_org)) >= 2:
            pipe_org = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(max_iter=1000, C=1.0)),
            ])
            pipe_org.fit(X_org, y_org)
            self.organ_gate = {"model": pipe_org, "label_map": lm_org, "class_names": cn_org}
            with open(ORGAN_GATE_PATH, "wb") as f:
                pickle.dump(self.organ_gate, f)
            print(f"  Organ gate: {len(X_org)} samples, {len(cn_org)} classes")
        else:
            print("  Not enough non_pancreas samples — organ gate skipped")

        # ── Disease classifier ────────────────────────────────────────────────
        class_map = BINARY_CLASSES if self.binary else THREE_CLASS
        X, y, class_names, label_map = _load_dataset(data_dir, class_map)

        if X is None:
            return {"success": False, "message": "No training images found. Check your data folder structure."}

        print(f"Training disease classifier on {len(X)} samples, {len(class_names)} classes...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ])

        # Cross-validation if enough data
        if len(X) >= 20:
            cv = StratifiedKFold(n_splits=min(5, len(X)//4), shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            cv_acc = scores.mean()
        else:
            cv_acc = None

        pipe.fit(X, y)
        train_acc = accuracy_score(y, pipe.predict(X))

        self.clf = pipe
        self.label_map = label_map
        self.class_names = class_names

        model_path = MODEL_PATH_BINARY if self.binary else MODEL_PATH_3CLASS
        with open(model_path, "wb") as f:
            pickle.dump({"model": pipe, "label_map": label_map, "class_names": class_names}, f)

        print(f"  Train acc: {train_acc:.2f}" + (f", CV acc: {cv_acc:.2f}" if cv_acc else ""))
        return {
            "success": True,
            "n_samples": len(X),
            "accuracy": cv_acc if cv_acc else train_acc,
        }

    def load_or_init(self):
        """Load saved model or use heuristic fallback."""
        model_path = MODEL_PATH_BINARY if self.binary else MODEL_PATH_3CLASS

        # Try loading organ gate
        if Path(ORGAN_GATE_PATH).exists():
            with open(ORGAN_GATE_PATH, "rb") as f:
                self.organ_gate = pickle.load(f)

        # Try loading classifier
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.clf         = data["model"]
            self.label_map   = data["label_map"]
            self.class_names = data["class_names"]
            return True

        # No trained model — use heuristic
        self.clf = None
        return False

    def predict(self, img_array: np.ndarray, img_pil: Image.Image) -> dict:
        """
        Classify a single image.
        Returns a rich result dict for the UI.
        """
        visual_feats = extract_visual_features(img_pil)

        # ── Organ gate ────────────────────────────────────────────────────────
        is_pancreas, detected_as, organ_conf = self._organ_gate_check(img_array, visual_feats)

        if not is_pancreas:
            return {
                "is_pancreas": False,
                "detected_as": detected_as,
                "organ_confidence": organ_conf,
                "verdict": None,
                "confidence": 0.0,
                "scores": {},
                "features": {},
                "explanation": "",
            }

        # ── Disease classification ────────────────────────────────────────────
        if self.clf is not None:
            feats = _extract_deep_features(img_array)
            proba = self.clf.predict_proba([feats])[0]
            scores_raw = {self.class_names[i]: float(proba[i]) for i in range(len(proba))}
        else:
            # Heuristic fallback (no trained model yet)
            scores_raw = self._heuristic_predict(visual_feats)

        verdict    = max(scores_raw, key=scores_raw.get)
        confidence = scores_raw[verdict]

        # Low-confidence → "Indeterminate"
        if confidence < 0.55:
            verdict    = "Indeterminate"
            confidence = max(scores_raw.values())

        # ── Feature interpretation ────────────────────────────────────────────
        feature_analysis = self._interpret_features(visual_feats, verdict)
        explanation      = self._build_explanation(visual_feats, verdict, scores_raw)

        return {
            "is_pancreas":   True,
            "detected_as":   "Abdominal CT",
            "organ_confidence": organ_conf,
            "verdict":       verdict,
            "confidence":    confidence,
            "scores":        scores_raw,
            "features":      feature_analysis,
            "explanation":   explanation,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _organ_gate_check(self, img_array: np.ndarray, visual_feats: dict):
        """Returns (is_pancreas, detected_as, confidence)."""
        if self.organ_gate is not None:
            feats = _extract_deep_features(img_array)
            lm    = self.organ_gate["label_map"]
            model = self.organ_gate["model"]
            proba = model.predict_proba([feats])[0]

            pancreas_idx   = lm.get("pancreas",   0)
            nonpancreas_idx = lm.get("non_pancreas", 1)

            p_pancreas = proba[pancreas_idx] if pancreas_idx < len(proba) else 0.5

            if p_pancreas < 0.40:
                return False, "Non-pancreatic image", float(1 - p_pancreas)
            return True, "Abdominal CT", float(p_pancreas)

        # Fallback heuristic: CT scans are dark + grayscale
        is_ct_like = (
            visual_feats["bilateral_symmetry"] > 0.3
            and visual_feats["mean_intensity"] < 0.55
        )
        return is_ct_like, "Abdominal CT" if is_ct_like else "Non-CT image", 0.6

    def _heuristic_predict(self, vf: dict) -> dict:
        """
        Rule-based heuristic when no trained model is available.
        Approximates radiological decision logic from the literature.
        Used as demo fallback — replace with trained model ASAP.
        """
        pdac_score = 0.0
        ipmn_score = 0.0
        cp_score   = 0.0
        nonpdac_score = 0.0

        # High edge density + low intensity central region → PDAC (solid hypodense mass)
        if vf["edge_density"] > 0.55 and vf["mean_intensity"] < 0.35:
            pdac_score += 0.4
        elif vf["edge_density"] > 0.4:
            pdac_score += 0.2

        # High dark fraction → cystic (IPMN)
        if vf["dark_region_fraction"] > 0.3:
            ipmn_score += 0.35

        # High bright spots → calcifications (CP)
        if vf["bright_spot_fraction"] > 0.08:
            cp_score += 0.4

        # High texture heterogeneity → PDAC or CP
        if vf["texture_heterogeneity"] > 0.20:
            pdac_score += 0.15
            cp_score   += 0.10

        # Normalize
        total = pdac_score + ipmn_score + cp_score + 0.15
        nonpdac_score = max(0.1, 0.5 - pdac_score)

        if self.binary:
            non_pdac = 1.0 - (pdac_score / max(total, 0.01))
            return {
                "PDAC":     round(pdac_score / max(total, 0.01), 3),
                "Non-PDAC": round(non_pdac, 3),
            }
        else:
            s = {
                "PDAC": pdac_score,
                "IPMN": ipmn_score,
                "CP":   cp_score,
            }
            total = sum(s.values()) + 1e-6
            return {k: round(v / total, 3) for k, v in s.items()}

    def _interpret_features(self, vf: dict, verdict: str) -> dict:
        """Map visual features to radiological concepts with risk levels."""

        def level(val, low_thresh, high_thresh):
            if val >= high_thresh: return "high"
            if val >= low_thresh:  return "med"
            return "low"

        feats = {}

        # Hypodensity (dark regions → mass, necrosis)
        lv = level(vf["dark_region_fraction"], 0.15, 0.30)
        feats["Hypodense region"] = {
            "level": lv,
            "label": {
                "high": "Prominent",
                "med":  "Moderate",
                "low":  "Absent",
            }[lv]
        }

        # Calcifications (bright spots)
        lv = level(vf["bright_spot_fraction"], 0.05, 0.12)
        feats["Calcifications"] = {
            "level": "med" if lv == "high" else "low",
            "label": {"high": "Extensive", "med": "Mild", "low": "Absent"}[lv]
        }

        # Structural complexity (edge density)
        lv = level(vf["edge_density"], 0.40, 0.60)
        feats["Structural complexity"] = {
            "level": lv,
            "label": {"high": "High (irregular)", "med": "Moderate", "low": "Low (smooth)"}[lv]
        }

        # Texture heterogeneity
        lv = level(vf["texture_heterogeneity"], 0.15, 0.25)
        feats["Parenchyma texture"] = {
            "level": lv,
            "label": {"high": "Heterogeneous", "med": "Mildly irregular", "low": "Homogeneous"}[lv]
        }

        return feats

    def _build_explanation(self, vf: dict, verdict: str, scores: dict) -> str:
        """Generate a brief text explanation of the classification."""
        explanations = {
            "PDAC": (
                "Findings consistent with solid pancreatic mass: hypodense region, "
                "irregular structural borders, and heterogeneous parenchymal texture. "
                "Abrupt duct cutoff pattern suggested. Urgent surgical evaluation indicated."
            ),
            "Non-PDAC": (
                "No dominant solid hypodense mass identified. Pattern more consistent "
                "with benign or pre-malignant etiology. Routine surveillance appropriate."
            ),
            "IPMN": (
                "Prominent cystic/dark regions with smooth duct morphology. "
                "Grape-like dilation pattern suggested. "
                "Main-duct IPMN surveillance per IAP guidelines recommended."
            ),
            "CP": (
                "Bright calcifications and heterogeneous parenchyma consistent with "
                "chronic inflammatory changes. Beaded duct morphology pattern. "
                "Medical management and enzyme replacement indicated."
            ),
            "Indeterminate": (
                "Findings are mixed — multiple diagnostic possibilities remain. "
                "Radiologist review and clinical correlation required. "
                "Consider EUS or MRCP for further characterisation."
            ),
        }
        base = explanations.get(verdict, "Unable to generate explanation.")

        # Add note if model is heuristic
        if self.clf is None:
            base += " ⚠ Heuristic mode — train model on your own images for accurate results."
        return base
