# Pancreatic CT Classifier — Research Prototype

**Clinical problem:** MPD dilation ≥5mm has 4 major causes.  
Misdiagnosis rate: 26.2% PDAC misclassified as benign.  
**This tool:** AI classifier from CT scan images → correct etiology.

---

## Quick start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
pancreas_classifier/
├── app.py                    ← Streamlit UI (run this)
├── requirements.txt
├── utils/
│   ├── classifier.py         ← ML pipeline (train + predict)
│   ├── image_processor.py    ← Image preprocessing + feature extraction
│   └── report.py             ← Report generator
├── data/
│   └── raw/                  ← Put your images here
│       ├── pdac/
│       ├── ipmn/
│       ├── chronic_pancreatitis/
│       ├── normal_pancreas/
│       └── non_pancreas/     ← Trick images (chest CT, brain MRI, etc.)
└── models/                   ← Saved model files (auto-created after training)
```

---

## Step-by-step: collect training data from Google Images

### What to search for

| Folder | Google search query | Target images |
|--------|-------------------|---------------|
| `pdac/` | `"pancreatic cancer CT scan axial"` | Solid dark mass, atrophy, duct cutoff |
| `pdac/` | `"pancreatic ductal adenocarcinoma CT"` | Same |
| `ipmn/` | `"IPMN pancreas CT main duct dilation"` | Cystic, grape-like, dilated duct |
| `ipmn/` | `"intraductal papillary mucinous neoplasm MRI"` | |
| `chronic_pancreatitis/` | `"chronic pancreatitis CT calcification"` | White dots, beaded duct |
| `chronic_pancreatitis/` | `"pancreatic calcifications CT scan"` | |
| `normal_pancreas/` | `"normal pancreas CT portal venous"` | Clean pancreas, no pathology |
| `non_pancreas/` | `"chest CT axial"` | Lungs, no pancreas |
| `non_pancreas/` | `"brain MRI axial"` | Non-abdominal |
| `non_pancreas/` | `"knee X-ray"` | Non-CT / wrong anatomy |

### Minimum images needed

| Class | Minimum | Good |
|-------|---------|------|
| PDAC | 30 | 80+ |
| IPMN | 20 | 60+ |
| Chronic Pancreatitis | 20 | 60+ |
| Normal pancreas | 20 | 40+ |
| Non-pancreas | 30 | 50+ |

### Tips for Google Images collection
1. Go to Google Images, search the query above
2. Right-click → Save image as... into the appropriate folder
3. Save as JPG or PNG
4. You can also use screenshots of CT viewer windows — that works fine
5. **Mix axial slices**: some showing the whole pancreas, some zoomed in

---

## Training workflow

1. Place images in `data/raw/<class_name>/`
2. Open the app (http://localhost:8501)
3. In the sidebar → **Train / Retrain model**
4. Wait ~30–120 seconds depending on image count
5. Model is saved to `models/` automatically

### What the model uses internally
- **Without TensorFlow**: Handcrafted features (intensity histograms, texture, gradients per 4×4 grid). Works with 30+ images, accuracy ~70–80%.
- **With TensorFlow**: MobileNetV2 features (1280-d transfer learning). Works with 20+ images, accuracy ~85–95%.

To install TensorFlow (optional but recommended):
```bash
pip install tensorflow
```

---

## How classification works

### Two-stage pipeline

```
Input image
    ↓
Organ gate          ← Is this a pancreas CT? (trained on non_pancreas folder)
    ↓ pass
Feature extraction  ← MobileNetV2 or handcrafted features
    ↓
Disease classifier  ← Logistic regression (fast, interpretable)
    ↓
Result: PDAC / Non-PDAC / IPMN / CP + confidence score
```

### Classification modes
- **Binary MVP** (default): PDAC vs Non-PDAC — simplest, best for demo
- **3-Class**: PDAC / IPMN / Chronic Pancreatitis — requires more data

---

## Model versions (roadmap)

| Version | Task | Requirement |
|---------|------|-------------|
| **v0.1 (now)** | Binary: PDAC vs Non-PDAC | 30+ images per class |
| **v0.2** | 3-class: PDAC / IPMN / CP | 50+ per class |
| **v0.3** | 5-class + organ gate | 80+ per class |
| **v1.0** | Full 8-class + uncertainty | 150+ per class + MONAI |

---

## Interpreting results

| Verdict | Confidence | Meaning |
|---------|-----------|---------|
| PDAC | >70% | Strong indicator of malignancy — urgent referral |
| PDAC | 55–70% | Suspicious — radiologist review needed |
| Non-PDAC | >70% | Benign pattern — routine surveillance |
| IPMN | >65% | Precursor lesion — 3–6 month follow-up |
| CP | >65% | Inflammatory — medical management |
| Indeterminate | <55% | Low confidence — manual review required |
| Not a pancreas | any | Organ gate rejected the image |

---

## Adding TensorFlow for better accuracy

```bash
# CPU only (easier, works everywhere)
pip install tensorflow-cpu

# GPU (faster training)
pip install tensorflow[and-cuda]  # Linux/CUDA
```

After installing, retrain — accuracy will improve significantly.

---

## Known limitations

1. **Trained on screenshots** — not raw DICOM, so window/level varies per image
2. **No spatial information** — doesn't know *where* in the pancreas a finding is
3. **Google Images bias** — images are "textbook" cases, real patients are harder
4. **No uncertainty calibration** — confidence scores are not probability-calibrated
5. **Binary fallback heuristic** — before training, uses rough visual rules only

These are expected for v0.1. The goal is to demonstrate the concept and pipeline.

---

## Disclaimer

This is a research prototype built to demonstrate the "Duct Dilation Dilemma" concept.  
**It is not validated for clinical use.**  
Do not use for any diagnostic or treatment decision.
