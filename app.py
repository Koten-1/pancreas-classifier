"""
Pancreatic Duct Dilation Classifier
MVP: PDAC vs Non-PDAC binary, expanding to 3-class
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

from utils.classifier import PancreasClassifier
from utils.image_processor import preprocess_image, validate_image
from utils.report import generate_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pancreas CT Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0f0f0f; color: #e8e8e0; }
  .main-title { font-size: 22px; font-weight: 500; color: #e8e8e0; margin-bottom: 4px; }
  .sub-title  { font-size: 13px; color: #888; margin-bottom: 24px; }
  .metric-card {
    background: #1a1a1a; border: 0.5px solid #333; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .metric-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .08em; }
  .metric-value { font-size: 22px; font-weight: 500; color: #e8e8e0; margin-top: 2px; }
  .verdict-pdac   { background:#2a1010; border:0.5px solid #7a2020; border-radius:10px; padding:16px; }
  .verdict-safe   { background:#0d1f12; border:0.5px solid #1a5c2a; border-radius:10px; padding:16px; }
  .verdict-unk    { background:#1c1a0e; border:0.5px solid #5c4f10; border-radius:10px; padding:16px; }
  .verdict-notp   { background:#101828; border:0.5px solid #1a3a5c; border-radius:10px; padding:16px; }
  .tag-high  { background:#2a1010; color:#f09090; border:0.5px solid #7a2020;
               border-radius:20px; padding:3px 10px; font-size:11px; display:inline-block; margin:2px; }
  .tag-med   { background:#1c1a0e; color:#e0c070; border:0.5px solid #5c4f10;
               border-radius:20px; padding:3px 10px; font-size:11px; display:inline-block; margin:2px; }
  .tag-low   { background:#0d1f12; color:#80d0a0; border:0.5px solid #1a5c2a;
               border-radius:20px; padding:3px 10px; font-size:11px; display:inline-block; margin:2px; }
  .step-label { font-size:11px; color:#555; margin-bottom:6px; text-transform:uppercase; letter-spacing:.06em; }
  .divider    { border:none; border-top:0.5px solid #222; margin:16px 0; }
  div[data-testid="stFileUploader"] { background:#161616; border-radius:10px; border:0.5px dashed #333; }
  .stButton>button { background:#1a1a1a; color:#ccc; border:0.5px solid #333; border-radius:8px; font-size:13px; }
  .stButton>button:hover { background:#222; border-color:#555; color:#fff; }
  [data-testid="stSidebar"] { background:#0d0d0d; border-right:0.5px solid #222; }
  .stProgress > div > div { background: #3a6a4a; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Configuration")
    st.markdown("---")

    model_mode = st.selectbox(
        "Classification mode",
        ["Binary: PDAC vs Non-PDAC (MVP)", "3-Class: PDAC / IPMN / CP"],
        index=0,
    )
    binary_mode = "Binary" in model_mode

    st.markdown("---")
    st.markdown("**Training data**")
    data_dir = st.text_input("Data folder path", value="data/raw")
    if st.button("Train / Retrain model"):
        with st.spinner("Training..."):
            clf = PancreasClassifier(binary=binary_mode)
            result = clf.train(data_dir)
            if result["success"]:
                st.success(f"Trained on {result['n_samples']} images — Acc: {result['accuracy']:.2f}")
            else:
                st.error(result["message"])

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
<div style='font-size:12px;color:#555;line-height:1.7'>
Gap: Duct Dilation Dilemma<br>
MPD ≥5mm → correct etiology<br><br>
<b style='color:#666'>Classes (MVP)</b><br>
🔴 PDAC — urgent<br>
🟡 MD-IPMN — surveillance<br>
🟢 Chronic Pancreatitis<br>
⚪ Non-pancreas — invalid<br><br>
v0.1 · Prototype
</div>
""", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Pancreatic CT Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Etiology of pancreatic duct dilation · Research prototype · Not for clinical use</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

# ── Upload panel ──────────────────────────────────────────────────────────────
with col_upload:
    st.markdown('<div class="step-label">Step 1 — Upload CT image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a CT scan image (JPG, PNG, DICOM-screenshot)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True, caption=f"{uploaded.name} · {img.size[0]}×{img.size[1]}px")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="step-label">Step 2 — Run classifier</div>', unsafe_allow_html=True)

        if st.button("▶  Classify image", use_container_width=True):
            with st.spinner("Processing..."):
                # Load or init classifier
                clf = PancreasClassifier(binary=binary_mode)
                clf.load_or_init()

                # Preprocess
                progress = st.progress(0, text="Preprocessing image...")
                img_array = preprocess_image(img)
                time.sleep(0.2)
                progress.progress(30, text="Extracting features...")
                time.sleep(0.2)

                # Classify
                result = clf.predict(img_array, img)
                progress.progress(80, text="Computing confidence...")
                time.sleep(0.1)
                progress.progress(100, text="Done")
                time.sleep(0.2)
                progress.empty()

                st.session_state["result"] = result
                st.session_state["img"] = img
                st.session_state["filename"] = uploaded.name
    else:
        st.markdown("""
<div style='text-align:center;padding:60px 20px;color:#444;font-size:13px;line-height:2'>
    Accepts CT scan screenshots from Google Images<br>
    or any JPEG/PNG of a CT slice<br><br>
    <span style='font-size:11px;color:#333'>
    Pancreas CT · Abdominal CT · Any organ (model will flag non-pancreas)
    </span>
</div>
""", unsafe_allow_html=True)


# ── Results panel ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="step-label">Step 3 — Results</div>', unsafe_allow_html=True)

    if "result" in st.session_state:
        r = st.session_state["result"]
        img_used = st.session_state["img"]

        # ── Pancreas detection gate ──────────────────────────────────────────
        if not r["is_pancreas"]:
            st.markdown(f"""
<div class="verdict-notp">
  <div style='font-size:15px;font-weight:500;color:#7aadda;margin-bottom:6px'>
    ⚠ Not a pancreatic CT
  </div>
  <div style='font-size:13px;color:#6090b0;line-height:1.7'>
    This image does not appear to contain a pancreas or abdominal CT anatomy.<br><br>
    Detected: <b style='color:#9abbd0'>{r["detected_as"]}</b><br>
    Confidence: {r["organ_confidence"]:.0%}
  </div>
</div>
""", unsafe_allow_html=True)
            st.markdown("---")
            st.info("Upload an axial CT image showing the pancreatic region (upper abdomen, portal venous or pancreatic protocol).")

        else:
            # ── Verdict card ─────────────────────────────────────────────────
            verdict = r["verdict"]
            conf    = r["confidence"]

            if verdict == "PDAC":
                card_cls   = "verdict-pdac"
                dot_color  = "#e05050"
                urgency    = "🔴 URGENT — Surgical team referral recommended"
                urge_color = "#e07070"
            elif verdict == "Non-PDAC":
                card_cls   = "verdict-safe"
                dot_color  = "#50c070"
                urgency    = "🟢 Low malignancy risk — Routine surveillance"
                urge_color = "#70d090"
            elif verdict == "IPMN":
                card_cls   = "verdict-unk"
                dot_color  = "#d0a030"
                urgency    = "🟡 Precursor lesion — 3–6 month MRI follow-up"
                urge_color = "#d0c060"
            elif verdict == "CP":
                card_cls   = "verdict-safe"
                dot_color  = "#50a080"
                urgency    = "🟢 Benign etiology — Medical management"
                urge_color = "#60c090"
            else:
                card_cls   = "verdict-unk"
                dot_color  = "#888"
                urgency    = "⚪ Low confidence — Radiologist review needed"
                urge_color = "#aaa"

            st.markdown(f"""
<div class="{card_cls}">
  <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
    <div style='width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0'></div>
    <div style='font-size:18px;font-weight:500;color:#e8e8e0'>{verdict}</div>
    <div style='font-size:22px;font-weight:500;color:{dot_color};margin-left:auto'>{conf:.0%}</div>
  </div>
  <div style='font-size:12px;color:{urge_color};margin-bottom:8px'>{urgency}</div>
  <div style='font-size:12px;color:#666'>{r["explanation"]}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Score bars ───────────────────────────────────────────────────
            st.markdown('<div class="step-label">Class probabilities</div>', unsafe_allow_html=True)
            for cls, score in sorted(r["scores"].items(), key=lambda x: -x[1]):
                pct = int(score * 100)
                bar_colors = {"PDAC":"#c04040","IPMN":"#c09020","CP":"#208050","Non-PDAC":"#3060a0","Indeterminate":"#555"}
                color = bar_colors.get(cls, "#555")
                st.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
  <span style='font-size:12px;color:#888;width:110px;flex-shrink:0'>{cls}</span>
  <div style='flex:1;height:8px;background:#222;border-radius:4px;overflow:hidden'>
    <div style='width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width .5s'></div>
  </div>
  <span style='font-size:12px;color:#888;width:36px;text-align:right;font-family:monospace'>{pct}%</span>
</div>
""", unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Feature analysis ─────────────────────────────────────────────
            st.markdown('<div class="step-label">Visual feature analysis</div>', unsafe_allow_html=True)
            feats = r.get("features", {})
            tags_html = ""
            for fname, fdata in feats.items():
                level = fdata["level"]
                tag_cls = {"high": "tag-high", "med": "tag-med", "low": "tag-low"}.get(level, "tag-low")
                tags_html += f'<span class="{tag_cls}">{fname}: {fdata["label"]}</span>'
            st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Export ───────────────────────────────────────────────────────
            st.markdown('<div class="step-label">Export</div>', unsafe_allow_html=True)
            report_text = generate_report(r, st.session_state.get("filename","unknown"))
            st.download_button(
                "⬇  Download report (.txt)",
                data=report_text,
                file_name=f"pancreas_report_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.markdown("""
<div style='text-align:center;padding:80px 20px;color:#333;font-size:13px'>
    Results will appear here after classification
</div>
""", unsafe_allow_html=True)


# ── Bottom: Data collection helper ───────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Data collection guide — what images to download from Google"):
    st.markdown("""
**Folder structure expected:**
```
data/raw/
├── pdac/          ← Google: "pancreatic cancer CT axial"
├── ipmn/          ← Google: "IPMN pancreas CT main duct"
├── chronic_pancreatitis/  ← Google: "chronic pancreatitis CT calcification"
├── normal_pancreas/       ← Google: "normal pancreas CT"
└── non_pancreas/          ← Google: "chest CT", "brain MRI", "knee X-ray"
                              (these train the organ-gate)
```

**Minimum images to start:**
| Class | Minimum | Notes |
|---|---|---|
| PDAC | 30 | Include solid mass, atrophy, cutoff sign |
| IPMN | 20 | Main-duct type, cystic dilation |
| Chronic Pancreatitis | 20 | Calcifications, beaded duct |
| Normal pancreas | 20 | Pancreas visible, no pathology |
| Non-pancreas | 30 | Any non-abdominal CT/MRI/X-ray |

**Search terms that work well:**
- `"pancreatic adenocarcinoma CT scan axial"`
- `"IPMN main duct MRI CT"`
- `"chronic pancreatitis CT calcification ductal"`
- `"normal pancreas CT portal venous"`
""")
