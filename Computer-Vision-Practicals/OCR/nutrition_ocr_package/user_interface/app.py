import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from models import load_yolo, load_doctr
from pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nutrition OCR",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.main { background: #0d0d0d; }

[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #222;
}

.hero {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;  left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 70% 30%, rgba(34,197,94,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0f0f0;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
}
.hero p {
    color: #666;
    font-size: 1rem;
    margin: 0;
}
.hero .accent { color: #22c55e; }

.step-badge {
    display: inline-block;
    background: #22c55e18;
    color: #22c55e;
    border: 1px solid #22c55e44;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.ocr-block {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #d4d4d4;
    white-space: pre-wrap;
    line-height: 1.7;
    max-height: 320px;
    overflow-y: auto;
}

.meta-pill {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #888;
    margin-right: 6px;
    margin-bottom: 6px;
}
.meta-pill span { color: #22c55e; font-weight: 600; }

.img-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
    text-align: center;
    padding: 6px 0 2px 0;
}

.no-detect {
    background: #1a0a0a;
    border: 1px solid #3a1a1a;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    color: #cc4444;
    font-size: 0.95rem;
}

hr.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    weights_path = st.text_input(
        "YOLO Weights Path",
        value="best.pt",
        help="Path to your trained best.pt file"
    )

    conf_threshold = st.slider(
        "Detection Confidence", 0.1, 0.9, 0.25, 0.05
    )

    st.markdown("---")
    st.markdown("### 📋 Pipeline Steps")
    st.markdown("""
    <div style='color:#666; font-size:0.82rem; line-height:2'>
    ① Upload image<br>
    ② YOLO detects label region<br>
    ③ Crop ROI<br>
    ④ Denoise → CLAHE → Sharpen<br>
    ⑤ docTR orientation fix<br>
    ⑥ OCR text extraction
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#444; font-size:0.75rem'>
    Models load once and are cached.<br>
    First run will download docTR weights.
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Nutrition <span class="accent">OCR</span></h1>
    <p>Upload a nutrition label image — YOLO detects the region, docTR reads the text.</p>
</div>
""", unsafe_allow_html=True)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#333; border:2px dashed #222; border-radius:12px'>
        <div style='font-size:3rem'>📷</div>
        <div style='margin-top:0.5rem; font-size:0.9rem'>Upload a nutrition label image to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load models ───────────────────────────────────────────────────────────────
try:
    yolo_model  = load_yolo(weights_path)
    doctr_model = load_doctr()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.error("Could not read image. Try a different file.")
    st.stop()

with st.spinner("Running pipeline..."):
    full_vis, box_results = run_pipeline(
        image_bgr, yolo_model, doctr_model, conf=conf_threshold
    )


# ── Step 1: Detection result ──────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 — Detection</div>', unsafe_allow_html=True)

col_img, col_info = st.columns([2, 1])
with col_img:
    st.image(full_vis, caption="YOLO detections", use_container_width=True)
with col_info:
    st.markdown(f"""
    <div class="result-card">
        <div style='color:#888; font-size:0.8rem; margin-bottom:1rem'>FILE</div>
        <div style='color:#ddd; font-size:0.9rem; word-break:break-all'>{uploaded.name}</div>
        <div style='color:#888; font-size:0.8rem; margin: 1.2rem 0 0.4rem'>DETECTIONS</div>
        <div style='color:#22c55e; font-size:2.5rem; font-weight:800; line-height:1'>{len(box_results)}</div>
        <div style='color:#555; font-size:0.8rem'>region(s) found</div>
        <div style='margin-top:1.2rem'>
            <span class="meta-pill">conf ≥ <span>{conf_threshold}</span></span>
            <span class="meta-pill">size <span>{image_bgr.shape[1]}×{image_bgr.shape[0]}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not box_results:
    st.markdown("""
    <div class="no-detect">
        No nutrition label regions detected.<br>
        <small>Try lowering the confidence threshold in the sidebar.</small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Per-box results ───────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

for r in box_results:
    st.markdown(
        f'<div class="step-badge">Box {r.box_idx} of {len(box_results)}</div>',
        unsafe_allow_html=True
    )

    # Meta pills
    st.markdown(f"""
    <div style='margin-bottom:1rem'>
        <span class="meta-pill">cls <span>{r.cls_id}</span></span>
        <span class="meta-pill">yolo <span>{r.yolo_conf:.2f}</span></span>
        <span class="meta-pill">ocr conf <span>{r.ocr_conf:.2f}</span></span>
        <span class="meta-pill">angle <span>{r.angle}°</span></span>
    </div>
    """, unsafe_allow_html=True)

    # 4 image columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="img-label">② ROI Crop</div>', unsafe_allow_html=True)
        st.image(r.roi_rgb, use_container_width=True)

    with c2:
        st.markdown('<div class="img-label">③ Denoised</div>', unsafe_allow_html=True)
        st.image(r.enhanced_rgb, use_container_width=True)

    with c3:
        st.markdown('<div class="img-label">④ CLAHE + Sharpen</div>', unsafe_allow_html=True)
        st.image(r.enhanced_rgb, use_container_width=True)

    with c4:
        st.markdown('<div class="img-label">⑤ Straightened</div>', unsafe_allow_html=True)
        st.image(r.straightened_rgb, use_container_width=True)

    # OCR text
    st.markdown('<div style="margin-top:1rem; margin-bottom:0.4rem; color:#555; font-size:0.75rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase">⑥ OCR Output</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ocr-block">{r.ocr_text}</div>',
        unsafe_allow_html=True
    )

    # Download OCR text
    st.download_button(
        label="⬇ Download OCR text",
        data=r.ocr_text,
        file_name=f"ocr_box{r.box_idx}.txt",
        mime="text/plain",
        key=f"dl_{r.box_idx}"
    )

    if r.box_idx < len(box_results):
        st.markdown('<hr class="divider">', unsafe_allow_html=True)