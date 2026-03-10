import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import tempfile
import os
import torch
from pathlib import Path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAV Object Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;900&display=swap');

:root {
    --bg: #060810;
    --surface: #0d1117;
    --surface2: #111827;
    --border: #1f2937;
    --border2: #374151;
    --accent: #6ee7b7;
    --accent2: #818cf8;
    --accent3: #f472b6;
    --text: #e2e8f0;
    --muted: #4b5563;
    --white: #ffffff;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

.hero {
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 90% 50%, rgba(110,231,183,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 10% 20%, rgba(129,140,248,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
}
.hero-title {
    font-size: clamp(22px, 3vw, 34px);
    font-weight: 900;
    color: var(--white);
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
}

.det-wrap {
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-top: 8px;
}
.det-table { width: 100%; border-collapse: collapse; }
.det-table th {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
.det-table td {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--text);
    padding: 9px 14px;
    border-bottom: 1px solid rgba(31,41,55,0.6);
    white-space: nowrap;
}
.det-table tr:last-child td { border-bottom: none; }
.det-table tr:hover td { background: rgba(255,255,255,0.02); }

.conf-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(110,231,183,0.1);
    color: var(--accent);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
}
.cls-tag {
    display: inline-block;
    background: rgba(129,140,248,0.1);
    color: var(--accent2);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 11px;
}

.sum-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}
.sum-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.sum-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
}
.sum-value { font-size: 24px; font-weight: 700; color: var(--white); }
.sum-value.green  { color: var(--accent); }
.sum-value.indigo { color: var(--accent2); }

.class-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.cbadge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border2);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--text);
}
.cbadge b { color: var(--accent); }

.upload-zone {
    border: 2px dashed var(--border2);
    border-radius: 16px;
    padding: 72px 40px;
    text-align: center;
    background: var(--surface);
}
.upload-icon { font-size: 52px; margin-bottom: 18px; }
.upload-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 8px;
}
.upload-hint {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.8;
}

.stButton > button {
    background: var(--accent) !important;
    color: #060810 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #a7f3d0 !important;
    box-shadow: 0 4px 20px rgba(110,231,183,0.3) !important;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.ok  { background: var(--accent);  box-shadow: 0 0 6px var(--accent); }
.status-dot.err { background: #f87171; box-shadow: 0 0 6px #f87171; }

.info-box {
    background: rgba(129,140,248,0.06);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--accent2);
    margin-top: 12px;
    line-height: 1.7;
}

.meta {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
}

.timing-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(244,114,182,0.08);
    border: 1px solid rgba(244,114,182,0.2);
    border-radius: 999px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--accent3);
    margin-top: 8px;
}

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid rgba(110,231,183,0.4) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(110,231,183,0.08) !important;
    border-color: var(--accent) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }
div[data-testid="stFileUploader"] {
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]
DEFAULT_MODEL_PATH = "best.pt"


# ─── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        model = YOLO(path)
    finally:
        torch.load = original_load
    return model


# ─── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, uploaded_bytes: bytes, conf_thresh: float, iou_thresh: float):
    if uploaded_bytes[:3] == b'\xff\xd8\xff':
        ext = ".jpg"
    elif uploaded_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        ext = ".png"
    elif uploaded_bytes[:2] == b'BM':
        ext = ".bmp"
    else:
        ext = ".png"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        os.close(tmp_fd)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_bytes)
        t0 = time.perf_counter()
        results = model.predict(
            source=tmp_path, conf=conf_thresh, iou=iou_thresh,
            verbose=False, save=False, imgsz=640, augment=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return results[0], elapsed


# ─── Draw detections ───────────────────────────────────────────────────────────
def draw_detections(image_bgr, result, box_thickness=2, show_labels=True):
    img = image_bgr.copy()
    detections = []
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            conf   = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (110, 231, 183), box_thickness)

            if show_labels:
                text = f"{label} {conf:.2f}"
                font, fscale = cv2.FONT_HERSHEY_SIMPLEX, 0.48
                (tw, th), _ = cv2.getTextSize(text, font, fscale, 1)
                pad = 4
                ty1 = max(0, y1 - th - 2 * pad)
                cv2.rectangle(img, (x1, ty1), (x1 + tw + 2 * pad, y1), (110, 231, 183), -1)
                cv2.putText(img, text, (x1 + pad, max(th + pad, y1 - pad)),
                            font, fscale, (6, 8, 16), 1, cv2.LINE_AA)

            detections.append({
                "class": label, "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "area": (x2 - x1) * (y2 - y1),
            })
    return img, detections


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 24px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4b5563;
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
            Configuration
        </div>
        <div style="font-size:18px;font-weight:700;color:#fff;line-height:1.3;">
            UAV Detection
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Model Path**")
    model_path = st.text_input(
        "model_path", value=DEFAULT_MODEL_PATH,
        label_visibility="collapsed",
        placeholder="e.g. best.pt",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Thresholds**")
    conf_thresh = st.slider("Confidence", 0.01, 0.95, 0.10, 0.01)
    iou_thresh  = st.slider("IoU (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Visualisation**")
    box_thickness = st.slider("Box thickness", 1, 5, 2)
    show_labels   = st.checkbox("Show labels", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Classes**")
    all_cls = st.checkbox("All classes", value=True)
    filter_classes = CLASS_NAMES
    if not all_cls:
        filter_classes = st.multiselect(
            "filter_classes", CLASS_NAMES, default=CLASS_NAMES,
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    model_exists = Path(model_path).exists()
    dot_cls  = "ok"  if model_exists else "err"
    stat_txt = "Model loaded" if model_exists else "Model not found"
    stat_col = "#6ee7b7" if model_exists else "#f87171"
    st.markdown(
        f'<span class="status-dot {dot_cls}"></span>'
        f'<span style="font-family:DM Mono,monospace;font-size:11px;color:{stat_col};">'
        f'{stat_txt}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box" style="margin-top:20px;">
        Dataset &nbsp;·&nbsp; VisDrone2019<br>
        Classes &nbsp;·&nbsp; 10 categories<br>
        Backbone · SFD-YOLOv8n
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🛸 Deep Learning · Computer Vision</div>
    <div class="hero-title">Real-Time <em>UAV Object Detection</em> System</div>
    <div class="hero-sub">
        SFD-YOLOv8 &nbsp;·&nbsp; DWR Attention &nbsp;·&nbsp;
        FasterBlock &nbsp;·&nbsp; Focal-EIoU &nbsp;·&nbsp; 4-Scale Detection Head
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
model = None
if model_exists:
    with st.spinner("Loading model weights…"):
        try:
            model = load_model(model_path)
            st.success("✅ Model ready")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
else:
    st.warning(
        f"⚠️ `{model_path}` not found. "
        "Place `best.pt` next to `app.py`, or update the path in the sidebar."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOADER
# ═══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Upload a UAV image — JPG / PNG / BMP / WEBP",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
    raw_bytes = uploaded.read()
    file_arr  = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr   = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w      = img_bgr.shape[:2]

    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Input Image</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
        st.markdown(
            f'<div class="meta">{uploaded.name} &nbsp;·&nbsp; {w}×{h} px'
            f' &nbsp;·&nbsp; {len(raw_bytes)/1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Detection Result</div>', unsafe_allow_html=True)

        detections = []
        out_rgb    = None

        if model is None:
            st.info("Load a valid model to see detections.")
        else:
            with st.spinner("Running inference…"):
                try:
                    result, elapsed_ms = run_inference(
                        model, raw_bytes, conf_thresh, iou_thresh
                    )

                    if not all_cls and filter_classes and result.boxes is not None:
                        keep_ids = [
                            CLASS_NAMES.index(c)
                            for c in filter_classes if c in CLASS_NAMES
                        ]
                        mask = [int(b.cls[0].cpu()) in keep_ids for b in result.boxes]
                        result.boxes = result.boxes[mask] if any(mask) else result.boxes[[]]

                    out_bgr, detections = draw_detections(
                        img_bgr, result,
                        box_thickness=box_thickness,
                        show_labels=show_labels,
                    )
                    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    st.image(out_rgb, use_container_width=True)

                    n = len(detections)
                    st.markdown(
                        f'<div class="timing-chip">⏱ {elapsed_ms:.1f} ms &nbsp;·&nbsp; '
                        f'{n} detection{"s" if n != 1 else ""} &nbsp;·&nbsp; '
                        f'conf ≥ {conf_thresh:.2f}</div>',
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.error(f"Inference error: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Detection Details ─────────────────────────────────────────────────────
    if detections:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Detection Details</div>', unsafe_allow_html=True)

        avg_conf = float(np.mean([d["confidence"] for d in detections]))
        class_counts: dict = {}
        for d in detections:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

        st.markdown(f"""
        <div class="sum-grid">
            <div class="sum-card">
                <div class="sum-label">Detections</div>
                <div class="sum-value green">{len(detections)}</div>
            </div>
            <div class="sum-card">
                <div class="sum-label">Avg Confidence</div>
                <div class="sum-value indigo">{avg_conf:.1%}</div>
            </div>
            <div class="sum-card">
                <div class="sum-label">Classes Found</div>
                <div class="sum-value">{len(class_counts)}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        badges = "".join(
            f'<span class="cbadge">{cls} <b>{cnt}</b></span>'
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
        )
        st.markdown(f'<div class="class-row">{badges}</div>', unsafe_allow_html=True)

        rows = ""
        for i, d in enumerate(sorted(detections, key=lambda x: -x["confidence"])):
            x1, y1, x2, y2 = d["bbox"]
            rows += f"""
            <tr>
                <td style="color:#4b5563;">{i+1}</td>
                <td><span class="cls-tag">{d['class']}</span></td>
                <td><span class="conf-badge">{d['confidence']:.3f}</span></td>
                <td>{x1}, {y1}</td>
                <td>{x2}, {y2}</td>
                <td>{d['area']:,} px²</td>
            </tr>"""

        st.markdown(f"""
        <div class="det-wrap">
        <table class="det-table">
            <thead><tr>
                <th>#</th><th>Class</th><th>Confidence</th>
                <th>Top-Left</th><th>Bottom-Right</th><th>Area</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

        if out_rgb is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            Image.fromarray(out_rgb).save(buf, format="PNG")
            st.download_button(
                label="⬇  Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"det_{uploaded.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
            )

        st.markdown('</div>', unsafe_allow_html=True)

    elif model is not None:
        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
            No detections at current threshold — try lowering the
            <b>Confidence</b> slider in the sidebar.
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🛸</div>
        <div class="upload-title">Drop a UAV Image to Begin</div>
        <div class="upload-hint">
            JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; BMP &nbsp;·&nbsp; WEBP<br>
            Detections rendered with bounding boxes &amp; confidence scores
        </div>
    </div>""", unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:56px;padding-top:20px;border-top:1px solid #1f2937;
            font-family:'DM Mono',monospace;font-size:11px;color:#1f2937;text-align:center;">
    Real-Time UAV Object Detection System &nbsp;·&nbsp; SFD-YOLOv8 &nbsp;·&nbsp; VisDrone2019
</div>""", unsafe_allow_html=True)
