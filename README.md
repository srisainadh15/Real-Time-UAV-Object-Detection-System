<div align="center">

<h1>🛸 Real-Time UAV Object Detection System</h1>
<h3>SFD-YOLOv8 · VisDrone2019 · Deep Learning</h3>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.21-00C4CC?style=flat-square)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-VisDrone2019-8B5CF6?style=flat-square)](https://github.com/VisDrone/VisDrone-Dataset)
[![GPU](https://img.shields.io/badge/GPU-RTX_2000_Ada_16GB-76B900?style=flat-square&logo=nvidia)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📌 What This Project Does

Standard object detectors like YOLOv8 are designed for ground-level photos where objects are large and clearly visible. When applied to **drone/UAV footage**, detection quality drops significantly because:

- A person captured from 80 m altitude may be only **8 × 12 pixels** in the image
- A single drone frame can contain **100+ tiny objects** at once
- Objects change size dramatically as the drone changes altitude
- Dense urban backgrounds create heavy background clutter

This project builds **SFD-YOLOv8** — a modified version of YOLOv8n — trained on the VisDrone2019 benchmark to solve these problems. The model detects 10 object categories in UAV imagery faster and more accurately than the original YOLOv8n baseline, while using **fewer parameters**.

---

## 📦 Dataset — VisDrone2019

> 🔗 **Download:** https://github.com/VisDrone/VisDrone-Dataset
> 📄 **Paper:** https://arxiv.org/abs/1905.10914

VisDrone2019 is the standard academic benchmark for UAV object detection. Collected across **14 cities in China** at altitudes from **1 m to 140 m** using multiple drone platforms.

### Dataset Size

| Split | Images | Object Instances |
|-------|:------:|:---------------:|
| Train | 6,471 | 343,204 |
| **Validation** | **548** | **38,759** |
| Test | 1,610 | — |
| **Total** | **8,629** | **~400,000** |

### 10 Object Classes

| ID | Class | Val Instances | Description |
|----|-------|:-------------:|-------------|
| 0 | pedestrian | 8,844 | Single walking person |
| 1 | people | 5,125 | Group of people standing together |
| 2 | bicycle | 1,287 | Bicycle with or without rider |
| 3 | **car** | **14,064** | Passenger car — most common class |
| 4 | van | 1,975 | Van or minibus |
| 5 | truck | 750 | Large goods truck |
| 6 | tricycle | 1,045 | Three-wheeled vehicle |
| 7 | awning-tricycle | 532 | Tricycle with overhead shade |
| 8 | bus | 251 | Full-size bus — rarest class |
| 9 | motor | 4,886 | Motorcycle or scooter |

### Why VisDrone is Challenging

```
Average object size  →  22 × 19 pixels   (a face is invisible at this scale)
Smallest objects     →  as small as 4 × 5 pixels
Class imbalance      →  car (14,064) vs bus (251)  =  56× difference
Occlusion            →  ~60% of instances partially hidden
```

---

## 🏗️ How SFD-YOLOv8 Differs from Standard YOLOv8

### Standard YOLOv8n Pipeline

```
Input 640×640
      │
  BACKBONE  (CSP + standard C2f blocks)    ← extracts image features
      │
   NECK     (PAN — 3-scale fusion)         ← merges features at different scales
      │
   HEAD     (P3 80×80, P4 40×40, P5 20×20) ← 3 detection outputs
      │
  Predictions
```

The smallest detection scale P3 has **stride 8** — each cell covers an 8×8 pixel area.
Objects smaller than ~8 px are nearly impossible to detect here.

---

### SFD-YOLOv8 — Our Proposed Architecture

<div align="center">
  <img src="images/architecture.png" alt="SFD-YOLOv8 Architecture" width="88%"/>
  <br/>
  <sub><i>Orange blocks = new or modified components. Blue blocks = kept from baseline YOLOv8.</i></sub>
</div>

<br/>

We made **3 structural changes** to solve the small-object detection problem:

---

#### 🔶 Change 1 — C2f_DWR Blocks in Backbone

Standard C2f uses a single 3×3 convolution that sees only a small local region.
It cannot simultaneously understand the object detail and the wider scene context.

**C2f_DWR (Dilation-Wise Residual)** runs 3 parallel convolutions with different dilation rates:

```
Dilation rate 1  →  sees  3×3 region   (fine detail — the object itself)
Dilation rate 2  →  sees  7×7 region   (nearby context — immediate surroundings)
Dilation rate 3  →  sees 13×13 region  (wide context — scene understanding)

All 3 outputs → combined → sigmoid gate → attended feature map
```

This helps the model understand that a tiny moving dot surrounded by a road is a pedestrian, even when only 8 pixels tall.

---

#### 🔶 Change 2 — C2f_FasterBlk in Neck

The PANet neck merges features from different backbone levels using C2f blocks.
Standard C2f applies Conv3×3 to all channels — expensive and redundant.

**FasterBlock** applies Conv3×3 to only **¼ of channels**. The rest pass through unchanged. A 1×1 conv mixes everything at the end:

```
Standard C2f:   ALL 256 channels → Conv3×3             (full cost)

FasterBlock:     64 channels     → Conv3×3             (computed)
                192 channels     → Identity (no-op)    (free)
                Both             → Concat → Conv1×1
```

Result: fewer parameters and faster inference with no accuracy drop.

---

#### 🔶 Change 3 — Extra P2 Detection Head at 160×160

This is the most impactful change. Standard YOLOv8 uses 3 detection scales:

| Head | Resolution | Stride | Each cell covers |
|------|:----------:|:------:|:----------------:|
| P3 | 80×80 | 8 px | 8×8 input pixels |
| P4 | 40×40 | 16 px | 16×16 input pixels |
| P5 | 20×20 | 32 px | 32×32 input pixels |

SFD-YOLOv8 adds a **4th head at P2**:

| Head | Resolution | Stride | Each cell covers |
|------|:----------:|:------:|:----------------:|
| **P2 ← NEW** | **160×160** | **4 px** | **4×4 input pixels** |
| P3 | 80×80 | 8 px | 8×8 input pixels |
| P4 | 40×40 | 16 px | 16×16 input pixels |
| P5 | 20×20 | 32 px | 32×32 input pixels |

At P2 stride-4, a 6-pixel pedestrian spans ~1.5 detection cells — enough to detect it.
At P3 stride-8, the same pedestrian is less than 1 cell and is typically missed entirely.

---

### Changes at a Glance

| Component | Location | Change | Benefit |
|-----------|----------|--------|---------|
| C2f_DWR | Backbone | Multi-dilation parallel convs (r=1,2,3) | Rich context for tiny objects |
| C2f_FasterBlk | Neck | Partial conv on ¼ channels only | Fewer parameters, faster speed |
| P2 Head | Detection Head | Extra 160×160 detection scale | Catches objects smaller than 10 px |

---

## 📊 Results

Evaluated on **VisDrone2019 validation set** — 548 images, 38,759 instances.

```
Hardware  :  NVIDIA RTX 2000 Ada Generation  (16 GB VRAM)
Python    :  3.10.19
PyTorch   :  2.6.0+cu124
Ultralytics: 8.4.21
```

### Overall Metrics

| Metric | Score |
|--------|:-----:|
| **mAP@0.5** | **38.33%** |
| **mAP@0.5:0.95** | **22.68%** |
| **Precision** | **49.21%** |
| **Recall** | **37.00%** |
| Inference speed | **3.7 ms / image** |
| Preprocess | 1.0 ms |
| Postprocess | 6.2 ms |

### Per-Class Results

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|:---------:|:------:|:-------:|:------------:|
| pedestrian | 0.596 | 0.383 | 0.448 | 0.204 |
| people | 0.572 | 0.319 | 0.373 | 0.146 |
| bicycle | 0.284 | 0.117 | 0.119 | 0.048 |
| **car** | **0.708** | **0.795** | **0.814** | **0.571** |
| van | 0.528 | 0.413 | 0.434 | 0.307 |
| truck | 0.427 | 0.315 | 0.301 | 0.202 |
| tricycle | 0.374 | 0.252 | 0.232 | 0.132 |
| awning-tricycle | 0.246 | 0.177 | 0.139 | 0.091 |
| bus | 0.653 | 0.478 | 0.518 | 0.369 |
| motor | 0.533 | 0.453 | 0.455 | 0.198 |
| **Overall** | **0.492** | **0.370** | **0.383** | **0.227** |

**Best class:** `car` — mAP@0.5 = **81.4%**
Cars are large, frequent (14,064 instances), and visually distinct from above.

**Hardest class:** `awning-tricycle` — mAP@0.5 = **13.9%**
Rare in training data, very small, and visually similar to regular tricycles.

---

## 🖼️ Detection Output Samples

<table>
<tr>
<td align="center"><img src="images/output1.jpg" width="420"/><br/><sub>Dense urban scene</sub></td>
<td align="center"><img src="images/output2.jpg" width="420"/><br/><sub>Mixed traffic scene</sub></td>
</tr>
<tr>
<td align="center"><img src="images/output3.jpg" width="420"/><br/><sub>High altitude — small objects</sub></td>
<td align="center"><img src="images/output4.jpg" width="420"/><br/><sub>Crowded intersection</sub></td>
</tr>
</table>

---

## 🌐 Streamlit Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

---

## 📓 Notebook

Full pipeline — **[`notebooks/sfd.ipynb`](notebooks/sfd.ipynb)**

Covers: environment setup → dataset preparation → model training → evaluation → inference → export

---

## ⬇️ Model Weights

Download `best.pt` from the **[Releases](../../releases/latest)** tab and place it in the project root folder.

---

## 📚 References

| Resource | Link |
|----------|------|
| VisDrone2019 Dataset | https://github.com/VisDrone/VisDrone-Dataset |
| VisDrone Paper | https://arxiv.org/abs/1905.10914 |
| YOLOv8 (Ultralytics) | https://github.com/ultralytics/ultralytics |
| FasterNet Paper | https://arxiv.org/abs/2303.03667 |
| DWRSeg Paper | https://arxiv.org/abs/2212.01173 |

---

<div align="center">
<sub>Real-Time UAV Object Detection · SFD-YOLOv8 · VisDrone2019 · SIC AI Internship 2025</sub>
</div>
