# 🌴 Oil Palm Tree Detection using 7-Channel Multispectral YOLO26

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-green)
![License](https://img.shields.io/badge/License-CC_BY_4.0-orange)

Implementation of **YOLO26** for **7-channel multispectral oil palm tree detection** using drone imagery.

Unlike the standard YOLO model that only accepts **RGB (3-channel)** images, this project modifies the first convolution layer to process **7 spectral bands**:

- Blue
- Green
- Red
- Red Edge
- Red Edge 2
- Near Infrared (NIR)
- Thermal

---

# 📂 Repository Structure

```
.
├── training.py
├── inference_multispectral.py
├── stretch_to_uint8.py
├── data.yaml
├── band_stats.json
├── best.pt
├── yolo26n.pt
└── README.md
```

---

# 📋 Table of Contents

- Background
- Model Architecture
- Dataset
- Installation
- Training
- Inference
- Results
- Output
- Technical Notes

---

# 🎯 Background

RGB imagery alone often cannot distinguish vegetation health effectively.

This project utilizes **7-band multispectral drone imagery**, allowing the model to learn additional spectral information from:

| Band | Description |
|-------|-------------|
| Band 1 | Blue |
| Band 2 | Green |
| Band 3 | Red |
| Band 4 | Red Edge |
| Band 5 | Red Edge 2 |
| Band 6 | Near Infrared (NIR) |
| Band 7 | Thermal |

NIR provides vegetation reflectance information, while Thermal records surface temperature, enabling richer feature extraction compared to conventional RGB imagery.

---

# 🏗 Model Architecture

Default YOLO:

```
Conv [3,16,3,2]
```

Modified YOLO26:

```
Conv [7,16,3,2]
```

The only required modification is adding

```yaml
channels: 7
```

inside `data.yaml`.

Ultralytics automatically rebuilds the first convolution layer while loading pretrained COCO weights.

---

# 📊 Dataset

Dataset characteristics:

| Property | Value |
|----------|-------|
| Sensor | Multispectral Drone |
| Format | GeoTIFF float32 |
| Bands | 7 |
| CRS | EPSG:32748 |
| Tile Size | 640 × 640 |
| Classes | 1 (Oil Palm) |
| Annotation | YOLO Format |
| Total Training Objects | ~1814 |

## 📥 Dataset Download

The dataset used in this project is publicly available on Google Drive:

**https://drive.google.com/drive/folders/1T5lxY6E3U5szevMbUaxcpreaPCh9ei3r?usp=drive_link**

The dataset includes:

- Original multispectral TIFF images
- YOLO annotations
- Train / Validation / Test split
- `band_stats.json`

---

# ⚙ Requirements

- Python 3.12
- CUDA 12.1
- NVIDIA GPU (Recommended)

Libraries

```
ultralytics>=8.4
torch
torchvision
numpy
opencv-python
rasterio
pyyaml
pyshp
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install ultralytics rasterio numpy opencv-python pyshp pyyaml

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA

```bash
python -c "import torch;print(torch.cuda.is_available())"
```

---

# 🚀 Training

Run

```bash
python training.py
```

The best model will be saved as

```
runs/detect/train/weights/best.pt
```

---

# 🔍 Inference

Run

```bash
python inference_multispectral.py
```

Input

```
Path to raster:
```

Output

- Shapefile
- Detection visualization (.jpg)

---

# 📈 Training Results

| Metric | Value |
|---------|--------|
| Model | YOLO26n |
| Epoch | 100 |
| Image Size | 640 |
| Channels | 7 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Precision | 0.760 |
| Recall | 0.856 |
| mAP50 | **0.846** |
| mAP50-95 | **0.373** |

---

# 📤 Output

The inference script produces:

```
detections.shp
detections.dbf
detections.shx
detections.prj
detections.jpg
```

The Shapefile can be opened directly in **QGIS**.

---

# 🔧 Technical Notes

### Why convert float32 to uint8?

Ultralytics currently expects image augmentation and verification to operate on uint8 images.

Each spectral band is stretched using the global percentile statistics stored in

```
band_stats.json
```

ensuring that training and inference use exactly the same normalization.

---

### Why keep `band_stats.json`?

Without global normalization, each tile would have different intensity ranges, reducing model consistency.

---

### Windows Users

When training on Windows:

```python
if __name__ == "__main__":
```

and

```python
workers=0
```

are recommended to avoid multiprocessing issues.

---

# 📄 License

Dataset annotation:

**CC BY 4.0**

---

# 👤 Author

**Pastika**

Computer Engineering

Universitas Diponegoro

---

If you use this project in your research, please cite the repository.
