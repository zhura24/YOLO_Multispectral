# 🌴 Deteksi Pohon Sawit Multispectral dengan YOLO26

Proyek ini mengimplementasikan deteksi pohon sawit menggunakan model **YOLO26** yang dimodifikasi untuk menerima input **7-channel multispectral** dari data drone (RGB + Red Edge + NIR + Thermal), menggantikan input default 3-channel (RGB) pada arsitektur YOLO standar.

---

## 📋 Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Arsitektur Model](#arsitektur-model)
- [Spesifikasi Dataset](#spesifikasi-dataset)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Alur Pipeline Lengkap](#alur-pipeline-lengkap)
- [Script yang Tersedia](#script-yang-tersedia)
- [Hasil Training](#hasil-training)
- [Inference](#inference)
- [Output](#output)
- [Catatan Teknis](#catatan-teknis)

---

## 🎯 Latar Belakang

Deteksi pohon sawit dari citra udara umumnya dilakukan menggunakan citra RGB biasa (3 channel). Proyek ini melangkah lebih jauh dengan memanfaatkan **data multispectral 7-band** dari drone, yang mengandung informasi spektral lebih kaya seperti:

| Band | Informasi |
|------|-----------|
| Band 1 | Blue (reflectance) |
| Band 2 | Green (reflectance) |
| Band 3 | Red (reflectance) |
| Band 4 | Red Edge |
| Band 5 | Red Edge 2 |
| Band 6 | NIR (Near-Infrared) — indikator kesehatan vegetasi |
| Band 7 | Thermal (suhu permukaan, ~27–36°C) |

Informasi NIR dan Thermal tidak tersedia di kamera RGB biasa, sehingga model ini secara teoritis mampu membedakan kondisi vegetasi yang lebih beragam.

---

## 🏗️ Arsitektur Model

Model yang dipakai adalah **YOLO26n** (Ultralytics) dengan modifikasi pada **layer Conv pertama**:

```
# Default YOLO (RGB):
Conv [3, 16, 3, 2]   ← in_channels=3

# Modifikasi proyek ini (Multispectral):
Conv [7, 16, 3, 2]   ← in_channels=7
```

Modifikasi dilakukan cukup dengan menambahkan key `channels: 7` di `data.yaml` — Ultralytics secara otomatis menyesuaikan layer pertama. Transfer learning dari pretrained weights COCO tetap berlaku parsial (606/708 layer dipertahankan).

**Kenapa perlu konversi ke uint8?**

Data drone asli berformat `float32` (nilai reflectance 0.0–1.0 dan thermal 30000-an). Pipeline data loading Ultralytics (validasi Pillow + augmentasi HSV/mosaic) dirancang untuk input `uint8` (0–255). Oleh karena itu, data di-stretch ke uint8 menggunakan statistik global yang tersimpan di `band_stats.json` — konsisten antara training dan inference.

---

## 📊 Spesifikasi Dataset

| Properti | Detail |
|----------|--------|
| Sumber data | Drone multispectral (MicaSense/sejenis) |
| Format asli | GeoTIFF float32, 7-band |
| Resolusi raster | 3521 × 2601 piksel |
| CRS | EPSG:32748 (UTM Zone 48S) |
| Jumlah tile (640×640) | 35 tile |
| Split | Train: 25 / Valid: 7 / Test: 3 |
| Kelas | 1 kelas (`sawit`) |
| Anotasi | Roboflow (bounding box, format YOLOv8) |
| Total instance train | ~1814 bounding box |

---

## ⚙️ Persyaratan

- Python 3.12
- NVIDIA GPU (diuji pada RTX 3050 Ti Laptop, 4GB VRAM)
- CUDA 12.1
- Windows 10/11 (PowerShell)

### Library

```
ultralytics>=8.4.82
torch==2.5.1+cu121
torchvision
rasterio>=1.5.0
numpy>=2.2.0
opencv-python>=4.6.0
pyyaml>=6.0
pyshp>=3.1.0
```

---

## 🚀 Instalasi

```bash
# 1. Clone repository
git clone https://github.com/username/nyawit-multispectral.git
cd nyawit-multispectral

# 2. Buat virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Install library
pip install rasterio numpy pyyaml pyshp ultralytics opencv-python

# 4. Install PyTorch versi CUDA (sesuaikan dengan versi CUDA driver kamu)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Verifikasi GPU terdeteksi
python -c "import torch; print(torch.cuda.is_available())"
# Output: True
```

---

## 🔄 Alur Pipeline Lengkap

```
[1] Raster mentah float32 (7-band, drone)
         ↓
[2] Hitung statistik global per-band → band_stats.json
    (compute_band_stats_single.py)
         ↓
[3] Retile di QGIS → tile 640×640 .tif (float32)
    (Raster Miscellaneous → Retile, tile 640x640, overlap 64px)
         ↓
[4] Anotasi tile PNG di Roboflow → export format YOLOv8
         ↓
[5] Rename & pasangkan label Roboflow ↔ TIFF asli
    (rename_pairs.ps1)
         ↓
[6] Stretch float32 → uint8 CHW + simpan multi-frame TIFF
    (stretch_to_uint8.py)
         ↓
[7] Training YOLO26n (channels: 7, GPU)
    (training.py)
         ↓
[8] Inference raster besar baru (sliding window otomatis)
    (inference_large_raster.py)
         ↓
[9] Output: Shapefile (.shp) + JPG visualisasi
```

---

## 📜 Script yang Tersedia

### 1. `compute_band_stats_single.py`
Menghitung statistik normalisasi global (percentile 1%–99%) dari raster mentah utuh. Output: `band_stats.json`.

```bash
python scripts/compute_band_stats_single.py
# Input: path raster mentah (.tif float32)
# Output: band_stats.json
```

### 2. `rename_pairs.ps1`
Script PowerShell untuk mencocokkan nama file label hasil export Roboflow (yang diberi suffix hash) dengan file TIFF asli berdasarkan nomor tile.

```powershell
# Edit 3 path di dalam script, lalu:
.\scripts\rename_pairs.ps1
```

### 3. `stretch_to_uint8.py`
Mengkonversi TIFF float32 (seluruh split train/valid/test) ke format uint8 CHW yang dibutuhkan Ultralytics, menggunakan statistik dari `band_stats.json`. Juga menghasilkan `data.yaml` baru dengan `channels: 7`.

```bash
python scripts/stretch_to_uint8.py
# Input: folder dataset hasil rename, band_stats.json
# Output: dataset siap training + data.yaml
```

### 4. `training.py`
Training YOLO26n dengan konfigurasi multispectral 7-band menggunakan GPU.

```bash
python scripts/training.py
# Input: path data.yaml
# Output: runs/detect/train-*/weights/best.pt
```

### 5. `inference_large_raster.py`
Inference pada raster besar mentah menggunakan sliding window 640×640 (overlap 64px). Otomatis stretch float32 → uint8, deteksi per tile, NMS antar-tile, dan output Shapefile georeferensi.

```bash
python scripts/inference_large_raster.py
# Input: path raster mentah float32
# Output: deteksi_*.shp (+ .shx, .dbf, .prj) + deteksi_*.jpg
```

### 6. `convert_to_uint8.py`
Mengkonversi raster mentah float32 ke GeoTIFF uint8 untuk keperluan visualisasi di QGIS (bukan untuk inference).

```bash
python scripts/convert_to_uint8.py
# Input: path raster mentah, path output
# Output: GeoTIFF uint8 siap dibuka di QGIS
```

---

## 📈 Hasil Training

| Metrik | Nilai |
|--------|-------|
| Model | YOLO26n |
| Epochs | 100 |
| Input channels | 7 (multispectral) |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW (lr=0.002) |
| **mAP50** | **0.846** |
| **mAP50-95** | **0.373** |
| Precision | 0.760 |
| Recall | 0.856 |
| Training time | ~3.2 menit (RTX 3050 Ti) |

---

## 🔍 Inference

Script inference mendukung raster dengan **resolusi berapapun** secara otomatis menggunakan sliding window:

```bash
python scripts/inference_large_raster.py
```

```
Path ke raster MENTAH UTUH (.tif): C:\path\ke\raster_baru.tif
Confidence threshold (Enter untuk default 0.25):
```

**Catatan penting:**
- Input harus raster **float32 mentah** dari sensor yang sama (7-band, urutan band sama)
- `band_stats.json` harus sama persis dengan yang dipakai saat training
- Untuk sensor berbeda, perlu menghitung ulang `band_stats.json` dan melatih ulang model

---

## 📤 Output Inference

### Shapefile (`.shp`)
Bounding box setiap pohon sawit dalam koordinat geografis asli (UTM), dapat dibuka langsung di QGIS:

| Kolom | Isi |
|-------|-----|
| `id` | Nomor urut deteksi |
| `kelas` | Nama kelas (`sawit`) |
| `confidence` | Skor kepercayaan model (0–1) |
| `x1_px`, `y1_px` | Koordinat piksel pojok kiri atas |
| `x2_px`, `y2_px` | Koordinat piksel pojok kanan bawah |

**Cara buka di QGIS:**
```
Layer → Add Layer → Add Vector Layer → pilih file .shp
```

### Gambar Visualisasi (`.jpg`)
Komposit RGB (band 1-2-3) dengan bounding box hijau overlay di setiap deteksi.

---

## 🔧 Catatan Teknis

### Kenapa uint8, bukan float32 langsung?
Pipeline data loading Ultralytics (validasi Pillow + augmentasi HSV/mosaic) di-hardcode untuk input uint8. Percobaan langsung dengan float32 menghasilkan error `More samples per pixel than can be decoded: 7` di tahap verifikasi dataset. Stretch ke uint8 menggunakan statistik global (`band_stats.json`) memastikan konsistensi skala antara training dan inference tanpa kehilangan informasi spektral yang signifikan untuk task deteksi objek.

### Kenapa `band_stats.json` harus disimpan?
File ini berisi referensi normalisasi global (percentile 1%–99% per-band) dari seluruh dataset. Tanpa file ini, setiap tile dinormalisasi secara independen sehingga skala piksel tidak konsisten antar tile — model akan "bingung" karena warna yang sama bisa punya nilai berbeda di tile yang berbeda.

### Format TIFF multi-frame vs band-stacking
Ultralytics membaca TIFF multispectral menggunakan `cv2.imreadmulti` (multi-frame/multi-page), bukan band-stacking standar `rasterio`. Oleh karena itu penyimpanan tile training menggunakan `cv2.imwritemulti`, bukan `rasterio.write()`.

### Windows-specific
Pada Windows, training **wajib** menggunakan:
- `if __name__ == "__main__":` — mencegah crash multiprocessing DataLoader
- `workers=0` — mencegah konflik shared-memory pada Windows

---

## 📄 Lisensi

Dataset anotasi: CC BY 4.0 (via Roboflow)

---

## 👤 Author

**Pastika** — Teknik Komputer, Universitas Diponegoro
