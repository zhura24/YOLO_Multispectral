import sys
import json
from pathlib import Path

try:
    import cv2
    import numpy as np
    import rasterio
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Modul belum terinstall: {e}")
    print("Jalankan: pip install ultralytics rasterio numpy opencv-python")
    sys.exit(1)


def load_band_stats(stats_path: Path):
    with open(stats_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def stretch_band(band: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    band = band.astype(np.float64)
    if p_high - p_low == 0:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low) * 255.0
    return scaled.astype(np.uint8)


def generate_tile_windows(width: int, height: int, tile_size: int = 640, overlap: int = 64):
    """
    Hasilkan daftar window (x_off, y_off, w, h) untuk sliding window
    sepanjang raster, dengan overlap antar tile.
    """
    stride = tile_size - overlap
    windows = []

    y = 0
    while y < height:
        x = 0
        h = min(tile_size, height - y)
        while x < width:
            w = min(tile_size, width - x)
            windows.append((x, y, w, h))
            if x + tile_size >= width:
                break
            x += stride
        if y + tile_size >= height:
            break
        y += stride

    return windows


def nms_global(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """
    NMS sederhana untuk menghapus deteksi duplikat di area overlap antar tile.
    boxes: array (N, 4) format [x1, y1, x2, y2] dalam koordinat raster asli.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return keep


MODEL_PATH = r"C:\Users\user\Downloads\Compressed\nyawit multispectral\runs\detect\train-3\weights\best.pt"
BAND_STATS_PATH = r"C:\Users\user\Downloads\Compressed\nyawit multispectral\band_stats.json"


def draw_detections_on_raster(raster_path: Path, boxes: np.ndarray, scores: np.ndarray,
                                out_path: Path, max_dim: int = 4000):
    """
    Buat gambar visual: composite RGB (band 1-3) dari raster asli + bounding box
    hasil deteksi digambar di atasnya. Disimpan sebagai PNG/JPG biasa supaya
    mudah dibuka di image viewer manapun (tidak perlu QGIS).
    """
    with rasterio.open(raster_path) as src:
        # Pakai band 1-3 sebagai komposit RGB untuk visualisasi
        # (band asli reflectance, distretch ke 0-255 hanya untuk tampilan)
        r = src.read(3)  # band 3 kira-kira Red
        g = src.read(2)  # band 2 kira-kira Green
        b = src.read(1)  # band 1 kira-kira Blue

    def stretch_for_display(band):
        p2, p98 = np.percentile(band, (2, 98))
        band = np.clip(band, p2, p98)
        return ((band - p2) / (p98 - p2) * 255).astype(np.uint8)

    rgb = np.stack([stretch_for_display(r), stretch_for_display(g), stretch_for_display(b)], axis=-1)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV pakai BGR

    # Gambar bounding box
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(rgb_bgr, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Resize kalau raster sangat besar, biar file tidak terlalu berat & mudah dibuka
    h, w = rgb_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        rgb_bgr = cv2.resize(rgb_bgr, (int(w * scale), int(h * scale)))

    cv2.imwrite(str(out_path), rgb_bgr)
    return out_path


def main():
    print("=== Inference Raster Besar Mentah (Sliding Window) - Deteksi Sawit ===\n")

    model_path = Path(MODEL_PATH)
    stats_path = Path(BAND_STATS_PATH)
    raster_path = Path(input("Path ke raster MENTAH UTUH (.tif): ").strip().strip('"'))

    if not model_path.is_file():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)
    if not stats_path.is_file():
        print(f"[ERROR] band_stats.json tidak ditemukan: {stats_path}")
        sys.exit(1)
    if not raster_path.is_file():
        print(f"[ERROR] Raster tidak ditemukan: {raster_path}")
        sys.exit(1)

    tile_size = 640
    overlap = 64

    conf_raw = input("Confidence threshold (Enter untuk default 0.25): ").strip()
    conf = float(conf_raw) if conf_raw else 0.25

    print("\nMemuat model...")
    model = YOLO(str(model_path))
    band_stats = load_band_stats(stats_path)

    print(f"\nMembuka raster: {raster_path.name}")
    with rasterio.open(raster_path) as src:
        width, height, n_bands = src.width, src.height, src.count
        print(f"Ukuran raster: {width} x {height} piksel, {n_bands} band")

        expected_n_bands = len(band_stats)
        if n_bands != expected_n_bands:
            print(f"[ERROR] Jumlah band raster ({n_bands}) tidak sama dengan jumlah band "
                  f"yang dipakai saat training ({expected_n_bands}).")
            print(f"        Model ini membutuhkan TEPAT {expected_n_bands} band, bukan {n_bands}.")
            print(f"        Pastikan kamu memakai raster 7-band yang utuh (bukan hasil split/RGB-only).")
            sys.exit(1)

        missing_bands = [b for b in range(1, n_bands + 1) if b not in band_stats]
        if missing_bands:
            print(f"[ERROR] band_stats.json tidak punya statistik untuk band: {missing_bands}")
            print(f"        band_stats.json yang dipakai punya statistik untuk band: {sorted(band_stats.keys())}")
            print("        Pastikan raster ini punya jumlah & urutan band yang SAMA dengan data training.")
            sys.exit(1)

        windows = generate_tile_windows(width, height, tile_size, overlap)
        print(f"Akan diproses sebagai {len(windows)} tile ({tile_size}x{tile_size}, overlap {overlap}px)\n")

        all_boxes = []
        all_scores = []
        all_classes = []

        for idx, (x_off, y_off, w, h) in enumerate(windows, start=1):
            print(f"[{idx}/{len(windows)}] Tile di posisi ({x_off}, {y_off}), ukuran {w}x{h} ...", end=" ")

            # Baca tile dari window raster
            bands_out = []
            for b in range(1, n_bands + 1):
                window = rasterio.windows.Window(x_off, y_off, w, h)
                data = src.read(b, window=window)
                stats = band_stats.get(b)
                bands_out.append(stretch_band(data, stats["p_low"], stats["p_high"]))

            # Susun array (C, H, W) uint8 langsung di memory, tanpa simpan ke disk
            tile_chw = np.stack(bands_out, axis=0)  # shape: (7, H, W)

            # Ultralytics model.predict() bisa menerima numpy array (H, W, C)
            # Kita kirim langsung tanpa tulis file ke disk
            tile_hwc = tile_chw.transpose(1, 2, 0)  # (H, W, 7)

            # Deteksi pada tile ini (langsung dari array, no disk I/O)
            results = model.predict(source=tile_hwc, conf=conf, save=False, verbose=False)

            n_det = 0
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # koordinat lokal tile
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()

                    # Konversi ke koordinat raster ASLI (tambahkan offset tile)
                    boxes_xyxy[:, [0, 2]] += x_off
                    boxes_xyxy[:, [1, 3]] += y_off

                    all_boxes.append(boxes_xyxy)
                    all_scores.append(scores)
                    all_classes.append(classes)
                    n_det = len(scores)

            print(f"{n_det} objek")

    if not all_boxes:
        print("\n[INFO] Tidak ada objek terdeteksi di seluruh raster.")
        return

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    print(f"\nTotal deteksi sebelum NMS antar-tile: {len(all_boxes)}")

    keep_idx = nms_global(all_boxes, all_scores, iou_threshold=0.5)
    final_boxes = all_boxes[keep_idx]
    final_scores = all_scores[keep_idx]
    final_classes = all_classes[keep_idx]

    print(f"Total deteksi setelah NMS antar-tile: {len(final_boxes)}")
    print(f"Confidence rata-rata: {final_scores.mean():.3f}, "
          f"min: {final_scores.min():.3f}, max: {final_scores.max():.3f}")

    # Simpan hasil sebagai Shapefile (.shp) dengan koordinat geografis asli
    # sehingga bisa langsung dibuka di QGIS dan ter-overlay di posisi yang benar
    print("\nMenyimpan hasil sebagai Shapefile (.shp)...")
    try:
        import shapefile  # library pyshp
    except ImportError:
        print("[ERROR] Library pyshp belum terinstall.")
        print("Jalankan: pip install pyshp")
        sys.exit(1)

    # Baca georeferensi dari raster asli
    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None

    out_shp = Path(f"deteksi_{raster_path.stem}.shp")

    with shapefile.Writer(str(out_shp), shapeType=shapefile.POLYGON) as shp:
        # Definisi kolom atribut
        shp.field("id",         "N", size=10)
        shp.field("kelas",      "C", size=20)
        shp.field("confidence", "N", size=10, decimal=4)
        shp.field("x1_px",     "N", size=10, decimal=1)
        shp.field("y1_px",     "N", size=10, decimal=1)
        shp.field("x2_px",     "N", size=10, decimal=1)
        shp.field("y2_px",     "N", size=10, decimal=1)

        for i, (cls, score, box) in enumerate(
                zip(final_classes, final_scores, final_boxes), start=1):

            x1_px, y1_px, x2_px, y2_px = box

            # Konversi koordinat piksel → koordinat geografis (meter/derajat)
            # pakai transform raster asli supaya posisi di QGIS tepat
            x1_geo, y1_geo = rasterio.transform.xy(raster_transform, y1_px, x1_px)
            x2_geo, y2_geo = rasterio.transform.xy(raster_transform, y2_px, x2_px)

            # Bounding box sebagai polygon (searah jarum jam)
            polygon = [
                [x1_geo, y1_geo],
                [x2_geo, y1_geo],
                [x2_geo, y2_geo],
                [x1_geo, y2_geo],
                [x1_geo, y1_geo],  # tutup polygon
            ]

            shp.poly([polygon])
            shp.record(
                i,
                "sawit",
                round(float(score), 4),
                round(float(x1_px), 1),
                round(float(y1_px), 1),
                round(float(x2_px), 1),
                round(float(y2_px), 1),
            )

    # Simpan file .prj (proyeksi) supaya QGIS tahu sistem koordinatnya
    if crs_wkt:
        prj_path = out_shp.with_suffix(".prj")
        with open(prj_path, "w") as prj:
            prj.write(crs_wkt)

    print(f"Shapefile disimpan di: {out_shp.resolve()}")
    print(f"Total {len(final_boxes)} pohon sawit terdeteksi di seluruh raster.")
    print("\nCara buka di QGIS:")
    print("  Layer → Add Layer → Add Vector Layer → pilih file .shp")
    print("  Layer akan otomatis ter-overlay di posisi yang benar di peta.")

    # Buat dan simpan gambar visual dengan bounding box
    print("\nMembuat visualisasi (gambar dengan bounding box)...")
    out_img = Path(f"deteksi_{raster_path.stem}.jpg")
    draw_detections_on_raster(raster_path, final_boxes, final_scores, out_img)
    print(f"Gambar hasil visual disimpan di: {out_img.resolve()}")


if __name__ == "__main__":
    main()