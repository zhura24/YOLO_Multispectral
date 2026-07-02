import sys
import json
from pathlib import Path

try:
    import rasterio
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Modul belum terinstall: {e}")
    print("Jalankan: pip install rasterio numpy")
    sys.exit(1)

BAND_STATS_PATH = r"C:\Users\user\Downloads\Compressed\nyawit multispectral\band_stats.json"


def stretch_band(band: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    band = band.astype(np.float64)
    if p_high - p_low == 0:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)


def main():
    print("=== Convert Raster Float32 → uint8 GeoTIFF ===\n")

    raster_path = Path(input("Path raster mentah (.tif): ").strip().strip('"'))
    raw_out = input("Path output — boleh folder saja atau path lengkap dengan nama file\n"
                    "  Contoh folder : C:\\Users\\user\\Downloads\\\n"
                    "  Contoh lengkap: C:\\Users\\user\\Downloads\\hasil_uint8.tif\n"
                    "> ").strip().strip('"')
    output_path = Path(raw_out)

    # Kalau user kasih folder saja, otomatis buatkan nama file
    if output_path.is_dir() or output_path.suffix.lower() not in (".tif", ".tiff"):
        stem = raster_path.stem + "_uint8"
        output_path = (output_path if output_path.suffix == "" else output_path.parent) / f"{stem}.tif"
        print(f"  → Output akan disimpan sebagai: {output_path}")
    stats_path  = Path(BAND_STATS_PATH)

    if not raster_path.is_file():
        print(f"[ERROR] File tidak ditemukan: {raster_path}")
        sys.exit(1)
    if not stats_path.is_file():
        print(f"[ERROR] band_stats.json tidak ditemukan: {stats_path}")
        sys.exit(1)

    with open(stats_path) as f:
        band_stats = {int(k): v for k, v in json.load(f).items()}

    with rasterio.open(raster_path) as src:
        print(f"Ukuran  : {src.width} x {src.height} piksel")
        print(f"Band    : {src.count}")
        print(f"CRS     : {src.crs}")
        print(f"Dtype   : {src.dtypes[0]}\n")

        if src.count != len(band_stats):
            print(f"[ERROR] Raster punya {src.count} band, "
                  f"band_stats.json punya {len(band_stats)} band.")
            sys.exit(1)

        profile = src.profile.copy()
        profile.update(dtype="uint8", compress="lzw")
        profile.pop("nodata", None)  # hapus nodata lama (-32767 tidak valid untuk uint8

        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                print(f"  Memproses band {b}/{src.count} ...")
                data    = src.read(b)
                stats   = band_stats[b]
                dst.write(stretch_band(data, stats["p_low"], stats["p_high"]), b)

    print(f"\nSelesai! Output disimpan di: {output_path}")
    print("Buka di QGIS: Layer → Add Raster Layer → pilih file output.")


if __name__ == "__main__":
    main()