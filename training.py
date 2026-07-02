# training_simple.py

from ultralytics import YOLO

if __name__ == "__main__":
    # Load model YOLO26n (pretrained), otomatis download kalau belum ada
    model = YOLO("yolo26n.pt")

    # Training pakai GPU (device=0), workers=0 wajib di Windows
    results = model.train(
        data=r"C:\Users\user\Downloads\Compressed\nyawit multispectral\data.yaml",
        epochs=100,
        imgsz=640,
        device=0,      
        workers=0,   
    )   