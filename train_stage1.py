from ultralytics import YOLO

if __name__ == "__main__":
    # Stage 1: train từ checkpoint COCO (yolov8s.pt)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data.yaml",      # đường dẫn file cấu hình dataset
        imgsz=640,              # kích thước ảnh
        epochs=150,             # số epoch Stage 1
        batch=16,               # nếu dư VRAM có thể tăng 32
        device=0,               # GPU 0 (RTX 4050)
        workers=2,              # để 2 cho Windows cho chắc

        # Learning rate & scheduler
        lr0=0.0015,             # LR khởi điểm (vừa phải, không quá gắt)
        lrf=0.01,               # LR cuối (default)
        cos_lr=True,            # cosine decay, mượt và ổn định

        # Regularization / ổn định training
        weight_decay=0.0005,
        optimizer="auto",       # để YOLO tự chọn (thường là AdamW)
        momentum=0.937,
        warmup_epochs=3.0,

        # Early stopping / reproducibility
        patience=30,            # dừng sớm nếu không cải thiện
        seed=0,
        deterministic=True,

        # Logging
        name="YOLO_Stage1",
        project="runs/detect",
        verbose=True,
    )

    print("Hoàn thành Stage 1. Kết quả lưu ở runs/detect/YOLO_Stage1")
