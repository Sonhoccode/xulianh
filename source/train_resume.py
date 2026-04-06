from ultralytics import YOLO

if __name__ == "__main__":
    # Resume từ checkpoint Stage 1
    model = YOLO("runs/detect/YOLO_Stage1/weights/last.pt")

    results = model.train(
        data="data.yaml",
        imgsz=640,
        epochs=200,             # giữ nguyên số epoch target
        batch=16,
        device=0,
        workers=2,

        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        optimizer="auto",
        momentum=0.937,
        warmup_epochs=3.0,

        patience=30,
        seed=0,
        deterministic=True,

        name="YOLO_Stage1",
        project="runs/detect",
        resume=True,            # quan trọng: cho YOLO hiểu là train tiếp
        verbose=True,
    )

    print("Đã resume và hoàn tất Stage 1.")
