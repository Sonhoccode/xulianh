from ultralytics import YOLO

if __name__ == "__main__":
    # Resume Stage 2 từ last.pt
    model = YOLO("runs/detect/YOLO_Stage2/weights/last.pt")

    results = model.train(
        data="data.yaml",
        imgsz=768,
        epochs=80,             
        batch=16,
        device=0,
        workers=2,

        lr0=0.0005,
        lrf=0.005,
        cos_lr=True,
        
        weight_decay=0.0005,
        optimizer="auto",
        momentum=0.937,
        warmup_epochs=2.0,

        patience=30,
        seed=0,
        deterministic=True,

        name="YOLO_Stage2",
        project="runs/detect",
        resume=True,            # rất quan trọng
        verbose=True,
    )

    print("Đã resume và hoàn tất Stage 2.")
