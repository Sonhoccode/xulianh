from ultralytics import YOLO

if __name__ == "__main__":
    # Stage 2: fine-tune từ best của Stage 1
    model = YOLO("runs/detect/YOLO_Stage1/weights/best.pt")

    results = model.train(
        data="data.yaml",
        imgsz=768,
        epochs=1000,             
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

        patience=100,          
        seed=0,
        deterministic=True,

        name="YOLO_Stage2",
        project="runs/detect",
        verbose=True,
        
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        translate=0.1,
        scale=0.5,
    )

    print("Hoàn thành Stage 2. Kết quả lưu ở runs/detect/YOLO_Stage2")
