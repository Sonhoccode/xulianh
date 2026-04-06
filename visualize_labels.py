import os
import cv2

DATASET_PATH = "dataset_S2"    # đường dẫn dataset
OUTPUT_PATH = "dataset_S2_visualized"    # folder render output

splits = ["train", "val", "test"]
exts = [".png", ".jpg", ".jpeg"]

def draw_yolo_boxes(img, labels):
    h, w = img.shape[:2]
    for lb in labels:
        cls, x, y, bw, bh = lb
        x, y, bw, bh = float(x), float(y), float(bw), float(bh)

        # YOLO → pixel box
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # vàng
    return img


def visualize_split(split):
    lbl_dir = os.path.join(DATASET_PATH, split, "labels")
    img_dir = os.path.join(DATASET_PATH, split, "images")

    # folder output
    out_dir = os.path.join(OUTPUT_PATH, split)
    os.makedirs(out_dir, exist_ok=True)

    label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

    for lbl_file in label_files:
        base = lbl_file.replace(".txt", "")
        img_path = None
        for e in exts:
            candidate = os.path.join(img_dir, base + e)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print("Ảnh thiếu:", base)
            continue

        # đọc ảnh
        img = cv2.imread(img_path)

        # đọc labels
        label_path = os.path.join(lbl_dir, lbl_file)
        labels = []
        with open(label_path, "r") as lf:
            for line in lf.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append(parts)

        # draw
        img = draw_yolo_boxes(img, labels)

        # lưu output
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    for split in splits:
        print(f"Rendering split: {split} ...")
        visualize_split(split)
    print("DONE ✓ Saved to folder:", OUTPUT_PATH)
