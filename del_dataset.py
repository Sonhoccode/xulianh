import os
import shutil
import random
from tqdm import tqdm

# ================== CONFIG ==================
SRC_DATASET = "dataset"              # dataset gốc (của bạn hiện tại)
NEW_DATASET = "dataset_balanced"     # dataset mới đã cân bằng
MAX_PER_CLASS = 600
VAL_RATIO = 0.2
RANDOM_SEED = 0
# ============================================

random.seed(RANDOM_SEED)

# Đường dẫn dataset gốc
train_img_dir = os.path.join(SRC_DATASET, "train/images")
train_lbl_dir = os.path.join(SRC_DATASET, "train/labels")
val_img_dir   = os.path.join(SRC_DATASET, "val/images")   if os.path.exists(os.path.join(SRC_DATASET, "val/images")) else None
val_lbl_dir   = os.path.join(SRC_DATASET, "val/labels")   if os.path.exists(os.path.join(SRC_DATASET, "val/labels")) else None
test_img_dir  = os.path.join(SRC_DATASET, "test/images")
test_lbl_dir  = os.path.join(SRC_DATASET, "test/labels")

# Tạo folder dataset_balanced
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(NEW_DATASET, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET, split, "labels"), exist_ok=True)

print("=== Bước 1: Gom toàn bộ labels (train + val nếu có) ===")
id_to_paths = {}    # base_name -> {"img": path, "lbl": path}
class_map = {}      # class_id -> [base_name, base_name, ...]

def collect_from(label_dir, img_dir):
    if not (label_dir and img_dir and os.path.exists(label_dir) and os.path.exists(img_dir)):
        return
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for label_file in tqdm(label_files, desc=f"Đọc labels từ {label_dir}"):
        lbl_path = os.path.join(label_dir, label_file)
        base = os.path.splitext(label_file)[0]

        # đọc class id dòng đầu
        with open(lbl_path, "r") as lf:
            line = lf.readline().strip().split()
            if not line:
                continue
            cls_id = int(line[0])

        # tìm ảnh tương ứng
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # lưu mapping
        id_to_paths[base] = {"img": img_path, "lbl": lbl_path}

        if cls_id not in class_map:
            class_map[cls_id] = []
        class_map[cls_id].append(base)

# Gom từ train và val (nếu có)
collect_from(train_lbl_dir, train_img_dir)
collect_from(val_lbl_dir, val_img_dir)

print(f"Phát hiện {len(class_map)} lớp trong toàn bộ train+val.")
total_samples = sum(len(v) for v in class_map.values())
print(f"Tổng số ảnh ban đầu (train+val): {total_samples}")

print("\n=== Bước 2: Cân bằng mỗi class tối đa 600 ảnh ===")
kept_ids = []
for cls, ids in class_map.items():
    original_len = len(ids)
    if original_len > MAX_PER_CLASS:
        ids = random.sample(ids, MAX_PER_CLASS)
    kept_ids.extend(ids)
    print(f"Class {cls}: {original_len} -> {len(ids)} ảnh giữ lại")

kept_ids = list(set(kept_ids))
print(f"Tổng số ảnh sau cân bằng: {len(kept_ids)}")

print("\n=== Bước 3: Shuffle và tách train/val ===")
random.shuffle(kept_ids)
val_count = int(len(kept_ids) * VAL_RATIO)
val_ids = kept_ids[:val_count]
train_ids = kept_ids[val_count:]

print(f"Số ảnh train: {len(train_ids)}")
print(f"Số ảnh val:   {len(val_ids)}")

def copy_items(id_list, split):
    print(f"\nCopy ảnh cho split: {split}")
    for base in tqdm(id_list, desc=f"Copy {split}"):
        paths = id_to_paths.get(base)
        if not paths:
            continue

        img_src = paths["img"]
        lbl_src = paths["lbl"]

        img_dst = os.path.join(NEW_DATASET, split, "images", os.path.basename(img_src))
        lbl_dst = os.path.join(NEW_DATASET, split, "labels", os.path.basename(lbl_src))

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)

copy_items(train_ids, "train")
copy_items(val_ids, "val")

print("\n=== Bước 4: Chuẩn hoá test (copy vào dataset_balanced) ===")
# test: lấy tối đa bằng số val
test_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(test_img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"Số ảnh test gốc: {len(test_files)}")
max_test = min(len(test_files), len(val_ids))
print(f"Sẽ copy {max_test} ảnh test sang dataset_balanced (bằng hoặc nhỏ hơn số val).")

test_keep = random.sample(test_files, max_test)

for base in tqdm(test_keep, desc="Copy test"):
    img_src = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(test_img_dir, base + ext)
        if os.path.exists(candidate):
            img_src = candidate
            break
    lbl_src = os.path.join(test_lbl_dir, base + ".txt")

    if img_src and os.path.exists(lbl_src):
        img_dst = os.path.join(NEW_DATASET, "test", "images", os.path.basename(img_src))
        lbl_dst = os.path.join(NEW_DATASET, "test", "labels", os.path.basename(lbl_src))

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)

print("\n=== HOÀN THÀNH dataset_balanced ===")
print(f"Train images: {len(os.listdir(os.path.join(NEW_DATASET, 'train/images')))}")
print(f"Val images:   {len(os.listdir(os.path.join(NEW_DATASET, 'val/images')))}")
print(f"Test images:  {len(os.listdir(os.path.join(NEW_DATASET, 'test/images')))}")
