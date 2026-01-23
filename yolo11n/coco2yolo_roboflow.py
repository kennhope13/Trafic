import json
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_coco_json(split_dir: Path) -> Path:
    # Roboflow hay đặt tên như dưới
    candidates = [
        split_dir / "_annotations.coco.json",
        split_dir / "annotations.coco.json",
        split_dir / "coco.json",
        split_dir / "_annotations.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: tìm file *.json lớn nhất
    js = list(split_dir.glob("*.json"))
    if not js:
        raise FileNotFoundError(f"Không thấy COCO json trong: {split_dir}")
    return max(js, key=lambda x: x.stat().st_size)

def ensure_images_dir(split_dir: Path) -> Path:
    """
    Ultralytics chuẩn nhất là: split/images và split/labels.
    Nếu ảnh đang nằm trực tiếp trong split/, ta chuyển vào split/images.
    """
    images_dir = split_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # nếu đã có ảnh trong images/ thì thôi
    if any(p.suffix.lower() in IMG_EXTS for p in images_dir.rglob("*")):
        return images_dir

    # chuyển ảnh từ split_dir vào images_dir (không đụng json)
    for p in split_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.move(str(p), str(images_dir / p.name))
    return images_dir

def coco_to_yolo_one_split(split_dir: Path, cat_id_to_idx: dict):
    coco_path = find_coco_json(split_dir)
    images_dir = ensure_images_dir(split_dir)
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    # index ảnh theo id
    img_by_id = {im["id"]: im for im in coco.get("images", [])}

    # gom ann theo image_id
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        image_id = ann["image_id"]
        anns_by_img.setdefault(image_id, []).append(ann)

    # tạo label file theo từng ảnh
    for image_id, im in img_by_id.items():
        file_name = Path(im["file_name"]).name  # lấy basename
        w = float(im["width"])
        h = float(im["height"])
        label_lines = []

        for ann in anns_by_img.get(image_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                continue
            cls = cat_id_to_idx[cat_id]

            # COCO bbox: [x_min, y_min, width, height]
            x, y, bw, bh = ann["bbox"]
            x_c = (x + bw / 2.0) / w
            y_c = (y + bh / 2.0) / h
            bw_n = bw / w
            bh_n = bh / h

            # clamp nhẹ cho an toàn
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            bw_n = min(max(bw_n, 0.0), 1.0)
            bh_n = min(max(bh_n, 0.0), 1.0)

            label_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}")

        (labels_dir / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8"
        )

    print(f"[OK] {split_dir.name}: {len(img_by_id)} images → labels at {labels_dir}")

def main(dataset_root: str):
    root = Path(dataset_root).resolve()

    # đọc categories từ train (thường đủ)
    train_dir = root / "train"
    coco_train = json.loads(find_coco_json(train_dir).read_text(encoding="utf-8"))
    cats = coco_train.get("categories", [])
    # map category_id (COCO có thể không liên tiếp) -> 0..nc-1 theo cat_id tăng dần
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]

    # convert từng split
    for split in ["train", "valid", "test"]:
        d = root / split
        if d.exists():
            coco_to_yolo_one_split(d, cat_id_to_idx)

    # tạo data.yaml
    yaml = []
    yaml.append(f"path: {root.as_posix()}")
    yaml.append("train: train/images")
    yaml.append("val: valid/images")
    yaml.append("test: test/images")
    yaml.append("names:")
    for i, n in enumerate(names):
        yaml.append(f"  {i}: {n}")

    (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {root/'data.yaml'} with nc={len(names)}")

if __name__ == "__main__":
    # chạy trong thư mục dataset root
    main(".")