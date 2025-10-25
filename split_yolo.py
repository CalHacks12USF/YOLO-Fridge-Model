import argparse, random, shutil
from pathlib import Path

def find_image_for(stem, images_dir):
    exts = [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to images dir (flat)")
    ap.add_argument("--labels", required=True, help="Path to labels dir (flat)")
    ap.add_argument("--val", type=float, default=0.2, help="Validation fraction (0–1)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    # Create split dirs if not present
    (images_dir / "train").mkdir(parents=True, exist_ok=True)
    (images_dir / "val").mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    random.shuffle(label_files)

    n_val = int(len(label_files) * args.val)
    val_set = set(label_files[:n_val])

    moved_train = moved_val = 0
    missing_images = []

    for lbl in label_files:
        stem = lbl.stem
        img = find_image_for(stem, images_dir)
        if img is None:
            missing_images.append(stem)
            continue

        if lbl in val_set:
            dst_img = images_dir / "val" / img.name
            dst_lbl = labels_dir / "val" / lbl.name
        else:
            dst_img = images_dir / "train" / img.name
            dst_lbl = labels_dir / "train" / lbl.name

        if args.copy:
            shutil.copy2(img, dst_img)
            shutil.copy2(lbl, dst_lbl)
        else:
            shutil.move(str(img), str(dst_img))
            shutil.move(str(lbl), str(dst_lbl))

        if lbl in val_set:
            moved_val += 1
        else:
            moved_train += 1

    print(f"Train labels: {moved_train}, Val labels: {moved_val}")
    if missing_images:
        print(f"WARNING: {len(missing_images)} labels without matching images (by stem). Examples: {missing_images[:10]}")

if __name__ == "__main__":
    main()
