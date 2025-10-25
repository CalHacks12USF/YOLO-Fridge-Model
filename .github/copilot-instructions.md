# Copilot instructions for YOLO-Fridge-Model

## Big picture
- This repo trains and serves a YOLO-based detector for fridge items, then streams live detections from two models in parallel: your custom "fridge" model and a COCO model (for person/bottle). Results are merged, smoothed over a short window, and POSTed to an HTTP endpoint only when the inventory changes.
- Dataset uses Ultralytics YOLO format with images/labels split into `train` and `val` subfolders. Class mapping and counts live in `data.yaml` (12 classes).

## Key files and roles
- `data.yaml` — Dataset config for Ultralytics (paths, class names/order). Names must match label indices used in `.txt` files.
- `split_yolo.py` — One-time utility to split a flat `images/` and `labels/` set into `train/` and `val/` by matching stems; supports copy or move. Warns if a label is missing a matching image.
- `YOLO_train.py` — Minimal training starter using Ultralytics. Update it to point to this repo's `data.yaml` and desired base weights. Artifacts land under `runs/` (e.g., `runs/detect/train/weights/best.pt`).
- `stream_ensemble_test.py` — Live inference + posting pipeline. Loads your fridge model and a COCO model, runs both per frame, merges counts, smooths via a deque, and posts JSON deltas on a fixed interval.

## Data conventions
- Expected layout after splitting:
  - `images/train`, `images/val`
  - `labels/train`, `labels/val`
- Label/image pairing is done by filename stem; supported image extensions: `.jpg|.jpeg|.png` (case-insensitive).
- `data.yaml` (current):
  - `train: images/train`, `val: images/val`
  - `names` = ['Bread','Eggs','Grapes','Milk','Orange','Soda Can','Yogurt','apple','avocado','banana','empty','lime']
  - Keep order stable; indices in label files must align with this list.

## Training workflow (Ultralytics YOLO)
- Recommended: start from a small pretrained family weight bundled here (e.g., `yolov8n.pt`) and train against `data.yaml` with your epochs/imgsz.
- The provided `YOLO_train.py` is a template; change `YOLO("yolo11n.pt").train(data="coco8.yaml", ...)` to your local weights and `data="data.yaml"`.
- Trained weights typically appear at `runs/detect/train/weights/best.pt` (check Ultralytics logs for the exact path).

## Streaming/inference workflow
- `stream_ensemble_test.py` CLI args (most impactful):
  - `--fridge_model` (default `runs/detect/train/weights/best.pt`)
  - `--coco_model` (default `yolov8n.pt`)
  - `--source` camera index as a string (e.g., "0") or a video/RTSP URL. On macOS, AVFoundation backend is selected automatically for numeric cameras.
  - `--conf_items`, `--conf_coco`, `--imgsz`, `--frame_skip`, `--show`, `--interval`
- Smoothing + change detection:
  - Maintains a short history (deque) and averages counts over ~N seconds.
  - Posts only when the sorted `inventory` list (name/quantity) changes.
- Posting JSON contract:
  - Headers: `Content-Type: application/json`; optional `Authorization: Bearer <SUPABASE_SERVICE_ROLE>` and `apikey`.
  - Body: `{ "timestamp": "YYYY-MM-DDTHH:MM:SSZ", "inventory": [{"name": str, "quantity": int}, ...] }`
  - Configure `SUPABASE_POST_URL` (required) and `SUPABASE_SERVICE_ROLE` (optional) via environment variables.

## External deps and versions
- Python packages: `ultralytics`, `opencv-python`, `requests` (plus transitive deps). A local venv folder `yolo-venv/` is present; prefer activating it for all scripts.
- Ultralytics version differences: class names may be at `model.names` or `names`; the code handles both.

## Gotchas and tips
- Ensure every label has a matching image (by stem) before training; `split_yolo.py` prints a summary and warns about missing pairs.
- Keep `data.yaml` names/order in sync with your label indices; retraining with a re-ordered list will mislabel predictions.
- When using a webcam on macOS, pass the camera index as a string (e.g., "0"); the script selects AVFoundation for stability.
- Secrets: never commit `SUPABASE_SERVICE_ROLE`; set it via environment only. Fail-fast occurs if `SUPABASE_POST_URL` is unset.
