# 🧊 FridgeVision — Real-Time YOLO Object Detection

This project uses a YOLOv8 model (`best.pt`) to detect and track the contents of your refrigerator in real time using your MacBook webcam.
It can show a live annotated feed and, when configured, post an inventory JSON to your backend whenever the contents change.

---

## 🚀 Quick Start

### 1️⃣ Clone and enter the repo

```bash
git clone https://github.com/CalHacks12USF/YOLO-Fridge-Model.git
cd YOLO-Fridge-Model
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install ultralytics opencv-python requests
```

### 4️⃣ Add your model

Place your trained YOLO weights in the project root as:

```
best.pt
```

If you don’t have one yet, see “Fine-tune or Train New Model” below.

---

## 🧠 Running the Model (Webcam)

### Simple run (Ultralytics CLI)

```bash
yolo predict model=best.pt source=0 conf=0.3 device=mps
```

- `source=0` → use your Mac’s webcam
- `conf=0.3` → confidence threshold
- `device=mps` → uses Apple GPU (M1/M2/M3)

To quit: press Ctrl+C or close the video window.

---

## 🧩 Run the Python script (counts + JSON posting)

This repo’s script is `stream_ensemble_test.py`. It runs your fridge model and a COCO model (person/bottle) together, smooths counts over a short window, and POSTS JSON only when inventory changes.

First, set environment variables (required to run the script):

```bash
export SUPABASE_POST_URL="https://your-endpoint.example.com/ingest"   # required
export SUPABASE_SERVICE_ROLE="<service-role-token>"                   # optional; adds Authorization header
```

Then run with your weights and the bundled COCO model:

```bash
python stream_ensemble_test.py \
	--fridge_model best.pt \
	--coco_model yolov8n.pt \
	--source "0" \
	--conf_items 0.45 --conf_coco 0.45 --imgsz 832 --frame_skip 1 --interval 10 --show
```

Notes
- On macOS, pass camera index as a string (e.g., "0"); the script selects AVFoundation for stability.
- Use a file path or RTSP URL for `--source` to process video instead of a webcam.
- `--show` opens a visualization window; press `q` or `Esc` to quit.

---

## ⚙️ Example JSON Output

The script POSTs this payload shape when inventory changes:

```json
{
	"timestamp": "2025-10-25T13:00:00Z",
	"inventory": [
		{ "name": "Milk", "quantity": 1 },
		{ "name": "banana", "quantity": 2 }
	]
}
```

Headers
- `Content-Type: application/json`
- Optional: `Authorization: Bearer <SUPABASE_SERVICE_ROLE>` and `apikey: <SUPABASE_SERVICE_ROLE>`

---

## 🧩 Optional: Fine-tune or Train New Model

If you have labeled data in YOLO format (see `data.yaml`), you can train via Ultralytics CLI:

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

Your best weights will be saved as:

```
runs/detect/train/weights/best.pt
```

You can also use the starter script `YOLO_train.py` (update it to point to `data.yaml`).

---

## 📸 Notes & Troubleshooting

- Camera access: System Settings → Privacy & Security → Camera → enable for Terminal/VS Code.
- No posts: verify `SUPABASE_POST_URL`, check console for `[WARN] POST failed:` details.
- Class mismatches: ensure your label indices match the order in `data.yaml` before training.
- Flicker: increase `--interval` or `--frame_skip`; the script averages counts over recent frames.

---

## 🏁 Summary

| Step | Command                                                   | Description          |
| ---- | --------------------------------------------------------- | -------------------- |
| 1    | `source .venv/bin/activate`                               | Activate environment |
| 2    | `pip install ultralytics opencv-python requests`          | Install dependencies |
| 3    | `yolo predict model=best.pt source=0 device=mps`          | Quick webcam test    |
| 4    | `python stream_ensemble_test.py --fridge_model best.pt …` | Run fridge detector  |
| 5    | *(optional)* `yolo train model=yolov8n.pt ...`            | Retrain model        |


