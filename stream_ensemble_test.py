import argparse, time, json, sys, platform, os
from datetime import datetime, timezone
from collections import Counter, deque
import cv2
import requests
from ultralytics import YOLO

# Lightweight .env loader (no external dependency required)
def _load_dotenv_inline(path: str = ".env"):
    try:
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # don't overwrite if already set in the environment
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        # best-effort; ignore errors silently
        pass

# Try python-dotenv first; fall back to inline loader
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    _load_dotenv_inline()

# COCO class indices for 'person' (0) and 'bottle' (39)
COCO_PERSON = 0
COCO_BOTTLE = 39

# Helper function: get current UTC time in ISO 8601 format
def iso_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Helper function: POST w/ exponential backoff
def post_with_backoff(url, payload, headers, tries=3, base=0.5):
    for i in range(tries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(base * (2 ** i))

# Main function
def main():
    # Parse args for CLI
    ap = argparse.ArgumentParser(description="Fridge + COCO(person,bottle) streaming predict (POST on change every 10s)")
    ap.add_argument("--fridge_model", default="runs/detect/train/weights/best.pt", help="Path to your trained fridge model")
    ap.add_argument("--coco_model",   default="yolov8n.pt", help="COCO model to use for person/bottle")
    ap.add_argument("--source",       default="0", help="Camera index (e.g. 0/1) or video/RTSP URL")
    ap.add_argument("--conf_items",   type=float, default=0.45)
    ap.add_argument("--conf_coco",    type=float, default=0.45)
    ap.add_argument("--imgsz",        type=int,   default=832)
    ap.add_argument("--frame_skip",   type=int,   default=1, help="Process every Nth frame (1 = every frame)")
    ap.add_argument("--show",         action="store_true", help="Show window with drawn boxes")
    ap.add_argument("--interval",     type=int,   default=10, help="Seconds between JSON posts/checks")
    args = ap.parse_args()

    # Endpoint / auth from env (.env supported)
    ENDPOINT = os.getenv("SUPABASE_POST_URL")
    API_KEY  = os.getenv("SUPABASE_SERVICE_ROLE")
    if not ENDPOINT:
        sys.exit("Set SUPABASE_POST_URL in your environment or .env file (e.g., SUPABASE_POST_URL=https://...)")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
        headers["apikey"] = API_KEY

    # Load models
    print("[Init] Loading models…")
    fridge = YOLO(args.fridge_model)
    coco   = YOLO(args.coco_model)

    f_names = fridge.model.names if hasattr(fridge.model, "names") else fridge.names
    c_names = coco.model.names   if hasattr(coco.model, "names")   else coco.names

    # Open source
    src = args.source
    if src.isdigit():
        cam_index = int(src)
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam_index, backend)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        sys.exit(f"Could not open video source: {args.source}")

    print(f"[Run] Streaming from {args.source} (press Q to quit)")
    frame_i = 0
    last_tick = 0.0
    last_sent_inventory = None  # cached JSON 'inventory' (list of dicts)

    # Optional: light temporal smoothing over the last few seconds to reduce flicker
    FPS_GUESS = 20
    history = deque(maxlen=args.interval * FPS_GUESS)  # ~N seconds of frame-level counts

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame_i += 1
            if frame_i % max(1, args.frame_skip) != 0:
                # still show the latest frame if requested
                if args.show:
                    cv2.imshow("Fridge + COCO(person,bottle)", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                continue

            inv = Counter()

            # Fridge detections (your custom classes)
            fr = fridge.predict(source=frame, conf=args.conf_items, imgsz=args.imgsz, verbose=False)[0]
            if fr.boxes is not None and len(fr.boxes):
                for b in fr.boxes:
                    inv[f_names[int(b.cls.item())]] += 1

            # COCO detections for person + bottle
            cr = coco.predict(source=frame, conf=args.conf_coco, imgsz=args.imgsz,
                              classes=[COCO_PERSON, COCO_BOTTLE], verbose=False)[0]
            if cr.boxes is not None and len(cr.boxes):
                for b in cr.boxes:
                    inv[c_names[int(b.cls.item())]] += 1  # 'person' or 'bottle'

            # keep a short history for stability
            history.append(inv.copy())

            # Show live stream with both overlays
            if args.show:
                vis_fr = fr.plot()
                vis_co = cr.plot()
                vis = cv2.addWeighted(vis_fr, 0.7, vis_co, 0.3, 0)
                cv2.imshow("Fridge + COCO(person,bottle)", vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            # Every N seconds, compute a stable summary and POST only if changed
            now = time.time()
            if now - last_tick >= args.interval:
                # aggregate over history (average occurrences per frame, rounded)
                agg = Counter()
                for h in history:
                    agg.update(h)
                denom = max(1, len(history))
                stable = {k: int(round(agg[k] / denom)) for k in agg}

                # build inventory list (sorted for deterministic comparison)
                inventory = [{"name": k, "quantity": int(v)} for k, v in sorted(stable.items())]

                if inventory != (last_sent_inventory or []):
                    payload = {
                        "timestamp": iso_utc(),
                        "inventory": inventory
                    }
                    try:
                        post_with_backoff(ENDPOINT, payload, headers)
                        print("[POST]", json.dumps(payload))
                        last_sent_inventory = inventory
                    except Exception as e:
                        print("[WARN] POST failed:", e)
                else:
                    print("[SKIP] No inventory change.")

                last_tick = now

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print("\n[Exit] Camera released.")

if __name__ == "__main__":
    main()
