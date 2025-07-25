import cv2
import numpy as np
from ultralytics import YOLO
from custom_tracker import CT
import argparse
import time
import sys
import traceback

VID = "person4.mp4"
OBJ = "person"
MODEL = "yolov8n.pt"
CONF = 0.6

CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
    "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
    "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
    "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
    "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
    "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
    "toothbrush": 79
}

PRESETS = {
    "person": {"min": 5000, "rmin": 2000, "mode": "normal"},
    "car": {"min": 8000, "rmin": 3000, "mode": "high_motion"},
    "dog": {"min": 3000, "rmin": 1500, "mode": "high_motion"},
    "cat": {"min": 2000, "rmin": 1000, "mode": "smooth"},
    "bicycle": {"min": 6000, "rmin": 2500, "mode": "normal"},
    "laptop": {"min": 4000, "rmin": 2000, "mode": "smooth"}
}

def args():
    p = argparse.ArgumentParser(description='Advanced Multi-Object Tracking System')
    p.add_argument('--object', '-o', type=str, help='Object to track')
    p.add_argument('--video', '-v', type=str, help='Video path')
    p.add_argument('--confidence', '-c', type=float, help='YOLO confidence threshold')
    p.add_argument('--ad', action='store_true', help='Enable auto re-detection')
    p.add_argument('--hd', action='store_true', help='Enable history re-detection')
    p.add_argument('--mode', '-m', type=str, choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    p.add_argument('--list', '-l', action='store_true', help='List available objects')
    return p.parse_args()

def center(b):
    x, y, w, h = b
    return (x + w/2, y + h/2)

def dist(b1, b2):
    c1 = center(b1)
    c2 = center(b2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def size_sim(b1, b2):
    _, _, w1, h1 = b1
    _, _, w2, h2 = b2
    a1 = w1 * h1
    a2 = w2 * h2
    if a1 == 0 or a2 == 0: return 0
    return min(a1, a2) / max(a1, a2)

def best_det(boxes, min_area, cid=None):
    best = None
    score = 0
    best_cid = None
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        conf = b.conf[0].cpu().numpy()
        cls = int(b.cls[0].cpu().numpy())
        if cid is not None and cls != cid:
            continue
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        area = w * h
        if area > min_area and conf > score:
            score = conf
            best = b
            best_cid = cls
    return best, score, best_cid

def best_match(cur, dets, hist):
    if not dets:
        return None
    best = None
    score = -1
    for d in dets:
        xyxy = d.xyxy[0].cpu().numpy()
        conf = d.conf[0].cpu().numpy()
        db = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
        dscore = max(0, 1 - (dist(cur, db) / 200))
        sscore = size_sim(cur, db)
        hscore = 0
        if hist:
            hdist = [dist(hb, db) for hb in hist[-5:]]
            hscore = max(0, 1 - (np.mean(hdist) / 200))
        tscore = (dscore * 0.4 + sscore * 0.3 + hscore * 0.2 + conf * 0.1)
        if tscore > score:
            score = tscore
            best = d
    return best

def cname(cid):
    for n, i in CLASSES.items():
        if i == cid:
            return n
    return f"class_{cid}"

def list_objs():
    print("\nAvailable preset objects:")
    for n, c in PRESETS.items():
        print(f"  '{n}' -> {c.get('mode', 'normal')} mode")
    print(f"\nAll YOLO classes ({len(CLASSES)} total):")
    for i, (n, cid) in enumerate(CLASSES.items()):
        if i > 0 and i % 4 == 0:
            print()
        print(f"  {n:15} (ID:{cid:2d})", end="")
    print("\n")

def main():
    a = args()
    if a.list:
        list_objs()
        return
    obj = a.object if a.object else OBJ
    vid = a.video if a.video else VID
    conf = a.confidence if a.confidence else CONF
    auto = a.ad
    hist = a.hd

    if obj.lower() not in CLASSES:
        print(f"Error: Unknown object '{obj}'")
        list_objs()
        return
    cid = CLASSES[obj.lower()]

    cfg = PRESETS.get(obj.lower(), {"min": 3000, "rmin": 1500, "mode": "normal"})
    print(f"Configuration:")
    print(f"  Target: {obj} (class_id: {cid})")
    print(f"  Video: {vid}")
    print(f"  Confidence: {conf}")
    print(f"  Min Area: {cfg['min']}")

    m = YOLO(MODEL)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"Error opening video {vid}")
        return

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    ftime = 1.0 / fps

    ok, frame = cap.read()
    if not ok: return

    print(f"\nDetecting initial {obj}...")
    res = m(frame, verbose=False, classes=[cid], conf=conf)
    b, c, dcid = best_det(res[0].boxes, cfg['min'], cid)
    if b is None:
        print(f"No suitable {obj} detection found.")
        return

    xyxy = b.xyxy[0].cpu().numpy()
    ibox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

    t = CT()
    mode = a.mode if a.mode else cfg['mode']
    t.set_tm(mode)
    print(f"  Tracking Mode: {t.tm}")
    t.init(frame, ibox)

    bhist = [ibox]
    pause = False
    flimit = False
    rc = 0
    hc = 0

    fcnt = 0
    st = time.time()
    afps = 0
    fc = 0
    lft = time.time()

    while True:
        if not pause:
            fcnt += 1
            ok, frame = cap.read()
            if not ok: break

            ts, box = t.upd(frame)
            if ts and box:
                bhist.append(box)
                if len(bhist) > 20: bhist.pop(0)

            if auto and not ts:
                rc += 1
                if rc >= int(fps / 2):
                    rc = 0
                    res = m(frame, verbose=False, classes=[cid], conf=conf * 0.8)
                    b2, c2, _ = best_det(res[0].boxes, cfg['rmin'], cid)
                    if b2 is not None:
                        xyxy = b2.xyxy[0].cpu().numpy()
                        nb = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                        if t.init(frame, nb):
                            ts, box = True, nb
                            bhist.append(nb)

            if hist and ts and box:
                hc += 1
                if hc >= 7:
                    hc = 0
                    res = m(frame, verbose=False, classes=[cid], conf=conf * 0.7)
                    if len(res[0].boxes) > 0:
                        bm = best_match(box, res[0].boxes, bhist)
                        if bm is not None:
                            xyxy = bm.xyxy[0].cpu().numpy()
                            nb = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                            if dist(box, nb) > 50:
                                t.init(frame, nb)
                                ts, box = True, nb
                                bhist.append(nb)

        # This part runs even when paused
        disp = frame.copy()
        if ts and box is not None:
            x, y, w, h = box
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(disp, f"Tracking: {len(t.tr)} features", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, f"TRACKING LOST - Frame {fcnt}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(disp, "Press 'r' for manual detection or 'f' for auto mode", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if not pause:
            ct = time.time()
            fc += 1
            if ct - st >= 1.0:
                afps = fc / (ct - st)
                fc = 0
                st = ct

        iy = 30
        cv2.putText(disp, f"Target: {obj}", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        iy += 25
        cv2.putText(disp, f"Frame: {fcnt}/{frames}", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        iy += 20
        cv2.putText(disp, f"Mode: {t.tm.upper()}", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        iy += 20
        fcol = (0, 255, 255) if flimit else (255, 255, 255)
        ftxt = f"FPS: {afps:.1f}" + (" (Limited)" if flimit else " (Unlimited)")
        cv2.putText(disp, ftxt, (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcol, 1)
        iy += 20
        if auto:
            cv2.putText(disp, f"Auto-redetect: ON", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            iy += 20
        if hist:
            cv2.putText(disp, f"History-redetect: ON ({7-hc})", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            iy += 20
        if not auto and not hist:
            cv2.putText(disp, "YOLO-ONCE MODE", (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        if pause:
            cv2.putText(disp, "PAUSED - Press 'p' to continue, SPACE to step", (10, disp.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(f"Tracking - {obj}", disp)

        if not pause and flimit:
            el = time.time() - lft
            wt = max(0, ftime - el)
            if wt > 0:
                time.sleep(wt)
        lft = time.time()

        k = cv2.waitKey(1 if not pause else 0) & 0xFF
        if k == ord('q'): break
        elif k == ord('p'): pause = not pause
        elif k == ord(' ') and pause:
            fcnt += 1
            ok, frame = cap.read()
            if not ok: break
        elif k == ord('f'): auto = not auto
        elif k == ord('e'): hist = not hist
        elif k == ord('l'): flimit = not flimit
        elif k == ord('s'): t.set_tm("smooth")
        elif k == ord('h'): t.set_tm("high_motion")
        elif k == ord('n'): t.set_tm("normal")
        elif k == ord('r'):
            res = m(frame, verbose=False, classes=[cid], conf=conf * 0.8)
            b2, c2, _ = best_det(res[0].boxes, cfg['rmin'], cid)
            if b2 is not None:
                xyxy = b2.xyxy[0].cpu().numpy()
                nb = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                t.init(frame, nb)
                bhist.append(nb)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()