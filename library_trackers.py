import cv2
import numpy as np
from ultralytics import YOLO

HYBRID = False
REID = False

VID = "carr3.mp4"
CLS = "car"
MODEL = "yolov8n.pt"
TRK = "KCF"

HEALTH_INT = 45
REDET_INT = 15
SCALE_SMOOTH = 0.2
YOLO_CONF = 0.5
IOU_TH = 0.3
REID_TH = 0.45
FEAT_RATE = 0.05

def make_trk(t):
    if t == 'CSRT': trk = cv2.TrackerCSRT_create()
    elif t == 'KCF': trk = cv2.TrackerKCF_create()
    elif t == 'MOSSE': trk = cv2.legacy.TrackerMOSSE_create()
    else: raise ValueError("Invalid tracker type specified.")
    return trk

def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    area_a, area_b = a[2]*a[3], b[2]*b[3]
    union = float(area_a + area_b - inter)
    return inter/union if union > 0 else 0

def get_hist(img, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if w <= 0 or h <= 0: return None
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def main():
    mdl = YOLO(MODEL)
    cap = cv2.VideoCapture(VID)
    if not cap.isOpened(): print(f"Error opening video {VID}"); return
    ok, frm = cap.read()
    if not ok: print("Error reading first frame."); return

    cls_id = list(mdl.names.keys())[list(mdl.names.values()).index(CLS.lower())]
    res = mdl(frm, verbose=False, classes=[cls_id])

    box0, feat0 = None, None
    if len(res[0].boxes) > 0:
        xyxy = res[0].boxes[0].xyxy[0].cpu().numpy()
        box0 = tuple(map(int, [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]))
        if REID:
            feat0 = get_hist(frm, box0)
            if feat0 is None: print("Initial object has invalid size."); return
            print(f"Found '{CLS}' and extracted its feature signature.")

    if not box0: print("Target not found in first frame."); return

    trk = make_trk(TRK)
    trk.init(frm, box0)

    if HYBRID:
        run_hybrid(cap, trk, mdl, box0, feat0)
    else:
        run_track(cap, trk)

    cap.release()
    cv2.destroyAllWindows()

def run_track(cap, trk):
    fps_sum = 0
    cnt = 0

    while True:
        ok, frm = cap.read()
        if not ok: break
        t0 = cv2.getTickCount()
        ok_trk, box = trk.update(frm)
        if ok_trk:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3]))
            cv2.rectangle(frm, p1, p2, (0,255,0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - t0)
        fps_sum += fps
        cnt += 1

        cv2.putText(frm, f"FPS: {int(fps)} ({TRK})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Pure Tracking Mode", frm)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if cnt > 0:
        avg_fps = fps_sum / cnt
        print(f"\nPure Tracking Mode finished. Average FPS: {avg_fps:.2f}")

def run_hybrid(cap, trk, mdl, box0, feat0):
    active, lost, cnt, fps_sum = True, 0, 0, 0
    sz, sz_tgt = (box0[2], box0[3]), (box0[2], box0[3])
    cls_id = list(mdl.names.keys())[list(mdl.names.values()).index(CLS.lower())]

    while True:
        ok, frm = cap.read()
        if not ok: break
        cnt += 1
        t0 = cv2.getTickCount()
        box_draw = None

        if active:
            ok_trk, box = trk.update(frm)
            healthy = True
            if ok_trk and cnt % HEALTH_INT == 0:
                det = mdl(frm, verbose=False, imgsz=320, conf=YOLO_CONF, classes=[cls_id])
                if len(det[0].boxes) > 0:
                    dbox = tuple(map(int, det[0].boxes[0].xyxy[0].cpu().numpy()))
                    dbox = (dbox[0], dbox[1], dbox[2]-dbox[0], dbox[3]-dbox[1])
                    if iou(box, dbox) < IOU_TH: healthy = False
                    else:
                        sz_tgt = (dbox[2], dbox[3])
                        if REID:
                            feat_now = get_hist(frm, dbox)
                            if feat_now is not None:
                                cv2.addWeighted(feat_now, FEAT_RATE, feat0, 1-FEAT_RATE, 0, feat0)
                else: healthy = False
            if ok_trk and healthy:
                w, h = int(sz[0]*(1-SCALE_SMOOTH)+sz_tgt[0]*SCALE_SMOOTH), int(sz[1]*(1-SCALE_SMOOTH)+sz_tgt[1]*SCALE_SMOOTH)
                sz = (w, h)
                cx, cy = int(box[0]+box[2]/2), int(box[1]+box[3]/2)
                box_draw = (cx-w//2, cy-h//2, w, h)
            else:
                active, lost = False, 0

        if not active:
            cv2.putText(frm, "Object lost! Searching...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            if cnt % REDET_INT == 0:
                det = mdl(frm, verbose=False, imgsz=416, conf=YOLO_CONF, classes=[cls_id])

                reacq = None
                if REID:
                    best_score, best_box = -1, None
                    for b in det[0].boxes:
                        cbox = tuple(map(int, b.xyxy[0].cpu().numpy()))
                        cbox = (cbox[0], cbox[1], cbox[2]-cbox[0], cbox[3]-cbox[1])
                        cfeat = get_hist(frm, cbox)
                        if cfeat is None: continue
                        sim = cv2.compareHist(feat0, cfeat, cv2.HISTCMP_CORREL)
                        if sim > best_score: best_score, best_box = sim, cbox
                    if best_box and best_score > REID_TH:
                        reacq = best_box
                        print(f"Re-acquired original target with similarity: {best_score:.2f}")
                else:
                    if len(det[0].boxes) > 0:
                        b = det[0].boxes[0]
                        reacq = tuple(map(int, b.xyxy[0].cpu().numpy()))
                        reacq = (reacq[0], reacq[1], reacq[2]-reacq[0], reacq[3]-reacq[1])
                        print("Re-acquired first available target (Re-ID disabled).")

                if reacq:
                    trk, active = make_trk(TRK), True
                    trk.init(frm, reacq)
                    sz, sz_tgt = (reacq[2], reacq[3]), (reacq[2], reacq[3])
                    box_draw = reacq

        if box_draw:
            p1, p2 = (box_draw[0], box_draw[1]), (box_draw[0]+box_draw[2], box_draw[1]+box_draw[3])
            cv2.rectangle(frm, p1, p2, (0,255,0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - t0)
        fps_sum += fps

        cv2.putText(frm, f"FPS: {int(fps)} ({TRK})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Hybrid Tracking Mode", frm)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if cnt > 0:
        avg_fps = fps_sum / cnt
        print(f"\nHybrid Tracking Mode finished. Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()