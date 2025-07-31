import cv2
import numpy as np
from ultralytics import YOLO

A = False
B = False

C = "person1.mp4"
D = "person"
E = "yolov8n.pt"
F = "CSRT"

G = 45
H = 15
I = 0.2
J = 0.5
K = 0.3
L = 0.45
M = 0.05

def N(t):
    if t == 'CSRT': o = cv2.TrackerCSRT_create()
    elif t == 'KCF': o = cv2.TrackerKCF_create()
    elif t == 'MOSSE': o = cv2.legacy.TrackerMOSSE_create()
    else: raise ValueError("Invalid tracker type specified.")
    return o

def O(a, b):
    p, q = max(a[0], b[0]), max(a[1], b[1])
    r, s = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    t = max(0, r-p) * max(0, s-q)
    u, v = a[2]*a[3], b[2]*b[3]
    w = float(u + v - t)
    return t/w if w > 0 else 0

def P(img, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if w <= 0 or h <= 0: return None
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def main():
    a = YOLO(E)
    b = cv2.VideoCapture(C)
    if not b.isOpened(): print(f"Error opening video {C}"); return
    ok, c = b.read()
    if not ok: print("Error reading first frame."); return

    d = list(a.names.keys())[list(a.names.values()).index(D.lower())]
    e = a(c, verbose=False, classes=[d])

    f, g = None, None
    if len(e[0].boxes) > 0:
        h = e[0].boxes[0].xyxy[0].cpu().numpy()
        f = tuple(map(int, [h[0], h[1], h[2]-h[0], h[3]-h[1]]))
        if B:
            g = P(c, f)
            if g is None: print("Initial object has invalid size."); return
            print(f"Found '{D}' and extracted its feature signature.")

    if not f: print("Target not found in first frame."); return

    i = N(F)
    i.init(c, f)

    if A:
        Q(b, i, a, f, g)
    else:
        R(b, i)

    b.release()
    cv2.destroyAllWindows()

def R(b, i):
    j = 0
    k = 0

    while True:
        ok, c = b.read()
        if not ok: break
        l = cv2.getTickCount()
        ok_trk, box = i.update(c)
        if ok_trk:
            m, n = (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3]))
            cv2.rectangle(c, m, n, (0,255,0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - l)
        j += fps
        k += 1

        cv2.putText(c, f"FPS: {int(fps)} ({F})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Pure Tracking Mode", c)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if k > 0:
        avg_fps = j / k
        print(f"\nPure Tracking Mode finished. Average FPS: {avg_fps:.2f}")

def Q(b, i, a, f, g):
    o, p, q, r = True, 0, 0, 0
    s, t = (f[2], f[3]), (f[2], f[3])
    u = list(a.names.keys())[list(a.names.values()).index(D.lower())]

    while True:
        ok, c = b.read()
        if not ok: break
        q += 1
        v = cv2.getTickCount()
        w = None

        if o:
            ok_trk, box = i.update(c)
            x = True
            if ok_trk and q % G == 0:
                det = a(c, verbose=False, imgsz=320, conf=J, classes=[u])
                if len(det[0].boxes) > 0:
                    dbox = tuple(map(int, det[0].boxes[0].xyxy[0].cpu().numpy()))
                    dbox = (dbox[0], dbox[1], dbox[2]-dbox[0], dbox[3]-dbox[1])
                    if O(box, dbox) < K: x = False
                    else:
                        t = (dbox[2], dbox[3])
                        if B:
                            feat_now = P(c, dbox)
                            if feat_now is not None:
                                cv2.addWeighted(feat_now, M, g, 1-M, 0, g)
                else: x = False
            if ok_trk and x:
                w_, h_ = int(s[0]*(1-I)+t[0]*I), int(s[1]*(1-I)+t[1]*I)
                s = (w_, h_)
                cx, cy = int(box[0]+box[2]/2), int(box[1]+box[3]/2)
                w = (cx-w_//2, cy-h_//2, w_, h_)
            else:
                o, p = False, 0

        if not o:
            cv2.putText(c, "Object lost! Searching...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            if q % H == 0:
                det = a(c, verbose=False, imgsz=416, conf=J, classes=[u])

                reacq = None
                if B:
                    best_score, best_box = -1, None
                    for b_ in det[0].boxes:
                        cbox = tuple(map(int, b_.xyxy[0].cpu().numpy()))
                        cbox = (cbox[0], cbox[1], cbox[2]-cbox[0], cbox[3]-cbox[1])
                        cfeat = P(c, cbox)
                        if cfeat is None: continue
                        sim = cv2.compareHist(g, cfeat, cv2.HISTCMP_CORREL)
                        if sim > best_score: best_score, best_box = sim, cbox
                    if best_box and best_score > L:
                        reacq = best_box
                        print(f"Re-acquired original target with similarity: {best_score:.2f}")
                else:
                    if len(det[0].boxes) > 0:
                        b_ = det[0].boxes[0]
                        reacq = tuple(map(int, b_.xyxy[0].cpu().numpy()))
                        reacq = (reacq[0], reacq[1], reacq[2]-reacq[0], reacq[3]-reacq[1])
                        print("Re-acquired first available target (Re-ID disabled).")

                if reacq:
                    i, o = N(F), True
                    i.init(c, reacq)
                    s, t = (reacq[2], reacq[3]), (reacq[2], reacq[3])
                    w = reacq

        if w:
            m, n = (w[0], w[1]), (w[0]+w[2], w[1]+w[3])
            cv2.rectangle(c, m, n, (0,255,0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - v)
        r += fps

        cv2.putText(c, f"FPS: {int(fps)} ({F})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Hybrid Tracking Mode", c)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if q > 0:
        avg_fps = r / q
        print(f"\nHybrid Tracking Mode finished. Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()