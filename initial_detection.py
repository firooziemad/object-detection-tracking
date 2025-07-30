import cv2
import numpy as np
from ultralytics import YOLO

<<<<<<< HEAD
# --- 1. CONFIGURATION ---
# The path to your video file.
VIDEO_PATH = "person4.mp4" 
# The name of the object you want to detect (e.g., "person", "car", "cat").
# Make sure this class is in the COCO dataset, which YOLO was trained on.
TARGET_CLASS_NAME = "person" 
# The YOLO model to use. "yolov8n.pt" is small and fast.
# The model will be downloaded automatically the first time you run this.
MODEL_NAME = "yolov8n.pt" 
=======
VID = "car1.mp4"
CLS = "train"
MOD = "yolov8n.pt"
>>>>>>> 464d17c (multi)

def win(size):
    h, w = size
    wy = np.hanning(h)
    wx = np.hanning(w)
    w2d = np.outer(wy, wx)
    return w2d

def gauss(size, s=8):
    h, w = size
    sx = w / s
    sy = h / s
    y, x = np.mgrid[0:h, 0:w]
    mx = w / 2
    my = h / 2
    g = np.exp(-((x - mx)**2 / (2 * sx**2) + (y - my)**2 / (2 * sy**2)))
    return g

def patch(img, ctr, ext, win_sz):
    cx = float(ctr[0])
    cy = float(ctr[1])
    eh, ew = ext
    wh, ww = win_sz
    x1 = int(cx - ew / 2)
    y1 = int(cy - eh / 2)
    x2 = x1 + ew
    y2 = y1 + eh
    ih, iw, _ = img.shape
    pl = max(0, -x1)
    pt = max(0, -y1)
    pr = max(0, x2 - iw)
    pb = max(0, y2 - ih)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, x2)
    y2 = min(ih, y2)
    p = img[y1:y2, x1:x2]
    if pl > 0 or pt > 0 or pr > 0 or pb > 0:
        p = cv2.copyMakeBorder(p, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    p = cv2.resize(p, (ww, wh), interpolation=cv2.INTER_LINEAR)
    p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    w2d = win((wh, ww))
    p = p * w2d
    m = np.mean(p)
    s = np.std(p)
    if s > 0:
        p = (p - m) / s
    else:
        p = p - m
    return p

def track(vid, box, pad=1.5, sc=[0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03], eta=0.02, lam=1e-5, conf=0.1):
    cap = cv2.VideoCapture(vid)
    ret, frm = cap.read()
    if not ret:
        print("Failed to read video")
        return
    x, y, w, h = box
    ctr = [float(x + w / 2), float(y + h / 2)]
    ow, oh = w, h
    scl = 1.0
    pad_now = pad
    win_sz = [int(h * pad_now), int(w * pad_now)]
    tgt = patch(frm, ctr, win_sz, win_sz)
    g = gauss(win_sz)
    F = np.fft.fft2(tgt)
    G = np.fft.fft2(g)
    H = G * np.conj(F) / (np.abs(F)**2 + lam)
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePost = np.array([[ctr[0]], [ctr[1]], [0], [0]], np.float32)
    while cap.isOpened():
        ret, frm = cap.read()
        if not ret:
            break
        pred = kf.predict()
        pred_ctr = pred[:2, 0].tolist()
        v = np.sqrt(pred[2, 0]**2 + pred[3, 0]**2)
        pad_now = pad + min(1.0, v / 50.0)
        best = -np.inf
        best_s = None
        best_map = None
        for s in sc:
            eff_s = scl * s
            ext = [int(win_sz[0] * eff_s * pad_now), int(win_sz[1] * eff_s * pad_now)]
            srch = patch(frm, pred_ctr, ext, win_sz)
            F2 = np.fft.fft2(srch)
            resp = H * F2
            resp = np.fft.ifft2(resp)
            resp = np.fft.fftshift(np.real(resp))
            mx = np.max(resp)
            if mx > best:
                best = mx
                best_s = s
                best_map = resp
        if best < conf:
            fctr = pred_ctr
            nw = ow * scl
            nh = oh * scl
            nx = fctr[0] - nw / 2
            ny = fctr[1] - nh / 2
        else:
            eff_best = scl * best_s
            my, mx = np.unravel_index(np.argmax(best_map), best_map.shape)
            sy = my - win_sz[0] // 2
            sx = mx - win_sz[1] // 2
            sh = np.array([sx * eff_best, sy * eff_best], dtype=np.float64)
            meas = np.array(pred_ctr) + sh
            kf.correct(meas.reshape(2, 1).astype(np.float32))
            fctr = kf.statePost[:2, 0].tolist()
            scl *= best_s
            nw = ow * scl
            nh = oh * scl
            nx = fctr[0] - nw / 2
            ny = fctr[1] - nh / 2
            ext2 = [int(win_sz[0] * scl * pad_now), int(win_sz[1] * scl * pad_now)]
            tgt2 = patch(frm, fctr, ext2, win_sz)
            F3 = np.fft.fft2(tgt2)
            H2 = G * np.conj(F3) / (np.abs(F3)**2 + lam)
            H = (1 - eta) * H + eta * H2
        cv2.rectangle(frm, (int(nx), int(ny)), (int(nx + nw), int(ny + nh)), (0, 255, 0), 2)
        cv2.putText(frm, f"Conf: {best:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Track', frm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print(f"Loading model {MOD}...")
    mdl = YOLO(MOD)
    print("Model loaded.")
    cap = cv2.VideoCapture(VID)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VID}")
        return
    ok, frm = cap.read()
    if not ok:
        print("Error: Could not read first frame.")
        cap.release()
        return
    print("First frame read.")
    res = mdl(frm, verbose=False)
    r = res[0]
    box = None
    print(f"Searching for '{CLS}' in first frame...")
    for b in r.boxes:
        cid = int(b.cls[0])
        cname = mdl.names[cid]
        if cname.lower() == CLS.lower():
            print(f"Found '{CLS}'!")
            xyxy = b.xyxy[0].cpu().numpy()
            x = int(xyxy[0])
            y = int(xyxy[1])
            w = int(xyxy[2] - xyxy[0])
            h = int(xyxy[3] - xyxy[1])
            box = (x, y, w, h)
            break
    if box is not None:
        print(f"Box: {box}")
        p1 = (box[0], box[1])
        p2 = (box[0] + box[2], box[1] + box[3])
        cv2.rectangle(frm, p1, p2, (0, 255, 0), 2)
        lbl = f"Track: {CLS}"
        cv2.putText(frm, lbl, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detect", frm)
        print("Press any key to start tracking.")
        cv2.waitKey(0)
        track(VID, box)
    else:
        print(f"Could not find '{CLS}' in first frame.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
