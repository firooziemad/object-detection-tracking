import cv2
import numpy as np
from ultralytics import YOLO
from custom_tracker import Tracker

import sys
import time
import argparse

VID = "input1.mp4"
OBJ = "car"
MOD = "yolov8n.pt"
CONF = 0.6
SRCH_FR = 20
RES = 600
RE_DIST = 250
MARG = {
    "top": 30,
    "bottom": 30,
    "left": 30,
    "right": 30
}

CLS = {
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
PRESET = {
    "person": {"min_area": 500, "redetect_min_area": 200, "tracking_mode": "normal"},
    "car": {"min_area": 100, "redetect_min_area": 50, "tracking_mode": "high_motion"},
    "dog": {"min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "high_motion"},
    "cat": {"min_area": 2000, "redetect_min_area": 1000, "tracking_mode": "smooth"},
    "bicycle": {"min_area": 6000, "redetect_min_area": 2500, "tracking_mode": "normal"},
    "motorcycle": {"min_area": 7000, "redetect_min_area": 3000, "tracking_mode": "high_motion"},
    "bottle": {"min_area": 1000, "redetect_min_area": 500, "tracking_mode": "smooth"},
    "laptop": {"min_area": 4000, "redetect_min_area": 2000, "tracking_mode": "smooth"}
}

def parse_arguments():
    p = argparse.ArgumentParser(description='Advanced Multi-Object Tracking System with Edge Detection')
    p.add_argument('--object', '-o', type=str, help='Object to track (overrides manual config)')
    p.add_argument('--video', '-v', type=str, help='Video path (overrides manual config)')
    p.add_argument('--confidence', '-c', type=float, help='YOLO confidence threshold (0.1-1.0)')
    p.add_argument('--mode', '-m', type=str, choices=['smooth', 'normal', 'high_motion'], help='Tracking mode')
    p.add_argument('--list', '-l', action='store_true', help='List available objects and exit')
    p.add_argument('--margin-top', type=int, help='Top edge margin in pixels')
    p.add_argument('--margin-bottom', type=int, help='Bottom edge margin in pixels')
    p.add_argument('--margin-left', type=int, help='Left edge margin in pixels')
    p.add_argument('--margin-right', type=int, help='Right edge margin in pixels')
    p.add_argument('--margin-all', type=int, help='Set all margins to same value')
    p.add_argument('--reentry-distance', type=int, help='Maximum pixels for re-entry acceptance (default: 100)')
    return p.parse_args()

def is_bbox_in_margin(b, f_shape, m):
    if b is None:
        return False
    x, y, w, h = b
    fh, fw = f_shape[:2]
    iw = w // 2
    ih = h // 2
    ix = x + (w - iw) // 2
    iy = y + (h - ih) // 2
    t = iy < m["top"]
    btm = (iy + ih) > (fh - m["bottom"])
    l = ix < m["left"]
    r = (ix + iw) > (fw - m["right"])
    return t or btm or l or r

def calculate_bbox_features(f, b):
    if b is None:
        return None
    x, y, w, h = b
    roi = f[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    if len(roi.shape) == 3:
        roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_g = roi
    feat = {
        'size': (w, h),
        'aspect_ratio': w / h if h > 0 else 1.0,
        'histogram': cv2.calcHist([roi_g], [0], None, [32], [0, 256]),
        'mean_intensity': np.mean(roi_g),
        'center': (x + w//2, y + h//2)
    }
    return feat

def compare_bbox_features(f1, f2, th=0.7):
    if f1 is None or f2 is None:
        return False
    w1, h1 = f1['size']
    w2, h2 = f2['size']
    sz = min(w1*h1, w2*h2) / max(w1*h1, w2*h2) if max(w1*h1, w2*h2) > 0 else 0
    ar = abs(f1['aspect_ratio'] - f2['aspect_ratio'])
    ar_s = max(0, 1 - ar)
    hc = cv2.compareHist(f1['histogram'], f2['histogram'], cv2.HISTCMP_CORREL)
    idiff = abs(f1['mean_intensity'] - f2['mean_intensity']) / 255.0
    isim = max(0, 1 - idiff)
    score = (sz * 0.3 + ar_s * 0.2 + hc * 0.3 + isim * 0.2)
    return score >= th

def get_bbox_center(b):
    x, y, w, h = b
    return (x + w/2, y + h/2)

def get_bbox_distance(b1, b2):
    c1 = get_bbox_center(b1)
    c2 = get_bbox_center(b2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def get_bbox_size_similarity(b1, b2):
    _, _, w1, h1 = b1
    _, _, w2, h2 = b2
    a1 = w1 * h1
    a2 = w2 * h2
    if a1 == 0 or a2 == 0:
        return 0
    r = min(a1, a2) / max(a1, a2)
    return r

def find_best_matching_detection(cur, dets, hist):
    best = -1
    match = None
    ref = hist[-1] if len(hist) > 0 else cur
    for d in dets:
        xyxy = d.xyxy[0].cpu().numpy()
        conf = d.conf[0].cpu().numpy()
        nb = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
        dist = get_bbox_distance(ref, nb)
        sz = get_bbox_size_similarity(ref, nb)
        md = 300
        ds = max(0, 1 - (dist / md))
        score = (ds * 0.4) + (sz * 0.4) + (conf * 0.2)
        if score > best:
            best = score
            match = d
    if best > 0.5:
        return match
    return None

def get_best_detection(boxes, min_a, cid=None):
    best_b = None
    best_s = 0
    best_cid = None
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        conf = b.conf[0].cpu().numpy()
        clsid = int(b.cls[0].cpu().numpy())
        if cid is not None and clsid != cid:
            continue
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        a = w * h
        if a > min_a and conf > best_s:
            best_s = conf
            best_b = b
            best_cid = clsid
    return best_b, best_s, best_cid

def get_class_name(cid):
    for n, i in CLS.items():
        if i == cid:
            return n
    return f"class_{cid}"

def list_available_objects():
    print("\nAvailable preset objects:")
    for n, c in PRESET.items():
        m = c.get('tracking_mode', 'normal')
        print(f"  '{n}' -> {m} mode")
    print(f"\nAll YOLO classes ({len(CLS)} total):")
    for i, (n, cid) in enumerate(CLS.items()):
        if i % 4 == 0:
            print()
        print(f"  {n:15} (ID:{cid:2d})", end="")
    print("\n")

def draw_margin_lines(f, m):
    h, w = f.shape[:2]
    o = f.copy()
    cv2.line(o, (0, m["top"]), (w, m["top"]), (255, 255, 0), 2)
    cv2.line(o, (0, h - m["bottom"]), (w, h - m["bottom"]), (255, 255, 0), 2)
    cv2.line(o, (m["left"], 0), (m["left"], h), (255, 255, 0), 2)
    cv2.line(o, (w - m["right"], 0), (w - m["right"], h), (255, 255, 0), 2)
    cv2.addWeighted(o, 0.3, f, 0.7, 0, f)

def main():
    a = parse_arguments()
    if a.list:
        list_available_objects()
        return
    obj = a.object if a.object else OBJ
    vid = a.video if a.video else VID
    conf = a.confidence if a.confidence else CONF
    marg = MARG.copy()
    if a.margin_all:
        marg = {k: a.margin_all for k in marg.keys()}
    else:
        if a.margin_top: marg["top"] = a.margin_top
        if a.margin_bottom: marg["bottom"] = a.margin_bottom
        if a.margin_left: marg["left"] = a.margin_left
        if a.margin_right: marg["right"] = a.margin_right
    re_dist = a.reentry_distance if a.reentry_distance else RE_DIST
    if obj.lower() in CLS:
        cid = CLS[obj.lower()]
    else:
        print(f"Error: Unknown object '{obj}'")
        list_available_objects()
        return
    ocfg = PRESET.get(obj.lower(), {
        "min_area": 3000, "redetect_min_area": 1500, "tracking_mode": "normal"
    })
    print(f"Configuration:")
    print(f"  Target: {obj} (class_id: {cid})")
    print(f"  Video: {vid}")
    print(f"  Confidence: {conf}")
    print(f"  Tracking Mode: {ocfg['tracking_mode']}")
    print(f"  Edge Margins: {marg}")
    print(f"  Max Re-entry Distance: {re_dist}px")
    print("\nLoading YOLO model...")
    try:
        model = YOLO(MOD)
        model.overrides['verbose'] = False
        model.overrides['device'] = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        print(f"Using device: {model.overrides['device']}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"Error opening video {vid}")
        return
    tfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    ftime = 1.0 / fps
    ibox = None
    iframe = None
    iconf = 0.0
    print(f"\nSearching for '{obj}' in the first {SRCH_FR} frames...")
    for fn in range(SRCH_FR):
        ok, f = cap.read()
        if not ok:
            print(f"  -> End of video or read error at frame {fn + 1}.")
            cap.release()
            return
        print(f"  Analyzing frame {fn + 1}/{SRCH_FR}...")
        res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf)
        best_b, bconf, bcid = get_best_detection(
            res[0].boxes, ocfg['min_area'], cid
        )
        if best_b is not None:
            xyxy = best_b.xyxy[0].cpu().numpy()
            ibox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
            iframe = f
            iconf = bconf
            print(f"Initial detection successful in frame {fn + 1}: {ibox} (confidence: {iconf:.3f})")
            break
    if ibox is None:
        print(f"\nCould not find a suitable '{obj}' in the first {SRCH_FR} frames.")
        cap.release()
        return
    f = iframe
    trk = CustomTracker()
    tmode = a.mode if a.mode else ocfg['tracking_mode']
    trk.set_tracking_mode(tmode)
    if not trk.init(f, ibox):
        print("Failed to initialize main tracker")
        cap.release()
        return
    aux = None
    trk_types = ['MIL']
    for tt in trk_types:
        try:
            if hasattr(cv2, f'Tracker{tt}_create'):
                aux = getattr(cv2, f'Tracker{tt}_create')()
                aux.init(f, ibox)
                print(f"Initialized auxiliary {tt} tracker")
                break
            elif hasattr(cv2.legacy, f'Tracker{tt}_create'):
                aux = getattr(cv2.legacy, f'Tracker{tt}_create')()
                aux.init(f, ibox)
                print(f"Initialized auxiliary {tt} tracker (legacy)")
                break
        except Exception as e:
            print(f"Failed to initialize {tt} tracker: {e}")
            continue
    aux_ok = aux is not None
    ref_feat = calculate_bbox_features(f, ibox)
    hist = [ibox]
    aux_hist = [ref_feat]
    print(f"\nControls:")
    print("  'q' - quit, 'p' - pause")
    print("  'r' - correct current tracker")
    print("  'd' - hard reset detection to best new object")
    print("  's/n/h' - tracking modes, 'f' - toggle FPS limit")
    fc = 0
    paused = False
    show_m = False
    last_t = time.time()
    fps_lim = False
    act_fps = 0
    fps_st = time.time()
    fps_fc = 0
    auto_redet = True
    redet_c = 0
    in_marg = False
    last_box = ibox
    pend_ver = False
    ver_c = 0
    pend_box = None
    aux_up_c = 0
    bg_det_c = 0
    exit_pos = None
    box = ibox
    trk_ok = True
    while True:
        if not paused:
            if fc > 0:
                ok, f = cap.read()
                if not ok:
                    print("End of video or read error")
                    break
            fc += 1
        if in_marg:
            bg_det_c += 1
            if bg_det_c >= 5:
                bg_det_c = 0
                res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf * 0.7)
                if len(res[0].boxes) > 0:
                    best_match = None
                    best_score = 0
                    for bd in res[0].boxes:
                        xyxy = bd.xyxy[0].cpu().numpy()
                        det_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                        if not is_bbox_in_margin(det_box, f.shape, marg):
                            det_c = get_bbox_center(det_box)
                            dist_exit = float('inf')
                            if exit_pos is not None:
                                dist_exit = np.sqrt((det_c[0] - exit_pos[0])**2 + (det_c[1] - exit_pos[1])**2)
                            if dist_exit <= re_dist:
                                det_feat = calculate_bbox_features(f, det_box)
                                mscore = 0
                                if aux_ok and len(aux_hist) > 0:
                                    for af in aux_hist[-3:]:
                                        if compare_bbox_features(det_feat, af, 0.5):
                                            mscore += 1
                                else:
                                    dist = get_bbox_distance(last_box, det_box)
                                    sz = get_bbox_size_similarity(last_box, det_box)
                                    if dist < 150 and sz > 0.5:
                                        mscore = 2
                                if mscore > best_score:
                                    best_score = mscore
                                    best_match = bd
                    if best_match is not None and best_score >= 1:
                        xyxy = best_match.xyxy[0].cpu().numpy()
                        new_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                        pend_ver = True
                        ver_c = 1
                        pend_box = new_box
                        in_marg = False
                        print(f"Re-entry candidate found. Starting 3-frame verification...")
        elif pend_ver:
            res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf * 0.7)
            best_match = None
            if len(res[0].boxes) > 0:
                best_match = find_best_matching_detection(pend_box, res[0].boxes, [pend_box])
            if best_match is not None:
                ver_c += 1
                xyxy = best_match.xyxy[0].cpu().numpy()
                pend_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                print(f"Verification success: {ver_c}/3")
                if ver_c >= 3:
                    print("Verification successful! Resuming normal tracking.")
                    if trk.init(f, pend_box):
                        if aux_ok:
                            try:
                                aux.init(f, pend_box)
                            except Exception as e:
                                print(f"Failed to reinitialize auxiliary tracker: {e}")
                        last_box = pend_box
                        box = pend_box
                        trk_ok = True
                    pend_ver = False
                    ver_c = 0
                    pend_box = None
            else:
                print("Verification failed. Returning to search mode.")
                pend_ver = False
                ver_c = 0
                pend_box = None
                in_marg = True
        else:
            if not paused:
                trk_ok, box = trk.update(f)
            cur_in_marg = is_bbox_in_margin(box, f.shape, marg)
            if trk_ok and box is not None and not cur_in_marg and aux_ok:
                aux_up_c += 1
                if aux_up_c >= 5:
                    aux_up_c = 0
                    try:
                        aux_ok2, aux_box = aux.update(f)
                        if aux_ok2:
                            aux_feat = calculate_bbox_features(f, aux_box)
                            if aux_feat:
                                aux_hist.append(aux_feat)
                                if len(aux_hist) > 10:
                                    aux_hist.pop(0)
                    except Exception as e:
                        aux_ok = False
                last_box = box
            if cur_in_marg and not in_marg:
                if box is not None:
                    exit_pos = get_bbox_center(box)
                in_marg = True
                bg_det_c = 0
            if auto_redet and not in_marg and trk_ok and box is not None:
                hist.append(box)
                if len(hist) > 20:
                    hist.pop(0)
                redet_c += 1
                if redet_c >= 8:
                    redet_c = 0
                    res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf * 0.78)
                    if len(res[0].boxes) > 0:
                        best_match = find_best_matching_detection(box, res[0].boxes, hist)
                        if best_match is not None:
                            xyxy = best_match.xyxy[0].cpu().numpy()
                            conf2 = best_match.conf[0].cpu().numpy()
                            new_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                            dist = get_bbox_distance(box, new_box)
                            sz = get_bbox_size_similarity(box, new_box)
                            if dist > 40 or sz < 0.75:
                                if trk.init(f, new_box):
                                    print(f"Auto-Correction successful: Dist: {dist:.1f}px, SizeSim: {sz:.2f}")
                                    box = new_box
        disp_f = f.copy()
        if show_m:
            draw_margin_lines(disp_f, marg)
        if pend_ver:
            status = f"Verifying Re-entry... ({ver_c}/3)"
            cv2.putText(disp_f, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif trk_ok and box is not None and not in_marg:
            x, y, w, h = box
            cv2.rectangle(disp_f, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(disp_f, f"Tracking: {len(trk.tracks)} features", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif in_marg:
            cv2.putText(disp_f, "Object in margin - Searching...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(disp_f, f"TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        info_y = 30
        cv2.putText(disp_f, f"Target: {obj}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        fps_col = (0, 255, 255) if fps_lim else (255, 255, 255)
        fps_txt = f"FPS: {act_fps:.1f}" + (" (Limited)" if fps_lim else "")
        cv2.putText(disp_f, fps_txt, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_col, 1)
        if paused:
            cv2.putText(disp_f, "PAUSED", (10, disp_f.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(f"Enhanced Tracking - {obj}", disp_f)
        if not paused:
            now = time.time()
            fps_fc += 1
            if now - fps_st >= 1.0:
                act_fps = fps_fc / (now - fps_st)
                fps_fc = 0
                fps_st = now
            if fps_lim:
                elap = now - last_t
                wait = max(0, ftime - elap)
                if wait > 0:
                    time.sleep(wait)
            last_t = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            paused = not paused
        elif k == ord('m'):
            show_m = not show_m
        elif k == ord('f'):
            fps_lim = not fps_lim
            print(f"FPS Limiter: {'ON' if fps_lim else 'OFF'}")
        elif k == ord(' ') and paused:
            ok, f = cap.read()
            if ok:
                fc += 1
            else:
                break
        elif k == ord('r'):
            print("Manual re-detection (correction)...")
            res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf * 0.7)
            if len(res[0].boxes) > 0 and box is not None:
                best_match = find_best_matching_detection(box, res[0].boxes, hist)
                if best_match is not None:
                    xyxy = best_match.xyxy[0].cpu().numpy()
                    conf2 = best_match.conf[0].cpu().numpy()
                    det_cls = int(best_match.cls[0].cpu().numpy())
                    new_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                    if trk.init(f, new_box):
                        if aux_ok:
                            try:
                                aux.init(f, new_box)
                            except Exception as e:
                                print(f"Failed to reinitialize auxiliary tracker: {e}")
                        det_name = get_class_name(det_cls)
                        print(f"Correction successful: {det_name} at {new_box} (conf: {conf2:.3f})")
                        in_marg = False
                        last_box = new_box
                        box = new_box
                else:
                    print(f"No suitable matching {obj} found.")
            else:
                print("No detections found for manual override.")
        elif k == ord('d'):
            print("Hard re-detection initiated: Searching for best new object...")
            res = model(f, imgsz=RES, verbose=False, classes=[cid], conf=conf)
            best_b, bconf, bcid = get_best_detection(
                res[0].boxes,
                ocfg['min_area'],
                cid
            )
            if best_b is not None:
                xyxy = best_b.xyxy[0].cpu().numpy()
                new_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))
                if trk.init(f, new_box):
                    if aux_ok:
                        try:
                            aux.init(f, new_box)
                        except Exception as e:
                            print(f"Failed to reinitialize auxiliary tracker: {e}")
                    det_name = get_class_name(bcid)
                    print(f"Hard reset successful: New target is {det_name} at {new_box} (conf: {bconf:.3f})")
                    in_marg = False
                    last_box = new_box
                    box = new_box
                    hist = [new_box]
                    aux_hist = [calculate_bbox_features(f, new_box)]
                else:
                    print("Failed to initialize tracker on new object.")
            else:
                print(f"No suitable '{obj}' found on screen for hard reset.")
        elif k == ord('s'):
            trk.set_tracking_mode("smooth")
        elif k == ord('h'):
            trk.set_tracking_mode("high_motion")
        elif k == ord('n'):
            trk.set_tracking_mode("normal")
        try:
            if cv2.getWindowProperty(f"Enhanced Tracking - {obj}", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    print(f"Processed {fc} frames")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()