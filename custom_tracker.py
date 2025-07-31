import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.fp = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )
        self.lk = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.box = None
        self.prev = None
        self.tracks = []
        self.track_len = 10
        self.interval = 5
        self.frame_idx = 0
        self.lost_count = 0
        self.max_lost = 15
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0.95, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.95, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.9, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.9]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.005
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.15
        self.box_hist = []
        self.max_hist = 5
        self.size_smooth = 0.12
        self.prev_size = None
        self.box_buf = []
        self.buf_size = 3
        self.conf_thr = 0.7
        self.mode = "normal"
        self.modes = {
            "smooth": {
                "detect_interval": 8,
                "max_corners": 60,
                "quality_level": 0.025,
                "min_distance": 15,
                "win_size": (25, 25),
                "max_level": 2,
                "max_lost_frames": 25,
                "size_smoothing": 0.08,
                "position_smoothing": 0.8,
                "kalman_process_noise": 0.003,
                "kalman_measurement_noise": 0.2
            },
            "normal": {
                "detect_interval": 5,
                "max_corners": 80,
                "quality_level": 0.02,
                "min_distance": 10,
                "win_size": (19, 19),
                "max_level": 2,
                "max_lost_frames": 15,
                "size_smoothing": 0.12,
                "position_smoothing": 0.6,
                "kalman_process_noise": 0.005,
                "kalman_measurement_noise": 0.15
            },
            "high_motion": {
                "detect_interval": 3,
                "max_corners": 120,
                "quality_level": 0.015,
                "min_distance": 8,
                "win_size": (15, 15),
                "max_level": 3,
                "max_lost_frames": 10,
                "size_smoothing": 0.18,
                "position_smoothing": 0.4,
                "kalman_process_noise": 0.01,
                "kalman_measurement_noise": 0.1
            }
        }
        self.pos_hist = []
        self.pos_hist_size = 10
        self.vel_hist = []
        self.vel_hist_size = 5

    def init(self, img, box):
        self.box = box
        self.lost_count = 0
        self.frame_idx = 0
        self.tracks = []
        self.box_hist = [box]
        self.box_buf = [box]
        self.pos_hist = [box[:2]]
        self.vel_hist = []
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        self.prev = gray.copy()
        x, y, w, h = box
        regions = [
            (x + w//6, y + h//6, 2*w//3, 2*h//3),
            (x + w//8, y + h//8, 3*w//4, 3*h//4),
            (x + w//12, y + h//12, 5*w//6, 5*h//6),
            (x, y, w, h)
        ]
        total = 0
        for i, reg in enumerate(regions):
            rx, ry, rw, rh = reg
            mask = np.zeros_like(gray)
            mask[ry:ry+rh, rx:rx+rw] = 255
            fp_params = self.fp.copy()
            fp_params['maxCorners'] = [40, 30, 25, 20][i]
            fp_params['qualityLevel'] = [0.03, 0.025, 0.02, 0.015][i]
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **fp_params)
            if corners is not None:
                for c in corners:
                    self.tracks.append([c])
                total += len(corners)
        x, y, w, h = box
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.prev_size = (w, h)
        print(f"Init: {total} features in {len(regions)} regions")
        return total > 0

    def set_mode(self, mode):
        if mode in self.modes:
            self.mode = mode
            cfg = self.modes[mode]
            self.interval = cfg["detect_interval"]
            self.max_lost = cfg["max_lost_frames"]
            self.size_smooth = cfg["size_smoothing"]
            self.fp.update({
                'maxCorners': cfg["max_corners"],
                'qualityLevel': cfg["quality_level"],
                'minDistance': cfg["min_distance"]
            })
            self.lk.update({
                'winSize': cfg["win_size"],
                'maxLevel': cfg["max_level"]
            })
            self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * cfg["kalman_process_noise"]
            self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * cfg["kalman_measurement_noise"]
            print(f"Mode set: {mode}")
            return True
        return False

    def outliers(self, pts):
        if len(pts) < 8:
            return pts
        center = np.median(pts, axis=0)
        dists = np.sqrt(np.sum((pts - center)**2, axis=1))
        q1 = np.percentile(dists, 25)
        q3 = np.percentile(dists, 75)
        iqr = q3 - q1
        lb = q1 - 1.2 * iqr
        ub = q3 + 1.2 * iqr
        mask = (dists >= lb) & (dists <= ub)
        return pts[mask]

    def predict(self):
        if len(self.vel_hist) < 2:
            return None
        v = np.mean(self.vel_hist[-3:], axis=0)
        if self.box is not None:
            x, y, w, h = self.box
            px = x + v[0]
            py = y + v[1]
            return (px, py, w, h)
        return None

    def smooth(self, new_box, conf=1.0):
        if self.box is None:
            return new_box
        x, y, w, h = self.box
        nx, ny, nw, nh = new_box
        cfg = self.modes[self.mode]
        pos_smooth = cfg.get("position_smoothing", 0.5)
        mag = np.sqrt((nx - x)**2 + (ny - y)**2)
        af = min(1.0, mag / 50.0)
        sf = pos_smooth * conf * (1.0 - af * 0.3)
        sx = x + (nx - x) * sf
        sy = y + (ny - y) * sf
        sw = w + (nw - w) * sf * 0.7
        sh = h + (nh - h) * sf * 0.7
        return (int(sx), int(sy), int(sw), int(sh))

    def update(self, img):
        if self.prev is None:
            return False, None
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        self.frame_idx += 1
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev, gray, p0, None, **self.lk
            )
            good_idx = []
            good_new = []
            good_old = []
            for i, (tr, np1, s, e) in enumerate(zip(self.tracks, p1, st, err)):
                if s == 1 and e < 50:
                    good_idx.append(i)
                    good_new.append(np1)
                    good_old.append(p0[i])
            new_tracks = []
            for i, np1 in zip(good_idx, good_new):
                tr = self.tracks[i]
                tr.append([np1[0]])
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
            self.tracks = new_tracks
            if len(good_new) >= 8:
                pts = np.array(good_new).reshape(-1, 2)
                fpts = self.outliers(pts)
                if len(fpts) >= 6:
                    xs = fpts[:, 0]
                    ys = fpts[:, 1]
                    xmin = np.percentile(xs, 8)
                    xmax = np.percentile(xs, 92)
                    ymin = np.percentile(ys, 8)
                    ymax = np.percentile(ys, 92)
                    nw = xmax - xmin
                    nh = ymax - ymin
                    if self.mode == "smooth":
                        pw = max(20, nw * 0.35)
                        ph = max(25, nh * 0.40)
                    elif self.mode == "normal":
                        pw = max(15, nw * 0.25)
                        ph = max(20, nh * 0.30)
                    else:
                        pw = max(10, nw * 0.20)
                        ph = max(15, nh * 0.25)
                    xmin = max(0, xmin - pw)
                    ymin = max(0, ymin - ph)
                    xmax = min(gray.shape[1], xmax + pw)
                    ymax = min(gray.shape[0], ymax + ph)
                    nw = xmax - xmin
                    nh = ymax - ymin
                    if self.prev_size is not None:
                        pw0, ph0 = self.prev_size
                        sc = abs(nw - pw0) + abs(nh - ph0)
                        asf = self.size_smooth * (1 + sc / 100.0)
                        asf = min(asf, 0.3)
                        mwc = max(8, pw0 * asf)
                        mhc = max(8, ph0 * asf)
                        if abs(nw - pw0) > mwc:
                            nw = pw0 + np.sign(nw - pw0) * mwc
                        if abs(nh - ph0) > mhc:
                            nh = ph0 + np.sign(nh - ph0) * mhc
                    self.prev_size = (nw, nh)
                    if nw > 20 and nh > 25:
                        self.kf.predict()
                        meas = np.array([xmin, ymin, nw, nh], dtype=np.float32)
                        self.kf.correct(meas)
                        st = self.kf.statePost.flatten()
                        kx = max(0, int(st[0]))
                        ky = max(0, int(st[1]))
                        kw = max(20, int(st[2]))
                        kh = max(25, int(st[3]))
                        kw = min(kw, gray.shape[1] - kx)
                        kh = min(kh, gray.shape[0] - ky)
                        nbox = (kx, ky, kw, kh)
                        conf = min(1.0, len(fpts) / 50.0)
                        self.box = self.smooth(nbox, conf)
                        self.box_hist.append(self.box)
                        if len(self.box_hist) > self.max_hist:
                            self.box_hist.pop(0)
                        if len(self.pos_hist) > 0:
                            prevp = self.pos_hist[-1]
                            curp = self.box[:2]
                            v = (curp[0] - prevp[0], curp[1] - prevp[1])
                            self.vel_hist.append(v)
                            if len(self.vel_hist) > self.vel_hist_size:
                                self.vel_hist.pop(0)
                        self.pos_hist.append(self.box[:2])
                        if len(self.pos_hist) > self.pos_hist_size:
                            self.pos_hist.pop(0)
                        self.lost_count = 0
                        if self.frame_idx % 30 == 0:
                            print(f"Track: {len(fpts)} features, box: {kw}x{kh}")
                    else:
                        self.lost_count += 1
                else:
                    self.lost_count += 1
            else:
                self.lost_count += 1
                if len(good_new) > 0:
                    print(f"Few features: {len(good_new)}")
        if (self.frame_idx % self.interval == 0 or len(self.tracks) < 25) and self.box is not None:
            x, y, w, h = self.box
            mask = np.zeros_like(gray)
            im = min(w//8, h//8, 15)
            mask[y+im:y+h-im, x+im:x+w-im] = 255
            for tr in self.tracks:
                cv2.circle(msk, tuple(map(int, tr[-1][0])), 12, 0, -1)
            nfp = self.fp.copy()
            nfp['maxCorners'] = 30
            nfp['qualityLevel'] *= 1.2
            nc = cv2.goodFeaturesToTrack(gray, mask=msk, **nfp)
            if nc is not None:
                for c in nc:
                    self.trks.append([c])
                if self.idx % 60 == 0:
                    print(f"Add features: {len(nc)} new")
        self.prev = gray.copy()
        ok = (self.lost <= self.max_lost and 
              len(self.trks) > 8 and 
              self.box is not None)
        return ok, self.box

    def draw(self, img):
        if self.box is not None:
            x, y, w, h = self.box
            th = max(2, min(4, len(self.trks) // 20))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), th)
            q = min(100, len(self.trks))
            qc = (0, int(255 * q / 100), int(255 * (1 - q / 100)))
            cv2.rectangle(img, (x, y-25), (x + int(w * q / 100), y-15), qc, -1)
            txt = f"{self.mode.upper()}: {len(self.trks)} pts"
            cv2.putText(img, txt, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, tr in enumerate(self.trks):
            age = len(tr)
            if age > 7:
                col = (0, 255, 0)
            elif age > 3:
                col = (0, 255, 255)
            else:
                col = (0, 128, 255)
            cv2.circle(img, tuple(map(int, tr[-1][0])), 3, col, -1)
            if len(tr) > 3:
                pts = np.array([np.int32(p[0]) for p in tr[-5:]]).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, col, 1)
        return img