import cv2
import numpy as np

class CT:
    def __init__(self):
        self.fp = dict(maxCorners=100, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.bb = None
        self.pg = None
        self.tr = []
        self.tl = 10
        self.di = 5
        self.fi = 0
        self.lc = 0
        self.mlf = 15
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.bbh = []
        self.mh = 3
        self.ssf = 0.25
        self.pbs = None
        self.tm = "normal"
        self.mc = {
            "smooth": {"detect_interval": 8, "max_corners": 50, "quality_level": 0.03, "min_distance": 12, "win_size": (19, 19), "max_level": 2, "max_lost_frames": 20},
            "normal": {"detect_interval": 5, "max_corners": 80, "quality_level": 0.02, "min_distance": 10, "win_size": (15, 15), "max_level": 2, "max_lost_frames": 15, "size_smoothing": 0.08},
            "high_motion": {"detect_interval": 3, "max_corners": 120, "quality_level": 0.015, "min_distance": 8, "win_size": (13, 13), "max_level": 3, "max_lost_frames": 10}
        }

    def init(self, f, bb):
        self.bb = bb
        self.lc = 0
        self.fi = 0
        self.tr = []
        self.bbh = [bb]
        if len(f.shape) == 3:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            g = f
        self.pg = g.copy()
        x, y, w, h = bb
        rs = [(x + w//4, y + h//4, w//2, h//2), (x + w//8, y + h//8, 3*w//4, 3*h//4), (x, y, w, h)]
        tf = 0
        for r in rs:
            rx, ry, rw, rh = r
            m = np.zeros_like(g)
            m[ry:ry+rh, rx:rx+rw] = 255
            rp = self.fp.copy()
            rp['maxCorners'] = 30
            cs = cv2.goodFeaturesToTrack(g, mask=m, **rp)
            if cs is not None:
                for c in cs:
                    self.tr.append([c])
                tf += len(cs)
        x, y, w, h = bb
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        print(f"Initialized with {tf} feature points in {len(rs)} regions")
        return tf > 0

    def set_tm(self, m):
        if m in self.mc:
            self.tm = m
            c = self.mc[m]
            self.di = c["detect_interval"]
            self.mlf = c["max_lost_frames"]
            if "size_smoothing" in c:
                self.ssf = c["size_smoothing"]
            self.fp.update({'maxCorners': c["max_corners"], 'qualityLevel': c["quality_level"], 'minDistance': c["min_distance"]})
            self.lk.update({'winSize': c["win_size"], 'maxLevel': c["max_level"]})
            print(f"Tracking mode set to: {m}")
            return True
        return False

    def lost(self, cp, fs):
        if len(cp) < 5: return True
        h, w = fs[:2]
        xc = cp[:, 0]
        yc = cp[:, 1]
        et = 20
        nl = np.sum(xc < et)
        nr = np.sum(xc > w - et)
        nt = np.sum(yc < et)
        nb = np.sum(yc > h - et)
        ep = nl + nr + nt + nb
        er = ep / len(cp)
        if er > 0.7: return True
        xr = np.max(xc) - np.min(xc)
        yr = np.max(yc) - np.min(yc)
        if xr < 30 or yr < 40: return True
        return False

    def upd(self, f):
        if self.pg is None: return False, None
        if len(f.shape) == 3:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            g = f
        self.fi += 1
        if len(self.tr) > 0:
            p0 = np.float32([t[-1] for t in self.tr]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.pg, g, p0, None, **self.lk)
            gn = p1[st == 1]
            go = p0[st == 1]
            nt = []
            for t, (x, y) in zip(self.tr, p1.reshape(-1, 2)):
                if st[len(nt)] == 1:
                    t.append([(x, y)])
                    if len(t) > self.tl:
                        del t[0]
                    nt.append(t)
            self.tr = nt
            if len(gn) >= 8:
                cp = gn.reshape(-1, 2)
                if self.lost(cp, g.shape):
                    self.lc += 3
                    return False, None
                if len(cp) > 15:
                    c = np.median(cp, axis=0)
                    d = np.sqrt(np.sum((cp - c)**2, axis=1))
                    th = np.median(d) * 2.0
                    vm = d <= th
                    cp = cp[vm]
                if len(cp) >= 5:
                    xc = cp[:, 0]; yc = cp[:, 1]
                    xmin, xmax = np.percentile(xc, 5), np.percentile(xc, 95)
                    ymin, ymax = np.percentile(yc, 5), np.percentile(yc, 95)
                    dw, dh = xmax - xmin, ymax - ymin
                    pw, ph = max(15, dw * 0.25), max(20, dh * 0.30)
                    xmin, ymin = max(0, xmin - pw), max(0, ymin - ph)
                    xmax, ymax = min(g.shape[1], xmax + pw), min(g.shape[0], ymax + ph)
                    nw, nh = xmax - xmin, ymax - ymin
                    if self.pbs is not None:
                        pw0, ph0 = self.pbs
                        mwc = max(5, pw0 * self.ssf)
                        mhc = max(5, ph0 * self.ssf)
                        if abs(nw - pw0) > mwc: nw = pw0 + mwc if nw > pw0 else pw0 - mwc
                        if abs(nh - ph0) > mhc: nh = ph0 + mhc if nh > ph0 else ph0 - mhc
                    self.pbs = (nw, nh)
                    if nw > 15 and nh > 20:
                        if self.lc <= 2:
                            self.kf.predict()
                            m = np.array([xmin, ymin, nw, nh], dtype=np.float32)
                            self.kf.correct(m)
                            s = self.kf.statePost.flatten()
                            sx, sy = max(0, int(s[0])), max(0, int(s[1]))
                            sw, sh = max(15, int(s[2])), max(20, int(s[3]))
                            sw, sh = min(sw, g.shape[1] - sx), min(sh, g.shape[0] - sy)
                            self.bb = (sx, sy, sw, sh)
                            self.bbh.append(self.bb)
                            if len(self.bbh) > self.mh: self.bbh.pop(0)
                            self.lc = 0
                        else: self.lc += 2
                    else: self.lc += 1
                else: self.lc += 1
            else: self.lc += 1
        saf = (self.fi % self.di == 0 or len(self.tr) < 20) and self.bb is not None and self.lc <= 2
        if saf:
            x, y, w, h = self.bb
            m = np.zeros_like(g)
            m[y:y+h, x:x+w] = 255
            for t in self.tr:
                cv2.circle(m, tuple(map(int, t[-1][0])), 8, 0, -1)
            nfp = self.fp.copy()
            nfp['maxCorners'] = 25
            nc = cv2.goodFeaturesToTrack(g, mask=m, **nfp)
            if nc is not None:
                for c in nc:
                    self.tr.append([c])
        self.pg = g.copy()
        suc = self.lc <= self.mlf and len(self.tr) > 5
        if not suc and self.lc > self.mlf:
            self.tr = []
            self.bb = None
        return suc, self.bb

    def draw(self, f):
        if self.bb is not None:
            x, y, w, h = self.bb
            cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for t in self.tr:
            cv2.circle(f, tuple(map(int, t[-1][0])), 2, (0, 255, 0), -1)
            if len(t) > 1:
                pts = np.array([np.int32(p[0]) for p in t]).reshape(-1, 1, 2)
                cv2.polylines(f, [pts], False, (0, 255, 0), 1)
        return f
		
		
		git add custom_tracker.py
git commit -m "Add draw method to tracker for feature completeness"