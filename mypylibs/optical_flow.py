"""Step 2: sparse Lucas-Kanade optical flow on the moving object.

Every frame yields one *measurement* of the object (position + velocity in
pixels / frame), or ``valid=False`` when the object cannot be measured (e.g. it
is behind the occluder). These measurements are what the Kalman filter will
consume in Step 3.

Pipeline per frame
------------------
1. Track the current keypoints with pyramidal LK (``cv.calcOpticalFlowPyrLK``)
   and run it backwards again (forward-backward check). Points whose
   round-trip error is large, or whose patch residual is high, are dropped:
   that is what happens when a point slides onto the occluder.
2. Seed new keypoints with ``cv.goodFeaturesToTrack`` but only inside a
   *motion mask* (frame differencing), so we never seed on the static
   background, the occluder or watermarks.
3. A freshly seeded point is a *candidate*. It is promoted to an *object*
   point once it has actually moved ``promote_px`` pixels; candidates that
   never move are discarded after ``candidate_ttl`` frames. Object points are
   kept even if the object later slows down or stops.
4. Object points are grouped into spatial clusters; the cluster with the most
   motion (sum of smoothed point speeds) is taken as *the* object. This keeps a
   retreating hand or flickering background from dragging the estimate.
   Within that cluster, points whose velocity disagrees with the median
   (median/MAD test, with a tolerance relative to the speed because points on
   a rolling ball move at anything from 0 to 2v) are rejected as outliers.
5. If enough points survive, the measurement is the median position and
   median velocity of the object cluster, with their standard deviations
   (useful later for R / P in the Kalman filter).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict

import cv2 as cv
import numpy as np


@dataclass
class OFConfig:
    # Lucas-Kanade
    win_size: int = 21          # LK search window (px)
    max_level: int = 3          # pyramid levels -> handles up to ~2^3 * win/2 px motion
    lk_iters: int = 30
    lk_eps: float = 0.01
    fb_max_err: float = 1.0     # max forward-backward round-trip error (px)
    lk_max_residual: float = 25.0  # max LK patch residual (mean abs intensity diff)

    # goodFeaturesToTrack
    max_corners: int = 150
    quality_level: float = 0.005  # low: smooth objects (potato, ball) only have weak corners
    min_distance: int = 7
    block_size: int = 7
    reseed_every: int = 1       # try to add points every n frames
    target_points: int = 60     # don't seed if we already have this many *object* points

    # Motion mask used for seeding
    motion_gap: int = 2         # compare frame t with frame t-gap
    motion_thresh: int = 15     # abs-diff threshold (grey levels)
    motion_dilate: int = 25     # dilation kernel size (px)

    # Point life-cycle
    promote_px: float = 4.0     # a candidate becomes an object point after moving this far...
    promote_speed: float = 0.8  # ...at an average speed of at least this (px/frame); rejects slow drift
    candidate_ttl: int = 8      # frames a candidate has to prove it moves
    still_speed: float = 0.5    # below this smoothed speed (px/frame) a point counts as 'still'
    speed_ema: float = 0.3      # smoothing factor of the per-point speed (jitter != motion)
    stale_frames: int = 10      # object points still for this long are dropped (stuck on occluder edge)
    stop_ratio: float = 0.25    # a point whose speed drops below this fraction of its previous
    stop_min_speed: float = 1.0 #   speed (if that was > stop_min_speed) in one frame hit the occluder
    outlier_k: float = 3.0      # MAD multiplier for velocity / position outliers
    vel_tol_px: float = 2.0     # always accept velocity deviations below this...
    vel_rel_tol: float = 0.6    # ...or below this fraction of the median speed (rolling objects:
                                #    top of a ball moves at 2v, bottom at ~0)
    cluster_radius: float = 0.08  # points closer than this * frame width are one cluster;
                                  # the cluster with the most motion is the object
    keep_bonus: float = 3.0       # score multiplier for the cluster holding last frame's object
                                  # points (identity persistence, no flipping between clusters)
    min_points: int = 4         # minimum object points for a valid measurement
    min_obj_speed: float = 1.0  # ...and the object's smoothed speed (px/frame) must reach this;
                                #    slower clusters are background flicker or a stopped object
    min_streak: int = 3         # raw measurements needed in a row (with a plausible jump) before
                                #    reporting one: kills one-off flicker detections
    max_jump: float = 0.10      # max jump between consecutive measurements: frame width * this + 2|v|

    # Visualisation
    trail_len: int = 15
    arrow_scale: float = 3.0
    arrow_every: int = 1        # draw velocity arrow every n frames (centroid)

    @classmethod
    def from_dict(cls, d: dict | None) -> "OFConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if not hasattr(cfg, k):
                raise KeyError(f'Unknown optical-flow parameter: {k}')
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


@dataclass
class Measurement:
    frame: int
    valid: bool
    x: float = np.nan
    y: float = np.nan
    vx: float = np.nan
    vy: float = np.nan
    sx: float = np.nan      # std of point positions
    sy: float = np.nan
    svx: float = np.nan     # std of point velocities
    svy: float = np.nan
    n_points: int = 0


@dataclass
class _Points:
    """Structure-of-arrays for the tracked keypoints."""
    p: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    origin: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    age: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    is_obj: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    vel: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    still: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    ema: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    sel: np.ndarray = field(default_factory=lambda: np.empty(0, bool))  # in the chosen object cluster

    def keep(self, m: np.ndarray) -> None:
        self.p, self.origin, self.age = self.p[m], self.origin[m], self.age[m]
        self.is_obj, self.vel, self.still = self.is_obj[m], self.vel[m], self.still[m]
        self.ema, self.sel = self.ema[m], self.sel[m]

    def add(self, new: np.ndarray) -> None:
        n = len(new)
        self.p = np.vstack([self.p, new]).astype(np.float32)
        self.origin = np.vstack([self.origin, new]).astype(np.float32)
        self.age = np.concatenate([self.age, np.zeros(n, np.int32)])
        self.is_obj = np.concatenate([self.is_obj, np.zeros(n, bool)])
        self.vel = np.vstack([self.vel, np.zeros((n, 2), np.float32)])
        self.still = np.concatenate([self.still, np.zeros(n, np.int32)])
        self.ema = np.concatenate([self.ema, np.zeros(n, np.float32)])
        self.sel = np.concatenate([self.sel, np.zeros(n, bool)])

    def __len__(self) -> int:
        return len(self.p)


def _mad_inliers(x: np.ndarray, k: float, tol: float, rel_tol: float = 0.0) -> np.ndarray:
    """Per-row inlier mask: distance to the median below max(k * MAD, tol, rel_tol * |median|)."""
    med = np.median(x, axis=0)
    tol = max(tol, rel_tol * float(np.linalg.norm(med)))
    d = np.linalg.norm(x - med, axis=1)
    mad = 1.4826 * np.median(d)
    return d <= max(k * mad, tol)


def _clusters(p: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of points linked when closer than `radius`. Returns labels."""
    n = len(p)
    adj = np.linalg.norm(p[:, None] - p[None], axis=2) < radius
    labels = np.full(n, -1)
    lab = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = lab
        while stack:
            j = stack.pop()
            for k in np.flatnonzero(adj[j] & (labels < 0)):
                labels[k] = lab
                stack.append(k)
        lab += 1
    return labels


class LKTracker:
    def __init__(self, cfg: OFConfig):
        self.cfg = cfg
        self.lk_params = dict(
            winSize=(cfg.win_size, cfg.win_size),
            maxLevel=cfg.max_level,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, cfg.lk_iters, cfg.lk_eps),
        )
        self.pts = _Points()
        self.history: list[np.ndarray] = []   # grey frames (only the last motion_gap+1)
        self.frame_idx = -1
        self._streak = 0
        self._last_raw: Measurement | None = None

    # ------------------------------------------------------------------ steps
    def _track(self, prev: np.ndarray, cur: np.ndarray) -> None:
        if len(self.pts) == 0:
            return
        p0 = self.pts.p.reshape(-1, 1, 2)
        p1, st1, err = cv.calcOpticalFlowPyrLK(prev, cur, p0, None, **self.lk_params)
        p0r, st2, _ = cv.calcOpticalFlowPyrLK(cur, prev, p1, None, **self.lk_params)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        p1 = p1.reshape(-1, 2)
        h, w = cur.shape
        ok = (
            (st1.ravel() == 1) & (st2.ravel() == 1)
            & (fb < self.cfg.fb_max_err)
            & (err.ravel() < self.cfg.lk_max_residual)
            & (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        )
        vel = p1 - self.pts.p
        speed, prev_speed = np.linalg.norm(vel, axis=1), np.linalg.norm(self.pts.vel, axis=1)
        # an object point that suddenly stops has latched onto the occluder edge
        stopped = (self.pts.is_obj & (prev_speed > self.cfg.stop_min_speed)
                   & (speed < self.cfg.stop_ratio * prev_speed))
        a = self.cfg.speed_ema
        self.pts.ema = np.where(self.pts.age == 0, speed, a * speed + (1 - a) * self.pts.ema).astype(np.float32)
        self.pts.still = np.where(self.pts.ema < self.cfg.still_speed, self.pts.still + 1, 0)
        self.pts.vel = vel.astype(np.float32)
        self.pts.p = p1.astype(np.float32)
        self.pts.age += 1
        self.pts.keep(ok & ~stopped)

    def _promote(self) -> None:
        c = self.cfg
        disp = np.linalg.norm(self.pts.p - self.pts.origin, axis=1)
        moved = (disp >= c.promote_px) & (disp >= c.promote_speed * np.maximum(self.pts.age, 1))
        self.pts.is_obj |= moved
        expired = (~self.pts.is_obj) & (self.pts.age > c.candidate_ttl)
        stale = self.pts.is_obj & (self.pts.still > c.stale_frames)
        self.pts.keep(~expired & ~stale)

    def _select_object(self, frame_w: int) -> None:
        c = self.cfg
        was_sel = self.pts.sel.copy()
        self.pts.sel[:] = False
        obj = np.flatnonzero(self.pts.is_obj)
        if len(obj) == 0:
            return
        labels = _clusters(self.pts.p[obj], c.cluster_radius * frame_w)
        scores = np.bincount(labels, weights=self.pts.ema[obj])
        held = np.bincount(labels, weights=was_sel[obj].astype(float), minlength=len(scores))
        if held.any():
            scores[np.argmax(held)] *= c.keep_bonus
        members = obj[labels == np.argmax(scores)]
        if len(members) >= 3:  # velocity outliers inside the object cluster are dropped for good
            good = _mad_inliers(self.pts.vel[members], c.outlier_k, c.vel_tol_px, c.vel_rel_tol)
            keep = np.ones(len(self.pts), bool)
            keep[members[~good]] = False
            self.pts.sel[members[good]] = True
            self.pts.keep(keep)
        else:
            self.pts.sel[members] = True

    def _motion_mask(self) -> np.ndarray | None:
        c = self.cfg
        if len(self.history) <= c.motion_gap:
            return None
        a = cv.GaussianBlur(self.history[-1], (5, 5), 0)
        b = cv.GaussianBlur(self.history[-1 - c.motion_gap], (5, 5), 0)
        _, m = cv.threshold(cv.absdiff(a, b), c.motion_thresh, 255, cv.THRESH_BINARY)
        m = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # low-texture objects only light up at their edges: close the blob to cover the interior
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, np.ones((c.motion_dilate * 2 + 1,) * 2, np.uint8))
        return cv.dilate(m, np.ones((c.motion_dilate, c.motion_dilate), np.uint8))

    def _seed(self, grey: np.ndarray) -> None:
        c = self.cfg
        if self.frame_idx % c.reseed_every or self.pts.is_obj.sum() >= c.target_points:
            return
        mask = self._motion_mask()
        if mask is None or not mask.any():
            return
        for x, y in self.pts.p:  # don't seed on top of existing points
            cv.circle(mask, (int(x), int(y)), c.min_distance, 0, -1)
        new = cv.goodFeaturesToTrack(
            grey, maxCorners=c.max_corners, qualityLevel=c.quality_level,
            minDistance=c.min_distance, blockSize=c.block_size, mask=mask,
        )
        if new is not None:
            self.pts.add(new.reshape(-1, 2))

    def _measure(self) -> Measurement:
        sel = self.pts.sel
        n = int(sel.sum())
        if n < self.cfg.min_points or np.median(self.pts.ema[sel]) < self.cfg.min_obj_speed:
            return Measurement(self.frame_idx, False, n_points=n)
        p, v = self.pts.p[sel], self.pts.vel[sel]
        (x, y), (vx, vy) = np.median(p, axis=0), np.median(v, axis=0)
        (sx, sy), (svx, svy) = p.std(axis=0), v.std(axis=0)
        return Measurement(self.frame_idx, True, float(x), float(y), float(vx), float(vy),
                           float(sx), float(sy), float(svx), float(svy), n)

    def _debounce(self, m: Measurement, frame_w: int) -> Measurement:
        """Only report a measurement once it has persisted for `min_streak` frames."""
        last, self._last_raw = self._last_raw, m
        if not m.valid:
            self._streak = 0
            return m
        if last is not None and last.valid:
            jump = np.hypot(m.x - last.x, m.y - last.y)
            ok = jump <= self.cfg.max_jump * frame_w + 2 * np.hypot(last.vx, last.vy)
            self._streak = self._streak + 1 if ok else 1
        else:
            self._streak = 1
        if self._streak < self.cfg.min_streak:
            return Measurement(m.frame, False, n_points=m.n_points)
        return m

    # ------------------------------------------------------------------ public
    def step(self, frame_bgr: np.ndarray) -> Measurement:
        self.frame_idx += 1
        grey = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if self.history:
            self._track(self.history[-1], grey)
        self.history = (self.history + [grey])[-(self.cfg.motion_gap + 1):]
        self._promote()
        self._select_object(grey.shape[1])
        meas = self._debounce(self._measure(), grey.shape[1])
        self._seed(grey)
        return meas

    @property
    def object_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Points in the selected object cluster (used for the measurement)."""
        return self.pts.p[self.pts.sel], self.pts.vel[self.pts.sel]

    @property
    def other_moving_points(self) -> np.ndarray:
        """Moving points that are not part of the selected object (hand, noise...)."""
        return self.pts.p[self.pts.is_obj & ~self.pts.sel]

    @property
    def candidate_points(self) -> np.ndarray:
        return self.pts.p[~self.pts.is_obj]


# ---------------------------------------------------------------------- driver
_GREEN, _GREY, _YELLOW, _WHITE, _RED = (60, 220, 60), (160, 160, 160), (0, 220, 255), (255, 255, 255), (40, 40, 230)
_ORANGE = (0, 140, 255)


def _draw(frame, tracker: LKTracker, meas: Measurement, trails: list, cfg: OFConfig):
    out = frame.copy()
    # fading trail of past measured positions
    for i in range(1, len(trails)):
        if trails[i - 1] is None or trails[i] is None:
            continue
        a = i / len(trails)
        cv.line(out, trails[i - 1], trails[i], tuple(int(c * a) for c in _YELLOW), 2, cv.LINE_AA)
    for x, y in tracker.candidate_points:
        cv.circle(out, (int(x), int(y)), 2, _GREY, -1, cv.LINE_AA)
    for x, y in tracker.other_moving_points:
        cv.circle(out, (int(x), int(y)), 3, _ORANGE, -1, cv.LINE_AA)
    pts, vel = tracker.object_points
    for (x, y), (vx, vy) in zip(pts, vel):
        p = (int(x), int(y))
        q = (int(x + vx * cfg.arrow_scale), int(y + vy * cfg.arrow_scale))
        cv.arrowedLine(out, p, q, _GREEN, 1, cv.LINE_AA, tipLength=0.3)
        cv.circle(out, p, 3, _GREEN, -1, cv.LINE_AA)
    if meas.valid:
        c = (int(meas.x), int(meas.y))
        cv.circle(out, c, 7, _RED, 2, cv.LINE_AA)
        if meas.frame % cfg.arrow_every == 0:
            tip = (int(meas.x + meas.vx * cfg.arrow_scale * 3), int(meas.y + meas.vy * cfg.arrow_scale * 3))
            cv.arrowedLine(out, c, tip, _RED, 3, cv.LINE_AA, tipLength=0.25)
        txt = f'#{meas.frame}  n={meas.n_points}  v=({meas.vx:+.1f}, {meas.vy:+.1f}) px/frame'
    else:
        txt = f'#{meas.frame}  NO MEASUREMENT  (n={meas.n_points})'
    s = out.shape[1] / 1280
    cv.rectangle(out, (0, 0), (int(620 * s), int(36 * s)), (0, 0, 0), -1)
    cv.putText(out, txt, (int(8 * s), int(25 * s)), cv.FONT_HERSHEY_SIMPLEX, 0.65 * s,
               _WHITE if meas.valid else (80, 80, 255), max(1, int(2 * s)), cv.LINE_AA)
    return out


def run_optical_flow(video_in: str, video_out: str | None, cfg: OFConfig,
                     csv_out: str | None = None) -> tuple[list[Measurement], float]:
    """Run the tracker over a whole video. Returns (measurements, fps)."""
    cap = cv.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(video_out, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if video_out else None

    tracker, meas, trails = LKTracker(cfg), [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = tracker.step(frame)
        meas.append(m)
        trails = (trails + [(int(m.x), int(m.y)) if m.valid else None])[-cfg.trail_len:]
        if writer:
            writer.write(_draw(frame, tracker, m, trails, cfg))
    cap.release()
    if writer:
        writer.release()

    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(asdict(meas[0]).keys()))
            wr.writeheader()
            for m in meas:
                wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(m).items()})
    return meas, fps
