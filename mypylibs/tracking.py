"""Step 4/5: track the object with the Kalman filter and plot the whole process.

What is tracked
---------------
The object is represented by the *median of its optical-flow keypoints* (a
robust centroid): it does not jump when a few points are lost or wrongly
added, which a mean or a single point of interest would.

What is drawn on ``XX_tr.mp4``
------------------------------
* red dots          raw optical-flow measurements (grey crosses: rejected by the gate)
* green line        filter estimate while it is being corrected by measurements
* yellow dashes     filter estimate while it is only predicting (object occluded)
* cyan line         trajectory 1: the corrected estimate after the object is back
* magenta dashes    trajectory 2: what the filter would have kept predicting had the
                    object *not* come back (pure prediction from the state just before
                    the first measurement after the occlusion, P grown with F P F^T + Q)
* ellipses          2-sigma position uncertainty from P (grows while predicting)
* white arrows      filter velocity every ``arrow_every`` frames
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from mypylibs.kalman import KFConfig, KFStep, process_noise, transition_matrix
from mypylibs.optical_flow import Measurement

# BGR
RED = (40, 40, 230)
GREEN = (60, 200, 60)
YELLOW = (0, 215, 255)
CYAN = (255, 200, 0)
MAGENTA = (255, 60, 255)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# segment modes
_MEAS, _PRED, _BACK = 0, 1, 2
_MODE_COLOUR = {_MEAS: GREEN, _PRED: YELLOW, _BACK: CYAN}


@dataclass
class _Branch:
    """Counterfactual trajectory: the filter keeps predicting, no more evidence."""
    start: int
    pts: list            # one (x, y) per frame from `start`
    covs: list           # 2x2 position covariance per frame
    end: int | None = None   # frame where a newer branch supersedes it


def _pt(xy) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def _line(img, a, b, colour, th, outline=True):
    if outline:
        cv.line(img, a, b, BLACK, th + 2, cv.LINE_AA)
    cv.line(img, a, b, colour, th, cv.LINE_AA)


def _ellipse(img, centre, cov, colour, th, n_sigma=2.0):
    vals, vecs = np.linalg.eigh(np.asarray(cov, float))
    vals = np.clip(vals, 0, None)
    axes = (int(n_sigma * np.sqrt(vals[1])), int(n_sigma * np.sqrt(vals[0])))
    if max(axes) < 2 or max(axes) > 5000:
        return
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    cv.ellipse(img, _pt(centre), axes, ang, 0, 360, colour, th, cv.LINE_AA)


def _counterfactual(s0: np.ndarray, P0: np.ndarray, n: int, Q: np.ndarray):
    """Propagate state s0 / covariance P0 with the motion model only, for n frames."""
    F = transition_matrix().astype(float)
    s, P, pts, covs = s0.astype(float), P0.astype(float), [], []
    for _ in range(n):
        pts.append(s[:2].copy())
        covs.append(P[:2, :2].copy())
        s, P = F @ s, F @ P @ F.T + Q
    return pts, covs


def _legend_size(sc):
    return int(255 * sc), int(22 * sc) * 6 + 2 * int(8 * sc)


def _pick_corner(W, H, sc, pts) -> tuple[int, int]:
    """Top-left of the legend box in the bottom/top corner the trajectory avoids most."""
    w, h = _legend_size(sc)
    pad, top = int(8 * sc), int(50 * sc)   # leave room for the banner at the top-left
    corners = [(pad, H - h - pad), (W - w - pad, H - h - pad), (W - w - pad, top)]
    pts = np.array([p for p in pts if p is not None]).reshape(-1, 2)

    def hits(c):
        x0, y0 = c
        return int(((pts[:, 0] > x0 - 20) & (pts[:, 0] < x0 + w + 20) &
                    (pts[:, 1] > y0 - 20) & (pts[:, 1] < y0 + h + 20)).sum())
    return min(corners, key=hits)


def _legend(img, sc, corner):
    items = [
        ('OF measurement', RED, 'dot'),
        ('KF corrected', GREEN, 'line'),
        ('KF predicting (occluded)', YELLOW, 'dash'),
        ('1) object back (corrected)', CYAN, 'line'),
        ('2) if it had not come back', MAGENTA, 'dash'),
        ('2-sigma uncertainty', WHITE, 'ell'),
    ]
    fs, lh, pad = 0.5 * sc, int(22 * sc), int(8 * sc)
    w, h = _legend_size(sc)
    x0, y0 = corner
    roi = img[y0:y0 + h, x0:x0 + w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    th = max(1, int(2 * sc))
    for i, (txt, col, kind) in enumerate(items):
        y = y0 + pad + i * lh + lh // 2
        a, b = (x0 + pad, y), (x0 + pad + int(30 * sc), y)
        if kind == 'dot':
            cv.circle(img, ((a[0] + b[0]) // 2, y), max(2, int(3 * sc)), col, -1, cv.LINE_AA)
        elif kind == 'line':
            cv.line(img, a, b, col, th + 1, cv.LINE_AA)
        elif kind == 'dash':
            m = (a[0] + b[0]) // 2
            cv.line(img, a, (m - int(3 * sc), y), col, th + 1, cv.LINE_AA)
            cv.line(img, (m + int(3 * sc), y), b, col, th + 1, cv.LINE_AA)
        else:
            cv.ellipse(img, ((a[0] + b[0]) // 2, y), (int(14 * sc), int(7 * sc)), 0, 0, 360, col, 1, cv.LINE_AA)
        cv.putText(img, txt, (b[0] + pad, y + int(5 * sc)), cv.FONT_HERSHEY_SIMPLEX, fs, WHITE, 1, cv.LINE_AA)


def _banner(img, sc, text, colour):
    fs, th = 0.7 * sc, max(1, int(2 * sc))
    (tw, tht), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(8 * sc)
    cv.rectangle(img, (0, 0), (tw + 2 * pad, tht + 2 * pad + 4), BLACK, -1)
    cv.putText(img, text, (pad, tht + pad), cv.FONT_HERSHEY_SIMPLEX, fs, colour, th, cv.LINE_AA)


def run_tracking_video(video_in: str, video_out: str, measurements: list[Measurement],
                       steps: list[KFStep], kf_cfg: KFConfig | None = None,
                       min_occlusion: int = 10, arrow_every: int = 10,
                       arrow_scale: float = 4.0, snapshot_out: str | None = None) -> list[dict]:
    """Render the tracking video. Returns one dict per reappearance with the error
    between trajectory 1 and trajectory 2 over time (for the report)."""
    cap = cv.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(video_in)
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(video_out, cv.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    sc = W / 1280
    th = max(2, int(round(3 * sc)))
    Q = process_noise((kf_cfg or KFConfig()).sigma_a).astype(float)
    F = transition_matrix().astype(float)
    n_frames = len(steps)

    # ---- pre-compute everything that does not depend on the image ------------------
    est, mode, breaks = [None] * n_frames, [None] * n_frames, set()
    branches: list[_Branch] = []
    gap, after_back, prev = 0, False, None
    for i, s in enumerate(steps):
        if not s.started:
            prev = None
            continue
        if s.corrected and s.K is None and i > 0 and prev is not None:
            breaks.add(i)          # filter (re)started: don't connect to the old track
            after_back = False
        if s.corrected and gap >= min_occlusion and prev is not None:
            # prior covariance at this frame = F P_{k-1} F^T + Q
            P_pre = F @ steps[i - 1].P.astype(float) @ F.T + Q
            if branches:
                branches[-1].end = i
            pts, covs = _counterfactual(s.pred, P_pre, n_frames - i, Q)
            branches.append(_Branch(i, pts, covs))
            after_back = True
        if s.corrected:
            gap = 0
        else:
            gap += 1
            if gap >= min_occlusion:
                after_back = False
        est[i] = s.post[:2]
        mode[i] = _MEAS if s.corrected else _PRED
        if s.corrected and after_back:
            mode[i] = _BACK
        prev = s

    legend_at = _pick_corner(W, H, sc, est + [(m.x, m.y) for m in measurements if m.valid]
                             + [p for b in branches for p in b.pts[:(b.end or n_frames) - b.start]])

    errors = []
    for b in branches:
        stop = b.end if b.end is not None else n_frames
        d = [float(np.hypot(*(b.pts[k] - est[b.start + k]))) for k in range(stop - b.start)
             if est[b.start + k] is not None]
        errors.append(dict(frame=b.start, divergence=d))

    # ---- render ----------------------------------------------------------------------
    frame_idx, snap_frame = 0, (branches[-1].start + min(60, n_frames - 1 - branches[-1].start)
                                if branches else n_frames - 1)
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= n_frames:
            break
        i, s, m = frame_idx, steps[frame_idx], measurements[frame_idx]
        out = frame

        # measurements (dots) and rejected ones (crosses)
        for k in range(i + 1):
            mk = measurements[k]
            if not mk.valid:
                continue
            p = _pt((mk.x, mk.y))
            if steps[k].rejected:
                cv.drawMarker(out, p, GREY, cv.MARKER_TILTED_CROSS, int(10 * sc), max(1, int(2 * sc)))
            else:
                cv.circle(out, p, max(2, int(3 * sc)), RED, -1, cv.LINE_AA)

        # counterfactual branches up to now (behind the estimate)
        for b in branches:
            if b.start > i:
                continue
            last = min(i, (b.end if b.end is not None else n_frames) - 1) - b.start
            for k in range(1, last + 1):
                if (k // 3) % 2 == 0:
                    _line(out, _pt(b.pts[k - 1]), _pt(b.pts[k]), MAGENTA, th)
            # head of the counterfactual: where the filter would think the object is now
            if b.end is None or i < b.end:
                head = b.pts[last]
                if 0 <= head[0] < W and 0 <= head[1] < H:
                    _ellipse(out, head, b.covs[last], MAGENTA, max(1, int(sc)))
                    cv.circle(out, _pt(head), int(6 * sc), MAGENTA, -1, cv.LINE_AA)
                    if est[i] is not None:
                        d = float(np.hypot(*(head - est[i])))
                        cv.line(out, _pt(head), _pt(est[i]), WHITE, 1, cv.LINE_AA)
                        mid = _pt((head + est[i]) / 2)
                        cv.putText(out, f'{d:.0f}px', (mid[0] + 6, mid[1] - 6), cv.FONT_HERSHEY_SIMPLEX,
                                   0.55 * sc, WHITE, max(1, int(sc)), cv.LINE_AA)

        # filter trajectory, coloured by mode of the segment's end frame
        for k in range(1, i + 1):
            if est[k] is None or est[k - 1] is None or k in breaks:
                continue
            if mode[k] == _PRED and (k // 3) % 2:
                continue   # dashed while predicting
            _line(out, _pt(est[k - 1]), _pt(est[k]), _MODE_COLOUR[mode[k]], th)

        # velocity arrows every n frames
        for k in range(0, i + 1, arrow_every):
            if est[k] is None:
                continue
            v = steps[k].post[2:4]
            a, b = _pt(est[k]), _pt(est[k] + arrow_scale * v)
            cv.arrowedLine(out, a, b, BLACK, max(1, int(sc)) + 2, cv.LINE_AA, tipLength=0.2)
            cv.arrowedLine(out, a, b, WHITE, max(1, int(sc)), cv.LINE_AA, tipLength=0.2)

        # current estimate + uncertainty
        if est[i] is not None:
            col = _MODE_COLOUR[mode[i]]
            _ellipse(out, est[i], s.P[:2, :2], col, max(1, int(sc)) + 1)
            cv.circle(out, _pt(est[i]), int(8 * sc), BLACK, -1, cv.LINE_AA)
            cv.circle(out, _pt(est[i]), int(6 * sc), col, -1, cv.LINE_AA)

        # status banner
        if not s.started:
            txt, col = f'#{i}  waiting for the object', GREY
        elif s.corrected:
            txt, col = f'#{i}  MEASURING   K_x={s.K[0, 0]:.2f}' if s.K is not None else f'#{i}  (re)started', \
                _MODE_COLOUR[mode[i]]
        else:
            n_pred = 0
            for k in range(i, -1, -1):
                if steps[k].corrected:
                    break
                n_pred += 1
            why = 'gated' if s.rejected else 'occluded'
            sd = np.sqrt(max(s.P[0, 0], 0))
            txt, col = f'#{i}  PREDICTING ({why}, {n_pred} fr)   sd_x={sd:.0f}px', YELLOW
        _banner(out, sc, txt, col)
        _legend(out, sc, legend_at)

        writer.write(out)
        if snapshot_out and i == snap_frame:
            cv.imwrite(snapshot_out, out)
        frame_idx += 1

    cap.release()
    writer.release()
    return errors
