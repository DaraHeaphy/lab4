"""Step 3: Kalman filter on the optical-flow measurements.

Model (units: pixels and frames, dt = 1 frame)
---------------------------------------------
State          s = [x, y, vx, vy]^T                  (constant-velocity model)
Transition     F = [[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1,  0],
                    [0, 0, 0,  1]]
Control        none (G is dropped: nothing pushes the object that we know of)
Measurement    z = [x, y, vx, vy]^T, H = I4          (optical flow measures both)
               or z = [x, y]^T, H = [I2 0]           (``measure_velocity: false``)
Process noise  Q = sigma_a^2 * [[dt^4/4, dt^3/2],    per axis: an unknown random
                                [dt^3/2, dt^2  ]]    acceleration with std sigma_a
Measurement    R_k = diag(sigma^2) recomputed every frame from the optical flow:
noise          the reported position/velocity is the *median* of n keypoints with
               spread s, whose standard error is ~1.2533 * s / sqrt(n). A floor
               keeps R from collapsing when all points agree.
Initial P      P_0 = R_0 of the first measurement for the measured components; an
               unmeasured velocity gets a large variance (we know nothing yet).
Gating         A measurement whose velocity innovation y is implausible given its
               covariance S = H P H^T + R (Mahalanobis d^2 = y^T S^-1 y above the
               chi^2 99.9% quantile, 2 dof; position innovation when velocity is not
               measured) is rejected: this catches the last frames before an
               occlusion, when only a sliver of the object is left and optical flow
               reports garbage. After ``max_rejects`` rejections in a row the filter
               is restarted on the measurement (it has lost the object).

The filter is started at the first valid measurement. Every frame it predicts;
when a measurement is available it is also corrected. When the object is hidden
it only predicts, so P grows and K will be large again when it reappears.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from mypylibs.optical_flow import Measurement

_MEDIAN_SE = 1.2533  # std error of the median = 1.2533 * sigma / sqrt(n) (Gaussian)
_CHI2_999 = {2: 13.816}  # chi^2 99.9% quantile, 2 degrees of freedom


@dataclass
class KFConfig:
    measure_velocity: bool = True   # correct velocity too (H = I4) or position only (H = [I2 0])
    sigma_a: float = 0.5            # process noise: std of unmodelled acceleration (px / frame^2)
                                    #   0.1-0.3: real pushes get gated out; >=1: chases noisy velocity
    pos_noise_floor: float = 1.0    # min std of a position measurement (px)
    vel_noise_floor: float = 0.5    # min std of a velocity measurement (px / frame); the median-SE
                                    #   underestimates the observed velocity noise 2-3x
    gate: bool = True               # reject measurements failing the chi^2 innovation test
    max_rejects: int = 3            # rejections in consecutive frames before restarting the filter
    init_vel_std: float = 20.0      # initial velocity std when velocity is not measured

    @classmethod
    def from_dict(cls, d: dict | None) -> "KFConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if not hasattr(cfg, k):
                raise KeyError(f'Unknown Kalman parameter: {k}')
            setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


@dataclass
class KFStep:
    frame: int
    started: bool
    corrected: bool                 # a measurement was used this frame
    rejected: bool                  # a measurement existed but failed the gate
    pred: np.ndarray                # a-priori state  s_{k|k-1}
    post: np.ndarray                # a-posteriori state s_{k|k} (== pred if not corrected)
    P: np.ndarray                   # a-posteriori covariance
    K: np.ndarray | None            # Kalman gain (None if not corrected)
    innovation: np.ndarray | None   # z - H s_{k|k-1}


def transition_matrix(dt: float = 1.0) -> np.ndarray:
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], np.float32)


def process_noise(sigma_a: float, dt: float = 1.0) -> np.ndarray:
    """Discrete white-noise-acceleration model, x and y independent."""
    q = np.array([[dt**4 / 4, dt**3 / 2],
                  [dt**3 / 2, dt**2]]) * sigma_a**2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q   # x, vx
    Q[np.ix_([1, 3], [1, 3])] = q   # y, vy
    return Q.astype(np.float32)


class ObjectKF:
    def __init__(self, cfg: KFConfig):
        self.cfg = cfg
        nz = 4 if cfg.measure_velocity else 2
        self.kf = cv.KalmanFilter(4, nz, 0, cv.CV_32F)
        self.kf.transitionMatrix = transition_matrix()
        self.kf.measurementMatrix = np.eye(nz, 4, dtype=np.float32)
        self.kf.processNoiseCov = process_noise(cfg.sigma_a)
        self.started = False
        self.n_rejects = 0

    # -------------------------------------------------------------- measurement
    def _z_and_R(self, m: Measurement) -> tuple[np.ndarray, np.ndarray]:
        c, se = self.cfg, _MEDIAN_SE / np.sqrt(max(m.n_points, 1))
        pos_std = np.maximum([m.sx * se, m.sy * se], c.pos_noise_floor)
        vel_std = np.maximum([m.svx * se, m.svy * se], c.vel_noise_floor)
        if c.measure_velocity:
            z, std = [m.x, m.y, m.vx, m.vy], np.concatenate([pos_std, vel_std])
        else:
            z, std = [m.x, m.y], pos_std
        return np.array(z, np.float32).reshape(-1, 1), np.diag(std**2).astype(np.float32)

    def _start(self, m: Measurement) -> None:
        z, R = self._z_and_R(m)
        P0 = np.zeros((4, 4), np.float32)
        P0[:len(z), :len(z)] = R
        s0 = np.zeros((4, 1), np.float32)
        s0[:len(z)] = z
        if not self.cfg.measure_velocity:
            P0[2, 2] = P0[3, 3] = self.cfg.init_vel_std**2
        self.kf.statePost, self.kf.errorCovPost = s0, P0
        self.started = True

    # -------------------------------------------------------------------- step
    def step(self, m: Measurement) -> KFStep:
        if not self.started:
            if not m.valid:
                nan = np.full(4, np.nan)
                return KFStep(m.frame, False, False, False, nan, nan, np.full((4, 4), np.nan), None, None)
            return self._restart(m)

        pred = self.kf.predict().ravel().copy()
        rejected, innovation = False, None
        if not m.valid:
            self.n_rejects = 0   # only rejections in consecutive frames count towards a restart
        else:
            z, R = self._z_and_R(m)
            H = self.kf.measurementMatrix
            innovation = (z - H @ self.kf.statePre).ravel()
            S = H @ self.kf.errorCovPre @ H.T + R
            # gate on the velocity block when velocity is measured: position wobble on large
            # objects is heavy-tailed (points come and go), velocity garbage at occlusion edges is not
            g = slice(2, 4) if self.cfg.measure_velocity else slice(0, 2)
            y, Sg = innovation[g], S[g, g]
            d2 = float(y @ np.linalg.solve(Sg, y))
            rejected = self.cfg.gate and d2 > _CHI2_999[2]
            if not rejected:
                self.n_rejects = 0
                self.kf.measurementNoiseCov = R
                post = self.kf.correct(z).ravel().copy()
                return KFStep(m.frame, True, True, False, pred, post, self.kf.errorCovPost.copy(),
                              self.kf.gain.copy(), innovation)
            self.n_rejects += 1
            if self.n_rejects >= self.cfg.max_rejects:
                return self._restart(m)

        # no (usable) evidence: the prediction is the estimate (OpenCV leaves statePost alone)
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        return KFStep(m.frame, True, False, rejected, pred, pred, self.kf.errorCovPost.copy(), None, innovation)

    def _restart(self, m: Measurement) -> KFStep:
        self._start(m)
        self.n_rejects = 0
        s = self.kf.statePost.ravel().copy()
        return KFStep(m.frame, True, True, False, s, s, self.kf.errorCovPost.copy(), None, np.zeros(len(s)))


def run_kalman(meas: list[Measurement], cfg: KFConfig, csv_out: str | None = None) -> list[KFStep]:
    kf = ObjectKF(cfg)
    steps = [kf.step(m) for m in meas]
    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'started', 'corrected', 'rejected',
                        'pred_x', 'pred_y', 'pred_vx', 'pred_vy',
                        'x', 'y', 'vx', 'vy', 'P_x', 'P_y', 'P_vx', 'P_vy',
                        'K_x', 'K_y', 'K_vx', 'K_vy'])
            for s in steps:
                k = np.diag(s.K)[:4] if s.K is not None else []
                k = list(k) + [np.nan] * (4 - len(k))   # position-only: K is 4x2
                w.writerow([s.frame, int(s.started), int(s.corrected), int(s.rejected),
                            *np.round(s.pred, 3), *np.round(s.post, 3),
                            *np.round(np.diag(s.P), 3), *np.round(k, 4)])
    return steps


def reappearance_errors(steps: list[KFStep], min_gap: int = 5) -> list[dict]:
    """For every gap of >= min_gap frames without measurements, how far off was the
    pure prediction when the object was measured again?"""
    out, gap = [], 0
    for s in steps:
        if not s.started:
            continue
        if not s.corrected:
            gap += 1
            if not (gap >= min_gap and s.rejected):   # a rejected reappearance still counts
                continue
        if gap >= min_gap and s.innovation is not None:
            out.append(dict(frame=s.frame, gap=gap,
                            pos_err=float(np.hypot(*s.innovation[:2])),
                            pred=s.pred[:2].tolist(), meas=(s.pred[:2] + s.innovation[:2]).tolist()))
        gap = 0
    return out
