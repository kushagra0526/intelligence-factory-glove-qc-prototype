"""QC for one glove episode: tactile↔camera sync, tactile health, IMU anatomy, retargeting feasibility → verdict.

The sync core (windowed cross-correlation + Theil-Sen offset/drift fit, re-align, re-verify) is adapted from
github.com/kushagra0526/human-archive-episode-qa-prototype.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synth import FINGERS, JOINTS, SAT_N

SYNC_SPEC_S = 0.033                     # one wrist-camera frame at 30 fps
WINDOW_S, GRID_HZ, MAX_LAG_S, LAG_STEP_S = 10.0, 200, 0.6, 0.005
ZERO_N, DEAD_RESPONSE_N, SAT_FRAC, OUT_OF_RANGE_FRAC = 0.5, 0.2, 0.02, 0.01
LIMITS = {"mcp_flex": (-30, 95), "mcp_abd": (-25, 30), "pip": (0, 115), "dip": (-5, 90), "wrist_flex": (-80, 80)}
ROBOTS = {"parallel_gripper": (85, 40), "three_finger": (110, 30), "five_finger": (120, 60)}  # max aperture mm, max grip N
REJECTING = {"SYNC_UNVERIFIED", "TAXEL_DEAD", "TAXEL_SATURATED", "JOINT_OUT_OF_RANGE"}
REPAIRED = {"SYNC_REALIGNED", "ZERO_DRIFT", "SHEAR_REVERSED"}


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def window_offsets(ref, other, duration):
    rts, re = ref
    ots, oe = other
    lags = np.arange(-MAX_LAG_S, MAX_LAG_S + 1e-9, LAG_STEP_S)
    out = []
    for c in np.arange(WINDOW_S / 2, duration - WINDOW_S / 2 + 1e-9, WINDOW_S):
        grid = np.arange(c - WINDOW_S / 2, c + WINDOW_S / 2, 1 / GRID_HZ)
        r = np.interp(grid, rts, re)
        if r.std() < 0.25 * re.std():
            continue
        cs = np.array([_corr(r, np.interp(grid + L, ots, oe)) for L in lags])
        i = int(cs.argmax())
        if cs[i] < 0.5:
            continue
        lag = lags[i]
        if 0 < i < len(cs) - 1:
            den = cs[i - 1] - 2 * cs[i] + cs[i + 1]
            if den != 0:
                lag += 0.5 * (cs[i - 1] - cs[i + 1]) / den * LAG_STEP_S
        out.append((float(c), float(lag), float(cs[i])))
    return out


def robust_fit(c, o, duration):
    pairs = [(o[j] - o[i]) / (c[j] - c[i]) for i in range(len(c)) for j in range(i + 1, len(c))]
    slope = float(np.median(pairs))
    noise = float(1.4826 * np.median(np.abs(o - (np.median(o - slope * c) + slope * c))))
    if abs(slope) * duration < max(SYNC_SPEC_S, 4 * noise):
        slope = 0.0
    icpt = float(np.median(o - slope * c))
    return icpt, slope, max(abs(icpt), abs(icpt + slope * duration))


@dataclass
class Report:
    ep_id: str
    meta: dict
    verdict: str = "ACCEPT"
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    sync: dict = field(default_factory=dict)
    baselines: list[float] = field(default_factory=list)   # per-taxel non-contact zero (N), before re-zero
    robots: list[str] = field(default_factory=list)
    aperture_mm: float = 0.0
    grip_n: float = 0.0

    def add(self, code, reason):
        if code not in self.codes:
            self.codes.append(code)
        self.reasons.append(reason)


def _load(d: Path, name):
    z = np.load(d / f"{name}.npz")
    return {k: z[k] for k in z.files}


def run_episode(d: Path) -> Report:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    dur = float(meta["duration_s"])
    rep = Report(meta["ep_id"], meta)
    tac, jo, cam = _load(d, "tactile"), _load(d, "joints"), _load(d, "wristcam")
    tts, T = tac["ts"].astype(float), tac["data"].astype(float).copy()       # (n, 5, 3)

    # 1. sync tactile contact force to camera proximity
    prox = (cam["ts"], np.exp(-cam["data"] / 15.0))
    total = T[:, :, 0].sum(axis=1)
    w = window_offsets(prox, (tts, total), dur)
    if len(w) < 3:
        rep.add("SYNC_UNVERIFIED", f"tactile↔camera sync could not be verified ({len(w)} usable windows)")
        rep.verdict = "REJECT"
        return rep
    c, o = np.array([x[0] for x in w]), np.array([x[1] for x in w])
    icpt, slope, worst = robust_fit(c, o, dur)
    rep.sync = {"before_ms": worst * 1e3, "offset_ms": icpt * 1e3, "drift_ms_per_min": slope * 60e3, "after_ms": worst * 1e3}
    if worst >= SYNC_SPEC_S:
        fixed = (tts - icpt) / (1 + slope)
        w2 = window_offsets(prox, (fixed, total), dur)
        after = robust_fit(np.array([x[0] for x in w2]), np.array([x[1] for x in w2]), dur)[2] if len(w2) >= 3 else float("inf")
        rep.sync["after_ms"] = after * 1e3
        if after < SYNC_SPEC_S:
            tts = fixed
            what = f"drift {slope * 60e3:+.0f} ms/min" if slope else f"offset {icpt * 1e3:+.0f} ms"
            rep.add("SYNC_REALIGNED", f"tactile clock {what} (worst {worst * 1e3:.0f} ms) → re-aligned, now {after * 1e3:.0f} ms < 33 ms")
        else:
            rep.add("SYNC_UNVERIFIED", f"tactile clock off by up to {worst * 1e3:.0f} ms and re-alignment did not verify")

    # contact from the camera (independent of any single taxel), on the tactile clock
    dist = np.interp(tts, cam["ts"], cam["data"])
    in_contact, free = dist < 4, dist > 30

    # 2. tactile health
    N = T[:, :, 0]
    rep.baselines = [float(np.median(N[free, i])) for i in range(5)]
    for i, f in enumerate(FINGERS):
        if rep.baselines[i] > ZERO_N:
            N[:, i] -= rep.baselines[i]
            rep.add("ZERO_DRIFT", f"{f} taxel reads {rep.baselines[i]:.2f} N with nothing touching → re-zeroed")
    others_force = np.median(N[in_contact].sum(axis=1))
    dead = []
    for i, f in enumerate(FINGERS):
        response = float(np.median(N[in_contact, i]) - np.median(N[free, i]))   # does it feel the grasp at all?
        if response < DEAD_RESPONSE_N and others_force > 3:
            dead.append(i)
            rep.add("TAXEL_DEAD", f"{f} taxel flat (responds {response:+.2f} N to contact) while the hand grips {others_force:.0f} N")
        sat = float(np.mean(T[in_contact, i, 0] >= SAT_N - 0.05))
        if sat > SAT_FRAC:
            rep.add("TAXEL_SATURATED", f"{f} taxel at its {SAT_N:.0f} N ceiling for {sat:.0%} of contact → peak force lost")
    S = T[:, :, 2]
    live = [i for i in range(5) if i not in dead]
    for i in live:
        ref = np.median(S[in_contact][:, [k for k in live if k != i]], axis=1)
        if _corr(S[in_contact, i], ref) < -0.3:
            S[:, i] *= -1
            rep.add("SHEAR_REVERSED", f"{FINGERS[i]} lift-axis shear runs opposite to the other fingers → sign flipped")

    # 3. IMU joint anatomy
    A = jo["data"].astype(float)
    for k, name in enumerate(JOINTS):
        lo, hi = LIMITS[next(key for key in LIMITS if name.endswith(key))]
        frac = float(np.mean((A[:, k] < lo) | (A[:, k] > hi)))
        if frac > OUT_OF_RANGE_FRAC:
            rep.add("JOINT_OUT_OF_RANGE", f"{name} outside human range [{lo}°, {hi}°] for {frac:.0%} of samples (max {A[:, k].max():.0f}°) → glove calibration slip")

    # 4. retargeting feasibility per robot body
    jd = np.interp(jo["ts"], cam["ts"], cam["data"])
    rep.aperture_mm = float(np.percentile(jo["aperture"][jd < 4], 95))
    rep.grip_n = float(np.percentile(N[in_contact].sum(axis=1), 95))
    rep.robots = [r for r, (ap, fmax) in ROBOTS.items() if rep.aperture_mm <= ap and rep.grip_n <= fmax]

    if any(c in REJECTING for c in rep.codes):
        rep.verdict = "REJECT"
    elif any(c in REPAIRED for c in rep.codes):
        rep.verdict = "FIXED"
    return rep


def run_all(root: Path) -> list[Report]:
    return [run_episode(p) for p in sorted(root.iterdir()) if (p / "meta.json").exists()]


def glove_rollup(reports: list[Report]) -> list[str]:
    """Per-glove trends across the shift: what to do before the next episode, not after the day."""
    out = []
    by: dict[str, list[Report]] = {}
    for r in reports:
        by.setdefault(r.meta["glove"], []).append(r)
    for g, rs in sorted(by.items()):
        rs.sort(key=lambda r: r.meta["shift_index"])
        for i, f in enumerate(FINGERS):
            z = [r.baselines[i] for r in rs if r.baselines]
            if len(z) >= 4 and z[-1] > ZERO_N:
                slope = float(np.polyfit(range(len(z)), z, 1)[0])
                if slope > 0.05:
                    first = next(r.ep_id for r in rs if r.baselines and r.baselines[i] > ZERO_N)
                    out.append(f"{g}: {f} taxel zero creeping {z[0]:.2f} → {z[-1]:.2f} N (+{slope:.2f} N/episode), over limit since {first} → recalibrate before next episode")
            dead = [r.ep_id for r in rs if any(x.startswith(f"{f} taxel flat") for x in r.reasons)]
            if dead:
                out.append(f"{g}: {f} taxel dead in {len(dead)} episodes from {dead[0]} on → swap glove ({len(dead) * 2} min of tactile data lost)")
        for code, msg in (("JOINT_OUT_OF_RANGE", "IMU calibration slip"), ("SHEAR_REVERSED", "shear axis reversed"), ("SYNC_REALIGNED", "tactile clock offset/drift")):
            hit = [r.ep_id for r in rs if code in r.codes]
            if hit:
                out.append(f"{g}: {msg} in {', '.join(hit)}")
    return out
