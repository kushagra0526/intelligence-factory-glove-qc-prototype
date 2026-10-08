"""Synthetic data-factory shift: 3 gloves × 8 pick/place episodes, with planted faults (labels.json).

Per episode, on separate clocks:
  tactile.npz  200 Hz  5 fingertip taxels × (normal, shear_x, shear_z) in N
  joints.npz   100 Hz  21 hand angles in degrees + edge-computed thumb–index aperture (mm)
  wristcam.npz  30 Hz  hand–object distance (mm), a stand-in for the depth-based contact cue
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUR = 120.0
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
SHARE = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
SAT_N = 15.0
JOINTS = [f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")] + ["wrist_flex"]


def _smooth_step(t, a, b, ramp=0.08):
    return np.clip((t - a) / ramp, 0, 1) * np.clip((b - t) / ramp, 0, 1)


def episode(rng, obj_width_mm, grip_n, faults: dict, ep_index: int):
    # ground-truth grasp schedule on the true clock
    grasps, t = [], 2.0 + rng.uniform(0, 1)
    while t < DUR - 8:
        on = t + rng.uniform(1.0, 2.0)            # reach
        off = on + rng.uniform(2.0, 3.5)          # hold + lift + place
        grasps.append((on, off))
        t = off + rng.uniform(2.0, 4.0)
    def contact(tt):
        return np.clip(sum(_smooth_step(tt, a, b) for a, b in grasps), 0, 1)
    def lifting(tt):  # load on the fingers while the object is off the table: middle part of each hold
        return np.clip(sum(_smooth_step(tt, a + 0.5, b - 0.5, 0.15) for a, b in grasps), 0, 1)

    # tactile clock: offset / drift / garbage
    n = int(DUR * 200)
    t_true = np.arange(n) / 200
    off, drift = faults.get("tactile_offset_s", 0.0), faults.get("tactile_drift_s_per_min", 0.0)
    ts_tac = t_true + off + drift * t_true / 60 + rng.normal(0, 0.0005, n)
    c, lift = contact(t_true), lifting(t_true)
    if faults.get("tactile_clock_garbage"):
        c = c[rng.permutation(n)]
        lift = lift[rng.permutation(n)]
    normal = (grip_n * SHARE)[None, :] * c[:, None] * (1 + 0.15 * rng.standard_normal((n, 1)).cumsum(0) / 400)
    normal += rng.normal(0, 0.05, (n, 5))
    shear_z = -0.4 * (grip_n * SHARE)[None, :] * lift[:, None] + rng.normal(0, 0.05, (n, 5))
    shear_x = rng.normal(0, 0.05, (n, 5))
    i = faults.get("dead_taxel")
    if i is not None:
        normal[:, i] = rng.normal(0, 0.01, n)
        shear_z[:, i] = rng.normal(0, 0.01, n)
    i = faults.get("zero_drift_taxel")
    if i is not None:
        normal[:, i] += faults["zero_drift_n"]
    i = faults.get("shear_reversed_taxel")
    if i is not None:
        shear_z[:, i] *= -1
    normal = np.clip(normal, 0 if not faults.get("zero_drift_n") else -1, SAT_N)
    tactile = np.stack([normal, shear_x, shear_z], axis=2)      # (n, 5, 3)

    # joints (true clock, 100 Hz): flexion follows the grasp, aperture follows the object
    m = int(DUR * 100)
    tj = np.arange(m) / 100
    cj = contact(tj)
    reach_open = np.clip(sum(_smooth_step(tj, a - 0.8, a + 0.1, 0.3) for a, b in grasps), 0, 1)
    open_mm = obj_width_mm + 25
    aperture = 40 + (open_mm - 40) * reach_open * (1 - cj) + (obj_width_mm - 40) * cj + rng.normal(0, 0.8, m)
    flex = np.clip(1 - (aperture - 20) / 120, 0, 1)
    ang = np.zeros((m, len(JOINTS)))
    for k, name in enumerate(JOINTS):
        if name.endswith("mcp_flex"):
            ang[:, k] = 10 + 60 * flex
        elif name.endswith("mcp_abd"):
            ang[:, k] = 5 + 8 * (1 - flex)
        elif name.endswith("pip"):
            ang[:, k] = 15 + 70 * flex
        elif name.endswith("dip"):
            ang[:, k] = 5 + 45 * flex
        else:
            ang[:, k] = 10 + 15 * np.sin(tj / 7)
    ang += rng.normal(0, 1.0, ang.shape)
    j = faults.get("calib_slip_joint")
    if j is not None:
        ang[:, JOINTS.index(j)] += 60

    # wrist camera: hand–object distance
    q = int(DUR * 30)
    tc = np.arange(q) / 30
    cc = contact(tc)
    phase = np.clip(sum(_smooth_step(tc, a - 1.2, b + 1.2, 1.0) for a, b in grasps), 0, 1)
    dist = (1 - cc) * (8 + 220 * (1 - phase)) + rng.normal(0, 2.0, q)
    return {"tactile": (ts_tac, tactile), "joints": (tj, ang, aperture), "wristcam": (tc, np.clip(dist, 0, None))}


# shift plan: (glove, operator, object, width mm, grip N, faults, expected codes, robots usable)
OBJ = {"tote handle": (60, 18), "cereal box": (70, 14), "wide bin": (100, 22), "drink pack": (75, 36), "server blade": (45, 24)}
ALL = ["parallel_gripper", "three_finger", "five_finger"]


def plan():
    rows = []
    objs = ["tote handle", "cereal box", "server blade", "wide bin", "drink pack", "cereal box", "tote handle", "server blade"]
    for g, op in (("G-03", "op-ana"), ("G-07", "op-ben"), ("G-11", "op-chen")):
        for k in range(8):
            rows.append({"glove": g, "operator": op, "k": k, "object": objs[(k + {"G-03": 0, "G-07": 3, "G-11": 5}[g]) % 8], "faults": {}, "expect": []})
    def ep(g, k):
        return next(r for r in rows if r["glove"] == g and r["k"] == k)
    ep("G-03", 2)["faults"] = {"calib_slip_joint": "index_pip"}; ep("G-03", 2)["expect"] = ["JOINT_OUT_OF_RANGE"]
    ep("G-03", 5)["faults"] = {"tactile_offset_s": 0.18}; ep("G-03", 5)["expect"] = ["SYNC_REALIGNED"]
    ep("G-03", 6)["faults"] = {"tactile_drift_s_per_min": 0.06}; ep("G-03", 6)["expect"] = ["SYNC_REALIGNED"]
    ep("G-03", 7)["faults"] = {"tactile_clock_garbage": True}; ep("G-03", 7)["expect"] = ["SYNC_UNVERIFIED"]
    ep("G-11", 1)["faults"] = {"shear_reversed_taxel": 0}; ep("G-11", 1)["expect"] = ["SHEAR_REVERSED"]
    for k in range(4, 8):                                       # ring taxel dies mid-shift
        ep("G-11", k)["faults"] = {"dead_taxel": 3}; ep("G-11", k)["expect"] = ["TAXEL_DEAD"]
    for k in range(8):                                          # middle taxel zero creeps up across the shift
        z = 0.3 * k
        ep("G-07", k)["faults"]["zero_drift_taxel"] = 2
        ep("G-07", k)["faults"]["zero_drift_n"] = z
        if z > 0.5:
            ep("G-07", k)["expect"].append("ZERO_DRIFT")
    return rows


def write(root: Path, seed: int = 26):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    labels = {}
    for r in plan():
        w, grip = OBJ[r["object"]]
        f = dict(r["faults"])
        if r["object"] == "drink pack" and r["glove"] == "G-07":   # heavy pack squeezed hard → index taxel clips
            grip = 48
            r["expect"].append("TAXEL_SATURATED")
        e = episode(rng, w, grip, f, r["k"])
        eid = f"{r['glove']}-e{r['k'] + 1:02d}"
        d = root / eid
        d.mkdir(exist_ok=True)
        np.savez_compressed(d / "tactile.npz", ts=e["tactile"][0], data=e["tactile"][1].astype(np.float32))
        np.savez_compressed(d / "joints.npz", ts=e["joints"][0], data=e["joints"][1].astype(np.float32), aperture=e["joints"][2].astype(np.float32))
        np.savez_compressed(d / "wristcam.npz", ts=e["wristcam"][0], data=e["wristcam"][1].astype(np.float32))
        (d / "meta.json").write_text(json.dumps({"ep_id": eid, "glove": r["glove"], "operator": r["operator"], "object": r["object"],
                                                 "shift_index": r["k"], "duration_s": DUR}, indent=1), encoding="utf-8")
        robots = [x for x in ALL if not (x == "parallel_gripper" and (w > 85 or grip > 40)) and not (x == "three_finger" and (w > 110 or grip > 30))]
        labels[eid] = {"codes": sorted(set(r["expect"])), "robots": None if f.get("tactile_clock_garbage") else robots}
    (root / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return labels
