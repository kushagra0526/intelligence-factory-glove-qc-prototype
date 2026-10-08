"""Run glove-episode QC over a synthetic data-factory shift and write out/report.html.

    python demo.py            # generates data/shift (seeded) if missing, runs QC, scores vs planted faults
"""
from __future__ import annotations

import html
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from gloveqc import qc
from gloveqc.synth import FINGERS, write

HERE = Path(__file__).parent
DATA, OUT = HERE / "data" / "shift", HERE / "out"
COL = {"ACCEPT": "#1f7a4d", "FIXED": "#2b6cb0", "REJECT": "#b3261e"}


def score(reports, labels):
    tp = fp = fn = 0
    rob_ok = rob_n = 0
    for r in reports:
        L = labels[r.ep_id]
        got, want = set(r.codes), set(L["codes"])
        tp, fp, fn = tp + len(got & want), fp + len(got - want), fn + len(want - got)
        if L["robots"] is not None:
            rob_n += 1
            rob_ok += r.robots == L["robots"]
    return tp, fp, fn, rob_ok, rob_n


def svg_sync(ep: Path) -> str:
    """Per-window tactile offsets before and after re-alignment for one drifting episode."""
    tac, cam = np.load(ep / "tactile.npz"), np.load(ep / "wristcam.npz")
    prox = (cam["ts"], np.exp(-cam["data"] / 15.0))
    total = tac["data"][:, :, 0].sum(axis=1)
    w = qc.window_offsets(prox, (tac["ts"], total), 120)
    c, o = np.array([x[0] for x in w]), np.array([x[1] for x in w])
    icpt, slope, _ = qc.robust_fit(c, o, 120)
    w2 = qc.window_offsets(prox, ((tac["ts"] - icpt) / (1 + slope), total), 120)
    X = lambda t: 50 + t / 120 * 520
    Y = lambda v: 150 - v * 1000 * 1.1
    pts = "".join(f"<circle cx={X(a):.1f} cy={Y(b):.1f} r=4 fill='#b3261e'/>" for a, b, _ in w)
    pts2 = "".join(f"<circle cx={X(a):.1f} cy={Y(b):.1f} r=4 fill='#1f7a4d'/>" for a, b, _ in w2)
    return f"""<svg viewBox='0 0 600 230' width=600><rect x=50 y={Y(0.033):.0f} width=520 height={Y(-0.033) - Y(0.033):.0f} fill='#e8f3ec'/>
<line x1=50 x2=570 y1={Y(0)} y2={Y(0)} stroke='#999'/><line x1={X(0)} y1={Y(icpt)} x2={X(120)} y2={Y(icpt + slope * 120)} stroke='#b3261e' stroke-dasharray='4'/>
{pts}{pts2}<text x=55 y={Y(0.033) - 4:.0f} font-size=11 fill='#1f7a4d'>±33 ms spec</text>
<text x=50 y=225 font-size=11 fill='#555'>0 s</text><text x=545 y=225 font-size=11 fill='#555'>120 s</text>
<text x=380 y=20 font-size=12 fill='#b3261e'>● before: drift {slope * 60e3:+.0f} ms/min</text><text x=380 y=38 font-size=12 fill='#1f7a4d'>● after re-alignment</text></svg>"""


def svg_drift(reports, glove="G-07", taxel=2) -> str:
    rs = sorted((r for r in reports if r.meta["glove"] == glove), key=lambda r: r.meta["shift_index"])
    z = [r.baselines[taxel] for r in rs]
    X = lambda i: 110 + i * 62
    Y = lambda v: 170 - v * 70
    bars = "".join(f"<rect x={X(i) - 18} y={Y(v):.1f} width=36 height={170 - Y(v):.1f} fill='{'#b3261e' if v > qc.ZERO_N else '#8fbf9f'}'/>"
                   f"<text x={X(i) - 14} y=188 font-size=11>e{i + 1:02d}</text><text x={X(i) - 14} y={Y(v) - 4:.1f} font-size=10>{v:.2f}</text>" for i, v in enumerate(z))
    return f"""<svg viewBox='0 0 600 200' width=600>{bars}<line x1=20 x2=590 y1={Y(qc.ZERO_N)} y2={Y(qc.ZERO_N)} stroke='#b3261e' stroke-dasharray='4'/>
<text x=22 y={Y(qc.ZERO_N) - 6} font-size=11 fill='#b3261e'>limit {qc.ZERO_N} N</text></svg>"""


def main() -> int:
    t0 = time.perf_counter()
    labels = write(DATA) if not (DATA / "labels.json").exists() else json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    reports = qc.run_all(DATA)
    tp, fp, fn, rob_ok, rob_n = score(reports, labels)
    v = Counter(r.verdict for r in reports)
    minutes = {k: 2 * n for k, n in v.items()}

    print(f"\n== shift: {len(reports)} episodes × 2 min, 3 gloves, 3 operators\n")
    for r in reports:
        print(f"  {r.ep_id:<10} {r.verdict:<7} {r.meta['object']:<13} robots: {', '.join(x.replace('_', ' ') for x in r.robots) or '—'}")
        for x in r.reasons:
            print(f"  {'':18}{x}")
    print(f"\n== verdicts: {v['ACCEPT']} ACCEPT · {v['FIXED']} FIXED · {v['REJECT']} REJECT  → "
          f"{minutes.get('ACCEPT', 0) + minutes.get('FIXED', 0)} of {2 * len(reports)} min usable")
    print(f"== vs planted faults: {tp} found, {fp} false, {fn} missed · robot tags correct {rob_ok}/{rob_n}")
    roll = qc.glove_rollup(reports)
    print("\n== glove rollup (act before the next episode):")
    for x in roll:
        print("  " + x)

    OUT.mkdir(exist_ok=True)
    rows = "".join(
        f"<tr><td class=m>{r.ep_id}</td><td>{html.escape(r.meta['operator'])}</td><td>{html.escape(r.meta['object'])}</td>"
        f"<td><span class=p style='background:{COL[r.verdict]}'>{r.verdict}</span></td>"
        f"<td>{''.join(f'<div>{html.escape(x)}</div>' for x in r.reasons) or '<span class=ok>clean</span>'}</td>"
        f"<td>{' '.join(f'<span class=t>{x.replace(chr(95), chr(32))}</span>' for x in r.robots) or '—'}</td></tr>" for r in reports)
    drifting = next(d for d in sorted(DATA.iterdir()) if d.name == "G-03-e07")
    (OUT / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Glove episode QC</title>
<style>body{{font:14px/1.45 'Segoe UI',system-ui,sans-serif;margin:28px;max-width:1450px;color:#1d1d1f;background:#fafaf8}}h1{{font-size:23px;margin:0}}
h2{{font-size:17px;margin:22px 0 8px}}.sub{{color:#666}}table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{border-top:1px solid #e5e5e0;padding:6px 9px;text-align:left;vertical-align:top}}
.p{{color:#fff;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600}}.m{{font-family:Consolas,monospace;white-space:nowrap}}.ok{{color:#1f7a4d}}
.t{{background:#eef;border-radius:4px;padding:0 5px;font-size:12px;white-space:nowrap}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}</style>
<h1>Glove episode QC · synthetic data-factory shift</h1>
<div class=sub>{len(reports)} episodes · tactile (5 taxels × normal/shear) + 21-DOF IMU + wrist-camera contact cue on separate clocks ·
{v['ACCEPT']} accept, {v['FIXED']} fixed, {v['REJECT']} reject → <b>{minutes.get('ACCEPT', 0) + minutes.get('FIXED', 0)} of {2 * len(reports)} min usable</b> ·
vs planted faults: {tp} found, {fp} false, {fn} missed · robot-fit tags {rob_ok}/{rob_n}</div>
<h2>Glove rollup: act before the next episode</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in roll)}</ul>
<div class=g><div><h2>Tactile↔camera sync, G-03-e07 (planted drift 60 ms/min)</h2>{svg_sync(drifting)}</div>
<div><h2>G-07 middle taxel zero across the shift</h2>{svg_drift(reports)}</div></div>
<h2>Episodes</h2><table><tr><th>episode</th><th>operator</th><th>object</th><th>verdict</th><th>findings</th><th>usable for</th></tr>{rows}</table>
<p class=sub>Synthetic data. Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Intelligence Factory.</p>""", encoding="utf-8")
    print(f"\nreport: out/report.html  ({time.perf_counter() - t0:.0f}s)")
    return 0 if fp == fn == 0 and rob_ok == rob_n else 1


if __name__ == "__main__":
    sys.exit(main())
