#!/usr/bin/env python3
"""Figures of paper 3 (first break of the Madelung order in the Thomas-Fermi atom): fig1..fig4, PDF and PNG (200 dpi) in out/.

Figure to data (all in code/):
  fig1  thresholds G(n,l) against n+l, N <= 30, and the gap between groups    tf_threshold_table_N30.csv
  fig2  groups 10 to 12 with the certified brackets, zoom on 7h < 11s < 8g     cor_c_z_ranges.log (brackets of tf_overlap_certificate_N12.log)
  fig3  Z intervals of first filling, N <= 12, break region near Z = 309 to 313  cor_c_z_ranges.log
  fig4  group spread s_N = G(N,0)/G(N,lmax) - 1 against 1/N, semiclassical lines  tf_threshold_table_N30.csv
Run (seconds, plotting only):  python3 make_figures.py
Every figure goes through check_layout (no text over text, over a line, arrow, marker or patch outline, or off the
canvas); a violation raises and the exit status is nonzero.  Layout check only, no files written:  --check
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
DATA = HERE

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, TEAL, GREY, INK = "#1f4e79", "#c8553d", "#3b8b8b", "#8a8f98", "#222831"
PURPLE = "#8e6bb0"
GOLD = "#b8860b"
LCOL = {0: BLUE, 1: ORANGE, 2: TEAL, 3: PURPLE, 4: GOLD, 5: "#b03060", 6: "#4a7c3a", 7: GREY}
plt.rcParams.update({
    "font.size": 11.5, "axes.labelsize": 12.5, "legend.fontsize": 10.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "serif", "font.serif": ["CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "pdf.fonttype": 42,
})

CHECK_ONLY = "--check" in sys.argv   # draw and layout-check only, write no files


# ------------------------------------------------------------ layout check
# Every figure is checked after drawing (see check_layout). A Text is exempt from the
# "text on an artist" test only if it carries a gid starting with "own-label": it is the
# label of a filled box or bar it sits inside on purpose (panel (a) subshell labels and
# the period numbers of panel (b) of the Madelung figure). Exempt texts are still tested
# against every other text and against the canvas edge. Tick labels touching axis spines
# are not tested (spines are not collected).
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle as _Rect
from matplotlib.transforms import Bbox
from matplotlib.legend import Legend


def _tight(bb, pad=0.0):
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def check_layout(fig, name):
    """Raise AssertionError if any text overlaps another text, touches a line, arrow,
    marker or patch outline, or leaves the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    texts = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and t in (t.axes.title, t.axes._left_title, t.axes._right_title) and not t.get_text():
            continue
        bb = t.get_window_extent(r)
        if not np.isfinite([bb.x0, bb.x1, bb.y0, bb.y1]).all() or bb.width == 0:
            continue
        # skip tick labels that matplotlib keeps but does not draw (outside the view)
        texts.append((t, bb))
    # tick labels outside the axes view are not drawn: drop them
    drawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_majorticklabels() + axis.get_minorticklabels():
                drawn.add(id(t))
    inview = []
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for axis, hor in ((ax.xaxis, True), (ax.yaxis, False)):
            ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
            lo, hi = axis.get_view_interval()
            for tk in ticks:
                loc = tk.get_loc()
                if not (min(lo, hi) - 1e-9 <= loc <= max(lo, hi) + 1e-9):
                    tk.label1.set_visible(False)
    texts = [(t, bb) for t, bb in texts if t.get_visible()]
    # artists
    items = []   # (label, path in display coords or None, marker boxes)
    for ax in fig.axes:
        for ln in ax.lines:
            if len(ln.get_xdata()) == 0 or not ln.get_visible():
                continue
            if ln.get_color() == "none":
                continue
            path = ln.get_transform().transform_path(ln.get_path())
            if ln.get_linestyle() in ("None", "none", " ", ""):
                path = None
            boxes = []
            if ln.get_marker() not in ("None", "none", None, ""):
                pts = ln.get_transform().transform(np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)]))
                h = ln.get_markersize() * dpi / 72 / 2 + ln.get_markeredgewidth()
                boxes = [Bbox.from_extents(x - h, y - h, x + h, y + h) for x, y in pts if np.isfinite([x, y]).all()]
            items.append((f"line {ln.get_label()!r}", path, boxes))
        for pa in ax.patches:
            if not pa.get_visible() or pa is ax.patch:
                continue
            path = _flat(pa.get_transform().transform_path(pa.get_path()))
            if isinstance(pa, _Rect) and not pa.get_ec()[3] and pa.get_fc()[3] == 0:
                continue
            items.append((f"patch {type(pa).__name__}", path, []))
        for ch in ax.get_children():
            if isinstance(ch, Text) and hasattr(ch, "arrow_patch") and ch.arrow_patch is not None:
                pa = ch.arrow_patch
                items.append(("annotation arrow", _flat(pa.get_path().transformed(pa.get_transform())), []))
        for lg in [ax.get_legend()]:
            if lg is not None and lg.get_frame_on():
                fb = lg.get_frame().get_window_extent(r)
                items.append(("legend frame", __import__("matplotlib").path.Path.unit_rectangle().transformed(
                    __import__("matplotlib.transforms", fromlist=["x"]).BboxTransformTo(fb)), []))
    problems = []
    fb = fig.bbox
    for ax in fig.axes:   # a legend must lie inside its axes (not clipped, not over the spine)
        lg = ax.get_legend()
        if lg is not None:
            lb, ab = lg.get_window_extent(r), ax.get_window_extent(r)
            if lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1 or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1:
                problems.append("legend extends beyond its axes")
    for i, (t, bb) in enumerate(texts):
        s = repr(t.get_text())[:30]
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            problems.append(f"text {s} extends beyond the canvas")
        for t2, bb2 in texts[i + 1:]:
            if bb.overlaps(bb2) and bb.fully_overlaps(bb2) or (bb.overlaps(bb2) and _inter(bb, bb2) > 0.5):
                problems.append(f"text {s} overlaps text {repr(t2.get_text())[:30]}")
        if (t.get_gid() or "").startswith("own-label"):
            continue
        tb = _tight(bb, 0.5)
        for lab, path, boxes in items:
            if path is not None and path.intersects_bbox(tb, filled=False):
                problems.append(f"text {s} touches {lab}")
            elif any(tb.overlaps(b) for b in boxes):
                problems.append(f"text {s} touches a marker of {lab}")
    if problems:
        raise AssertionError(f"layout check FAILED for {name}:\n  " + "\n  ".join(sorted(set(problems))))
    print(f"layout {name}: PASS ({len(texts)} texts, {len(items)} artists)")


def _flat(path):
    """Polyline version of a path (Bezier control points of circles and arcs lie off the curve)."""
    from matplotlib.path import Path
    verts, codes = [], []
    for poly in path.to_polygons(closed_only=False):
        verts.extend(poly)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(poly) - 1))
    return Path(np.array(verts), codes) if verts else path


def _inter(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return min(w, h)


def save(fig, name):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(pad=0.6)
    if os.environ.get("NOCHECK") != "1":
        check_layout(fig, name)
    if CHECK_ONLY:
        plt.close(fig)
        return
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ data
LET = "spdfghi"


def lab(n, l):
    return f"{n}{LET[l]}"


def load_table():
    """rows (N, n, l, G) from the N <= 30 table; N is the group number n + l."""
    rows = []
    with open(os.path.join(DATA, "tf_threshold_table_N30.csv")) as fh:
        next(fh)
        for ln in fh:
            f = ln.split(",")
            rows.append((int(f[0]), int(f[1]), int(f[2]), float(f[4])))
    return rows


def load_brackets():
    """rows (N, n, l, G_lo, G_hi, Z_lo, Z_hi) of the certified table in cor_c_z_ranges.log."""
    import re
    out = []
    with open(os.path.join(DATA, "cor_c_z_ranges.log")) as fh:
        for ln in fh:
            m = re.match(r"(\d+)([spdfghi])\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+\[[\d.]+, [\d.]+\])?\s*$", ln)
            if m:
                n, l = int(m[1]), LET.index(m[2])
                out.append((int(m[3]), n, l, float(m[4]), float(m[5]), float(m[6]), float(m[7])))
    return out


ROWS = load_table()
BR = load_brackets()
assert len(ROWS) == 240
BR = list({(b[1], b[2]): b for b in BR}.values())
assert sum(1 for b in BR if b[0] <= 12) == 42, len(BR)
NMAX = 30
G = {(n, l): g for (_N, n, l, g) in ROWS}
lmax = lambda N: (N - 1) // 2


def group_edges(N):
    return G[(N - lmax(N), lmax(N))], G[(N, 0)]    # lowest (largest l) and highest (l = 0)


# ------------------------------------------------------------ fig1
def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for N in range(1, NMAX + 1):
        lo, hi = group_edges(N)
        a.add_patch(plt.Rectangle((N - 0.34, lo), 0.68, hi - lo, fc="#e3e7ec", ec="none", zorder=0))
    for N, n, l, g in ROWS:
        a.plot([N], [g], "o", ms=3.6, color=LCOL.get(l, GREY), mew=0, zorder=3)
    for l in range(0, 8):
        a.plot([], [], "o", ms=5, color=LCOL[l], mew=0, label=f"$l={l}$" if l < 7 else "$l\\geq 7$")
    a.set_xlabel("$n+l$")
    a.set_ylabel("threshold $G(n,l)$")
    a.set_xlim(0, NMAX + 1)
    a.set_ylim(0, 1000 if False else max(g for *_x, g in ROWS) * 1.05)
    a.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.1, columnspacing=0.8)
    a.text(0.97, 0.05, "grey bands: groups", transform=a.transAxes, ha="right", color=GREY, fontsize=10)
    a.set_title("(a)", loc="left", fontsize=11.5)
    # (b) gap between the top of group N and the bottom of group N+1, relative
    Ns = list(range(2, NMAX))
    gap = [(group_edges(N + 1)[0] - group_edges(N)[1]) / group_edges(N)[1] for N in Ns]
    b.axhline(0, color=INK, lw=0.8)
    b.axvspan(10.5, NMAX - 0.5, color="#f4e3dd", lw=0, zorder=0)
    b.plot(Ns, gap, "o-", ms=3.6, lw=0.9, color=INK, zorder=3)
    b.plot([11], [gap[9]], "o", ms=7, mfc="none", mec=ORANGE, mew=1.6, zorder=4)
    b.text(15.5, 0.22, "overlap from $N=11$", color=ORANGE, ha="left", va="bottom")
    b.plot([15.3, 11.45], [0.2, gap[9] + 0.012], "-", lw=0.9, color=ORANGE, zorder=2)
    b.set_xlabel("group $N=n+l$")
    b.set_ylabel("relative gap to the next group")
    b.set_xlim(1, NMAX)
    b.set_title("(b)", loc="left", fontsize=11.5)
    # mark 11s and 7h in (a): the two levels in the zoom
    for (n, l) in ((11, 0), (7, 5)):
        N = n + l
        a.plot([N], [G[(n, l)]], "o", ms=9, mfc="none", mec=INK, mew=1.2, zorder=4)
    a.text(7.0, 175, "$11s$", color=INK, ha="right", va="center")
    a.plot([7.4, 10.7], [168, 88], "-", lw=0.8, color=INK, zorder=2)
    a.text(15.0, 25, "$7h$", color=INK, ha="left", va="center")
    a.plot([14.6, 12.5], [30, 72], "-", lw=0.8, color=INK, zorder=2)
    save(fig, "fig1")


# ------------------------------------------------------------ fig2
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1.05]})
    sel = [r for r in BR if 10 <= r[0] <= 12]
    for N, n, l, lo, hi, zl, zh in sel:
        mid = 0.5 * (lo + hi)
        a.errorbar([N], [mid], yerr=[[mid - lo], [hi - mid]], fmt="o", ms=4, color=LCOL[l], capsize=2, lw=1.2)
    a.set_xticks([10, 11, 12])
    a.set_xlim(9.5, 12.5)
    a.set_xlabel("group $N=n+l$")
    a.set_ylabel("threshold $G(n,l)$")
    for l in range(0, 6):
        a.plot([], [], "o", ms=5, color=LCOL[l], mew=0, label=f"$l={l}$")
    a.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.1, columnspacing=0.8)
    a.set_title("(a)", loc="left", fontsize=11.5)
    a.set_ylim(55, 108)
    a.add_patch(plt.Rectangle((11.55, 80.4), 0.9, 3.0, fc="none", ec=GREY, lw=0.9, ls="--"))
    # (b) zoom: levels with G between 80.5 and 83 : 7h, 11s, 8g (plus nothing else)
    zoom = [r for r in sel if 80.5 < r[3] < 83]
    zoom.sort(key=lambda r: r[3])
    assert [lab(r[1], r[2]) for r in zoom] == ["7h", "11s", "8g"], zoom
    for i, (N, n, l, lo, hi, zl, zh) in enumerate(zoom):
        mid = 0.5 * (lo + hi)
        b.errorbar([i], [mid], yerr=[[mid - lo], [hi - mid]], fmt="o", ms=5, color=LCOL[l], capsize=4, lw=1.6)
        b.text(i, hi + 0.12, f"${lab(n, l)}$", ha="center", va="bottom", fontsize=12.5)
        b.text(i, lo - 0.12, f"[{lo:.3f}, {hi:.3f}]", ha="center", va="top", fontsize=9.5, color=GREY)
    b.set_xlim(-0.6, 2.6)
    b.set_ylim(80.2, 83.3)
    b.set_xticks([])
    b.spines["bottom"].set_visible(False)
    b.set_ylabel("threshold $G$ (certified bracket)")
    b.set_title("(b)", loc="left", fontsize=11.5)
    b.text(0.5, 0.02, "$7h<11s<8g$", transform=b.transAxes, ha="center", fontsize=12, color=INK)
    save(fig, "fig2")


# ------------------------------------------------------------ fig3
S_INF, C_ODD, C_EVEN = 0.25209, -0.666, -0.747


def fig_spread():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Ns = range(3, NMAX + 1)
    s = {N: G[(N, 0)] / group_edges(N)[0] - 1 for N in Ns}
    x = np.linspace(0, 1 / 3, 50)
    ax.plot(x, S_INF + C_ODD * x, "-", lw=1.1, color=BLUE, zorder=2, label="odd $N$: $s_\\infty-0.666/N$")
    ax.plot(x, S_INF + C_EVEN * x, "--", lw=1.1, color=ORANGE, zorder=2, label="even $N$: $s_\\infty-0.747/N$")
    ax.axhline(S_INF, color=GREY, lw=0.8, ls=":", zorder=1)
    ax.plot([0], [S_INF], "D", ms=5, color=INK, zorder=4, clip_on=False)
    odd = [N for N in Ns if N % 2]
    even = [N for N in Ns if N % 2 == 0]
    ax.plot([1 / N for N in odd], [s[N] for N in odd], "o", ms=5, color=BLUE, zorder=3, label="data, odd $N$")
    ax.plot([1 / N for N in even], [s[N] for N in even], "s", ms=5, color=ORANGE, zorder=3, label="data, even $N$")
    ax.set_xlabel("$1/N$")
    ax.set_ylabel("group spread $s_N$")
    ax.set_xlim(-0.012, 1 / 3 + 0.01)
    ax.set_ylim(0, 0.3)
    ax.text(0.012, S_INF + 0.007, "$s_\\infty=0.25209$", color=INK, ha="left", va="bottom")
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig4")


# ------------------------------------------------------------ fig4
def fig_z():
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 6.6), gridspec_kw={"height_ratios": [1.5, 1]})
    sel = [r for r in BR if r[0] <= 12]
    z7h = next(r for r in sel if (r[1], r[2]) == (7, 5))
    z11s = next(r for r in sel if (r[1], r[2]) == (11, 0))
    zlo, zhi = z7h[5], z11s[6]
    a.axvspan(zlo, zhi, color="#f4e3dd", lw=0, zorder=0)
    for N, n, l, gl, gh, zl, zh in sel:
        a.plot([0.5 * (zl + zh)], [N], "|", ms=11, mew=2.6, color=LCOL[l], zorder=3)
    a.set_yticks(range(1, 13))
    a.set_ylim(0.4, 12.6)
    a.set_xlim(-15, 420)
    a.set_ylabel("group $N=n+l$")
    a.set_xlabel("$Z$")
    for l in range(0, 6):
        a.plot([], [], "|", ms=11, mew=2.6, color=LCOL[l], label=f"$l={l}$")
    a.legend(loc="lower right", frameon=False, ncol=2)
    a.text(zlo - 6, 4, "break region", color=ORANGE, ha="right", va="center")
    a.set_title("(a)", loc="left", fontsize=11.5)
    # (b) zoom on the break
    near = [r for r in sel if 300 <= r[5] <= 322]
    near.sort(key=lambda r: r[5])
    ys = {(r[1], r[2]): k for k, r in enumerate(near)}
    b.cla()
    b.axvspan(zlo, zhi, color="#f4e3dd", lw=0, zorder=0)
    for r in near:
        N, n, l, gl, gh, zl, zh = r
        b.plot([zl, zh], [ys[(n, l)]] * 2, "-", lw=7, color=LCOL[l], solid_capstyle="butt", zorder=3)
        b.text(zh + 0.5, ys[(n, l)], f"${lab(n, l)}$", va="center", ha="left")
    b.set_ylim(-0.7, len(near) - 0.3)
    b.set_xlim(300, 322)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel("$Z$ (zoom)")
    b.set_title("(b)", loc="left", fontsize=11.5)
    save(fig, "fig3")


if __name__ == "__main__":
    for f in (fig1, fig2, fig_z, fig_spread):
        f()
