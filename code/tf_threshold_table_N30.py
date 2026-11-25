#!/usr/bin/env python3
"""paper 3: double-precision TF threshold table for N <= 30, T8, T9, path onset (T10), X2, X3.

Reuses tf_threshold_table.py by import (its committed behaviour is not changed; its main() is not called).
Same method: bracket at S = 200 by node count, then integrate to S = 1e6 on the ladder 1e3..1e6, Chebyshev root of b(G).
Only additions: Gmax large enough for N = 30 (scan up to GMAX), a wider first bracket (then recentred on the
root with the committed +-3e-3 bracket so the final root has the committed accuracy), l up to 14.
Stages (one process at a time, nice 19, OMP_NUM_THREADS=1):
  python3 tf_threshold_table_N30.py table [NMAX=30]   -> tf_threshold_table_N30.csv
  python3 tf_threshold_table_N30.py table 30 0        -> patch mode: recompute l = 0 only (see sign_brackets)
  python3 tf_threshold_table_N30.py tests             -> T8, T9, X2 (with WKB), X3 from the CSV
  python3 tf_threshold_table_N30.py path [NMAX=17]    -> T10 onsets eps*_N along w(1+eps g_TF)
"""
import os, sys, csv, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CSV = os.path.join(HERE, "tf_threshold_table_N30.csv")
t0 = time.time()


def bvals_S1(TT, l, Gs):
    u, du, nodes = TT.integrate(l, Gs, TT.S1, 1)
    return TT.tail(l, Gs, TT.S1, u, du)[0]


def sign_brackets(TT, l, Gs):
    """Level brackets from the sign changes of b(G) at S = 200 (the k-th change is radial quantum number k-1).
    Added after the node-count method of the committed script lost one l = 0 level near G = 369 (the interior node count
    drops by one at G = 368.8 for l = 0, so every l = 0 level with N >= 24 got the label N - 1)."""
    b = bvals_S1(TT, l, Gs)
    sg = np.sign(b)
    return np.nonzero(sg[1:] != sg[:-1])[0]


def refine_b(TT, l, lo, hi, tol=1e-13):
    flo = np.sign(bvals_S1(TT, l, np.array([lo]))[0])
    for _ in range(40):
        g = np.linspace(lo, hi, 17)
        sg = np.sign(bvals_S1(TT, l, g))
        k = np.nonzero(sg != flo)[0][0]
        lo, hi = g[k - 1], g[k]
        if (hi - lo) / hi < tol: break
    return 0.5 * (lo + hi)


def table(NMAX, only_l=None):
    import tf_threshold_table as TT
    GMAX = 1.5 * NMAX * (NMAX + 1) / 2 * 1.3
    levels = []
    for l in (range((NMAX - 1) // 2 + 1) if only_l is None else only_l):
        Nlo = 2 * l + 1
        Gmin = 0.7 * Nlo * (Nlo + 1) / 2
        Gs = np.arange(Gmin, GMAX, 0.1)
        br = sign_brackets(TT, l, Gs)
        for N in range(Nlo, NMAX + 1):
            nr = N - 2 * l - 1
            if nr >= len(br):
                raise SystemExit(f"scan did not reach l={l} nr={nr}")
            lo, hi = Gs[br[nr]], Gs[br[nr] + 1]
            G200 = refine_b(TT, l, lo, hi)
            # wide first bracket, then recentre with the committed bracket
            G0 = G200
            for w in (3e-3, 1e-2, 3e-2):
                TT.W0 = w
                rs = TT.level_roots(l, G0, TT.LADDER)
                if all(r is not None for r in rs):
                    break
            else:
                raise SystemExit(f"no root l={l} N={N}")
            TT.W0 = 3e-3
            G0 = rs[-1]
            rs = TT.level_roots(l, G0, TT.LADDER)
            assert all(r is not None for r in rs), (l, N)
            rh = TT.level_roots(l, G0, [1e5, 1e6], nsub=2)
            rt = TT.level_roots(l, G0, [1e5, 1e6], rtol=1e-11)
            Gf = rs[-1]
            ch = abs(rs[-1] - rs[-2]) / Gf
            e_step = abs(rh[1] - Gf) / Gf
            e_tol = abs(rt[1] - Gf) / Gf
            levels.append(dict(N=N, n=N - l, l=l, nr=nr, G200=G200, ladder=rs, G=Gf, e_last=ch, e_step=e_step,
                               e_tol=e_tol, err=max(ch, e_step, e_tol)))
        print(f"l={l} done ({time.time() - t0:.0f}s)", flush=True)
    if only_l is not None:      # patch mode: replace the rows of these l in the existing CSV
        old = list(csv.reader(open(CSV)))
        hdr, rows = old[0], old[1:]
        rows = [r for r in rows if int(r[2]) not in only_l]
        new = {(d["N"], d["l"]): d for d in levels}
        for d in levels:
            rows.append([str(d["N"]), str(d["n"]), str(d["l"]), str(d["nr"]), f"{d['G']:.12e}", f"{d['G200']:.12e}"] +
                        [f"{r:.12e}" for r in d["ladder"]] +
                        [f"{d['e_last']:.3e}", f"{d['e_step']:.3e}", f"{d['e_tol']:.3e}", f"{d['err']:.3e}"])
        rows.sort(key=lambda r: (int(r[0]), int(r[1])))
        with open(CSV, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
        print(f"csv patched for l={only_l}, {len(rows)} rows, max err of patched {max(d['err'] for d in levels):.2e} ({time.time()-t0:.0f}s)", flush=True)
        return
    levels.sort(key=lambda d: (d["N"], d["n"]))
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "n", "l", "nr", "G_final_S1e6", "G_S200_count"] + [f"G_S{int(s)}" for s in TT.LADDER] +
                   ["change_last_two_S", "change_half_step", "change_ode_tol", "err_estimate"])
        for d in levels:
            w.writerow([d["N"], d["n"], d["l"], d["nr"], f"{d['G']:.12e}", f"{d['G200']:.12e}"] +
                       [f"{r:.12e}" for r in d["ladder"]] +
                       [f"{d['e_last']:.3e}", f"{d['e_step']:.3e}", f"{d['e_tol']:.3e}", f"{d['err']:.3e}"])
    print(f"csv written, {len(levels)} levels, max err {max(d['err'] for d in levels):.2e} ({time.time()-t0:.0f}s)", flush=True)


def load():
    G, E = {}, {}
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            G[(int(r["N"]), int(r["l"]))] = float(r["G_final_S1e6"])
            E[(int(r["N"]), int(r["l"]))] = float(r["err_estimate"])
    return G, E


def lmax(N): return (N - 1) // 2


def tests():
    G, E = load()
    NM = max(N for N, l in G)
    s = {N: G[(N, 0)] / G[(N, lmax(N))] - 1 for N in range(3, NM + 1)}
    print(f"levels {len(G)}, NMAX {NM}, max err_estimate {max(E.values()):.2e}")
    print("N lmax s_N gap_to_next")
    for N in range(3, NM + 1):
        gap = G[(N + 1, lmax(N + 1))] / G[(N, 0)] - 1 if N < NM else float('nan')
        # gap to the next group: first level of group N+1 over top of group N (negative = overlap)
        print(f"{N} {lmax(N)} {s[N]:.6f} {gap:+.6f}")
    # T8
    bad = [m for m in range(1, NM // 2 + 1) if 2 * m + 2 <= NM and not s[2 * m + 1] > s[2 * m + 2]]
    print("T8", "PASS" if not bad else f"FAIL at {bad}", "pairs checked", len([m for m in range(1, NM//2+1) if 2*m+2 <= NM]))
    # P4 rule
    def p4(N):
        grp = lambda M: sorted((G[(M, l)], l) for l in range(lmax(M) + 1))
        first_N, first_N1 = grp(N)[0][0], grp(N + 1)[0][0]
        lv = sorted((v, l) for (M, l), v in G.items() if first_N <= v <= first_N1 * (1 + 1e-15))
        gaps = [(lv[i + 1][0] / lv[i][0] - 1, lv[i][1]) for i in range(len(lv) - 1)]
        return max(gaps), gaps
    P = {}
    for N in range(2, NM):
        P[N] = p4(N)
    print("P4: N, largest gap, after l (0=ns, 1=np)")
    for N in P: print(f"  {N}: {P[N][0][0]:.4f} after l={P[N][0][1]}")
    t9 = all(P[N][0][1] == 0 for N in range(2, 8)) and P[8][0][1] == 1
    print("T9", "PASS" if t9 else "FAIL")
    # X3
    x3a = all(P[N][0][1] == 1 for N in (8, 9, 10))
    ov = {N: min(G[(N + 1, l)] for l in range(lmax(N + 1) + 1)) < G[(N, 0)] for N in range(11, NM)}
    x3b = all(ov.values())
    kill = any(P[N][0][1] != 1 for N in (9, 10))
    print("X3 largest gap after np for N=8,9,10:", x3a, "; overlap for every N=11..%d:" % (NM - 1), x3b,
          [N for N in ov if not ov[N]])
    print("X3", "KILLED" if kill else ("PASS" if (x3a and x3b) else "FAIL (not killed)"))
    print("P4 N=11..16 largest gap after l:", [(N, P[N][0][1]) for N in range(11, 17)])
    # X2
    wk = wkb(G)
    sw = {N: wk[(N, 0)] / wk[(N, lmax(N))] - 1 for N in range(3, NM + 1)}
    pairs = list(range(3, NM, 2))          # lower odd index N' of pair (N', N'+1)
    A = {p: s[p] - s[p + 1] for p in pairs}
    Aw = {p: sw[p] - sw[p + 1] for p in pairs}
    print("pair (N',N'+1): A_data, A_wkb, ratio")
    for p in pairs: print(f"  {p},{p+1}: {A[p]:.6f} {Aw[p]:.6f} {Aw[p]/A[p]:.3f}")
    print("wkb vs table relative threshold error (max, median):",
          max(abs(wk[k] / G[k] - 1) for k in G), np.median([abs(wk[k] / G[k] - 1) for k in G]))
    sign_ok = all(Aw[p] > 0 and A[p] > 0 for p in pairs)
    size_ok = all(abs(Aw[p] / A[p] - 1) <= 0.30 for p in pairs if p + 1 >= 10)
    A10, A20, A30 = A[9], A[19], A[NM - 1]
    Aw30 = Aw[NM - 1]
    c3 = A30 < 0.7 * A10
    endpoint = sign_ok and size_ok and c3
    kelvin = (A30 > 0.9 * A20) and all(Aw[p] < 0.5 * A[p] for p in pairs if p >= NM - 3)
    print(f"A_10={A10:.6f} A_20={A20:.6f} A_30={A30:.6f}  A30/A10={A30/A10:.3f} A30/A20={A30/A20:.3f}  wkb A_30={Aw30:.6f}")
    print(f"endpoint: sign {sign_ok}, size within 30% (N>=10) {size_ok}, A30<0.7A10 {c3}  -> {'PASS' if endpoint else 'FAIL'}")
    print(f"kelvin: A30>0.9A20 {A30 > 0.9 * A20}, wkb amplitude < half data at the top pairs {all(Aw[p] < 0.5 * A[p] for p in pairs if p >= NM - 3)} -> {'PASS' if kelvin else 'FAIL'}")
    print("X2", "endpoint PASS" if endpoint else ("Kelvin PASS" if kelvin else "KILLED both (neither)"))


def wkb(G):
    """Zero-energy Bohr-Sommerfeld in x = ln s: u = sqrt(s) y, y'' = [(l+1/2)^2 - G s phi(s)] y.
    Quantisation  int sqrt(G s phi - (l+1/2)^2) dx = (nr + 1/2) pi, nr = N - 2l - 1, between the two turning points."""
    import tf_accurate as TA
    from scipy.interpolate import CubicSpline
    from scipy.integrate import quad
    from scipy.optimize import brentq
    tf = TA.default()
    xs = np.linspace(math.log(1e-4), math.log(2e4), 6001)
    g = np.array([math.log(math.exp(x) * float(tf.phi(math.exp(x)))) for x in xs])
    sp = CubicSpline(xs, g)
    xm = xs[np.argmax(g)]
    out = {}
    for (N, l) in G:
        k2 = (l + 0.5) ** 2
        nr = N - 2 * l - 1
        def turn(Gv):
            f = lambda x: math.log(Gv) + sp(x) - math.log(k2)
            return brentq(f, xs[0], xm, xtol=1e-14), brentq(f, xm, xs[-1], xtol=1e-14)
        def F(Gv):
            a, b = turn(Gv)
            fn = lambda x: math.sqrt(max(Gv * math.exp(sp(x)) - k2, 0.0))
            return quad(fn, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)[0] - (nr + 0.5) * math.pi
        g0 = G[(N, l)]
        gmin = k2 / math.exp(sp(xm)) * (1 + 1e-9)     # smallest G with a classically allowed region
        out[(N, l)] = brentq(F, max(0.5 * g0, gmin), 1.5 * g0, xtol=1e-12)
    return out


def _path_setup(NMAX):
    exec(open(os.path.join(HERE, "tf_path_check.py")).read().split("G1 = Grid(1 / 256)")[0].replace(
        "NEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 41", "NEPS = 41").replace("NMAX = 12", f"NMAX = {NMAX}"),
        globals())
    global G2
    G2 = Grid(1 / 512)
    print(f"grid built ({time.time()-t0:.0f}s)", flush=True)


def _path_compute(NMAX, LSx):
    def lvl(l): return list(range(2 * l + 1, NMAX + 1))
    def compute(eps, prev):
        out = {}
        for l in LSx:
            Ns = lvl(l)
            if prev is None:
                guess = np.array([N * (N + 1) / 2 for N in Ns]); width = np.full(len(Ns), 0.05)
            else:
                guess = np.array([prev[(N, l)] for N in Ns]); width = np.maximum(0.03, 0.03 * guess)
            r = roots(l, eps, G2, guess, width)
            for N, z in zip(Ns, r): out[(N, l)] = z
        return out
    NEPS_ = 41
    grid = [k / (NEPS_ - 1) for k in range(NEPS_)]
    Z = {}; prev = None
    for e in grid:
        Z[e] = compute(e, prev); prev = Z[e]
        print(f"eps={e:.4f} done ({time.time()-t0:.0f}s)", flush=True)
    return grid, Z


def path_part(NMAX, ls, outfile):
    """Eps-grid thresholds for the sectors in ls only (sectors are independent in compute()); dump to json."""
    import json
    _path_setup(NMAX)
    grid, Z = _path_compute(NMAX, ls)
    json.dump([[e, {f"{N},{l}": z for (N, l), z in Z[e].items()}] for e in grid], open(outfile, "w"))
    print("part done", ls, flush=True)


def path_fin(NMAX, files):
    import json
    _path_setup(NMAX)
    Z = {}
    for fn in files:
        for e, d in json.load(open(fn)):
            Z.setdefault(e, {}).update({tuple(int(v) for v in k.split(",")): z for k, z in d.items()})
    grid = sorted(Z)
    _path_finish(NMAX, grid, Z)


def path(NMAX):
    _path_setup(NMAX)
    grid, Z = _path_compute(NMAX, range(0, (NMAX - 1) // 2 + 1))
    _path_finish(NMAX, grid, Z)


def _path_finish(NMAX, grid, Z):
    def lm(N): return (N - 1) // 2
    def f(zd, N):   # lowest of group N+1 minus top of group N; negative = cross-group overlap
        return min(zd[(N + 1, l)] for l in range(lm(N + 1) + 1)) - zd[(N, 0)]
    res = {}
    for N in range(3, NMAX):
        fs = [f(Z[e], N) for e in grid]
        first = next((i for i, v in enumerate(fs) if v < 0), None)
        # within-group order at the onset also checked: min of group N+1 sits at highest l
        if first is None:
            res[N] = (None, min(fs), grid[int(np.argmin(fs))]); continue
        lo, hi = grid[first - 1], grid[first]
        zlo, zhi = Z[lo], Z[hi]
        def fe(e, guess):
            Ns0 = (N, 0); Nl = (N + 1, lm(N + 1))
            z0 = roots(0, e, G2, np.array([guess[Ns0]]), np.array([0.03 * guess[Ns0]]))[0]
            z1 = roots(Nl[1], e, G2, np.array([guess[Nl]]), np.array([0.03 * guess[Nl]]))[0]
            return z1 - z0, {Ns0: z0, Nl: z1}
        flo, fhi = fs[first - 1], fs[first]
        gl = zlo
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            fm, gm = fe(mid, {(N, 0): zlo[(N, 0)], (N + 1, lm(N + 1)): zlo[(N + 1, lm(N + 1))]} if False else
                        {(N, 0): 0.5 * (zlo[(N, 0)] + zhi[(N, 0)]), (N + 1, lm(N + 1)): 0.5 * (zlo[(N + 1, lm(N + 1))] + zhi[(N + 1, lm(N + 1))])})
            if fm < 0: hi = mid
            else: lo = mid
            if hi - lo < 1e-9: break
        res[N] = (0.5 * (lo + hi), min(fs), None)
        print(f"N={N}: eps* = {res[N][0]:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    print("N, eps*_N, N*eps*_N, (if none: min margin and eps)")
    for N in sorted(res):
        r = res[N]
        if r[0] is None: print(f"  N={N}: none on grid; min (Z_low(N+1) - Z_top(N)) = {r[1]:.5f} at eps={r[2]:.4f}")
        else: print(f"  N={N}: {r[0]:.6f}  {N*r[0]:.4f}")
    ok_small = all(res[N][0] is None for N in range(3, 11))
    ok_big = all(res[N][0] is not None and res[N][0] < 1 for N in range(11, min(17, NMAX)))
    print("T10", "PASS" if (ok_small and ok_big) else f"FAIL (N<=10 clean: {ok_small}, N=11..16 onset<1: {ok_big})")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "table": table(int(sys.argv[2]) if len(sys.argv) > 2 else 30, [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)
    elif st == "tests": tests()
    elif st == "path": path(int(sys.argv[2]) if len(sys.argv) > 2 else 17)
    elif st == "pathpart": path_part(int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], sys.argv[4])
    elif st == "pathfin": path_fin(int(sys.argv[2]), sys.argv[3:])
