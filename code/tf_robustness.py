#!/usr/bin/env python3
"""Robustness of the paper 3 orderings to corrected one-body fields (robustness-paper3.md; F2, F3 of review-paper3-r3.md).

Problem: u'' = [l(l+1)/s^2 - W(s)/s - eps] u, scaled units of the paper, G = 2 b Z = C_G Z^(2/3), tail t = 2 b = C_G Z^(-1/3),
C_G = 2 * 0.8853 (same constant as the committed table: Z(8g) = 317.4 from G = 82.389).
Fields: W = max(A phi, c) (Latter / kappa interpolation, A = G, c = kappa t) or W = A phi + c (Fermi-Amaldi, A = G (1-1/Z), c = t).
Engine: u = sqrt(s) f, x = ln s, f'' = [(l+1/2)^2 - W s - eps s^2] f, Numerov with step H on x in [ln S0, ln S1], vectorised over
parameter sets (l, Z, eps); counting sign changes of f gives the number of eigenvalues below eps (Sturm), and eigenvalues
are found by vectorised bisection.  Double precision; phi from tf_accurate (the same 60 digit TF function as the paper).
Modes:  python3 tf_robustness.py check | aufbau [Zmax] | cross
Output: tf_robustness.log (written by the caller, tee).  Cached phi grid: tf_robustness_phi.npy (derived data, optional).
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CG = 2 * 0.8853
LET = "spdfghi"
S0, S1, H = 1e-6, 1e4, 0.01
X = np.arange(np.log(S0), np.log(S1) + H / 2, H)
SS = np.exp(X)
_cache = os.path.join(HERE, "tf_robustness_phi.npy")


def phigrid():
    if os.path.exists(_cache):
        a = np.load(_cache)
        if a.shape == SS.shape:
            return a
    import tf_accurate as TA
    tf = TA.default()
    a = np.array([tf.phi(float(s)) for s in SS])
    try:
        np.save(_cache, a)
    except OSError:
        pass
    return a


PHI = None


def count(l, A, c, mode, eps, W0=None):
    """number of eigenvalues below eps; all args arrays of equal shape (M,). mode 0: W = max(A phi, c); 1: W = A phi + c."""
    global PHI
    if PHI is None:
        PHI = phigrid()
    l, A, c, eps = [np.asarray(v, float) for v in (l, A, c, eps)]
    mode = np.asarray(mode)
    h2 = H * H / 12.0
    L2 = (l + 0.5) ** 2

    def Q(n):
        ap = A * PHI[n]
        w = np.where(mode == 0, np.maximum(ap, c), ap + c)
        return L2 - w * SS[n] - eps * SS[n] ** 2
    w0 = np.where(mode == 0, np.maximum(A, c), A + c)
    fa = np.exp((l + 0.5) * X[0]) * (1 - w0 * S0 / (2 * l + 2))
    fb = np.exp((l + 0.5) * X[1]) * (1 - w0 * SS[1] / (2 * l + 2))
    Qa, Qb = Q(0), Q(1)
    cnt = np.zeros(l.shape, int)
    done = np.zeros(l.shape, bool)      # beyond the turning point Numerov is unreliable (h^2 Q/12 > 0.2) and there are no more nodes
    np.seterr(all="ignore")
    for n in range(1, len(X) - 1):
        Qc = Q(n + 1)
        fc = (2 * (1 + 5 * h2 * Qb) * fb - (1 - h2 * Qa) * fa) / (1 - h2 * Qc)
        done |= (h2 * Qc > 0.2)
        cnt += ((fc * fb < 0) & ~done)
        fa, fb, Qa, Qb = fb, fc, Qb, Qc
        if n % 16 == 0:
            m = np.maximum(np.abs(fa), np.abs(fb))
            m = np.where(m > 1e60, m, 1.0)
            fa = fa / m; fb = fb / m
    return cnt


def fieldpar(Z, field, kappa=1.0):
    """A, c, mode at charge Z for a field name."""
    Z = np.asarray(Z, float)
    G = CG * Z ** (2 / 3)
    t = CG * Z ** (-1 / 3)
    if field == "latter":
        return G, t, 0
    if field == "interp":
        return G, kappa * t, 0
    if field == "fa":
        return G * (1 - 1 / Z), t, 1
    if field == "fa0":      # rescaling only, no tail
        return G * (1 - 1 / Z), 0 * t, 0
    if field == "tf":
        return G, 0 * t, 0
    raise ValueError(field)


def eigen(l, Zs, ks, field, kappa=1.0, it=44):
    """E_{k,l}(Z) for arrays Zs, ks (same shape). E = 0 means not bound (field tf, fa0)."""
    Zs = np.asarray(Zs, float); ks = np.asarray(ks)
    A, c, mode = fieldpar(Zs, field, kappa)
    A = A + 0 * Zs; c = c + 0 * Zs
    lo = -(np.maximum(A, c) ** 2) / 4 * 1.2 - 1.0
    hi = np.full(Zs.shape, -1e-12)
    ll = np.full(Zs.shape, float(l))
    cb = count(ll, A, c, mode, hi)
    ok = cb >= ks + 1
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        cm = count(ll, A, c, mode, mid)
        up = cm >= ks + 1
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), 0.0)


def cap(l): return 2 * (2 * l + 1)


def aufbau(field, kappa=1.0, Zmax=330, Nmax=14):
    Zs = np.arange(1, Zmax + 1, dtype=float)
    lev = []   # (N, n, l, k)
    E = {}
    for l in range(0, (Nmax - 1) // 2 + 1):
        kmax = Nmax - 2 * l - 1
        Zg = np.repeat(Zs[None, :], kmax + 1, 0).ravel()
        kg = np.repeat(np.arange(kmax + 1)[:, None], len(Zs), 1).ravel()
        e = eigen(l, Zg, kg, field, kappa).reshape(kmax + 1, len(Zs))
        for k in range(kmax + 1):
            E[(l, k)] = e[k]
        print(f"  {field} kappa={kappa} l={l} done {time.time()-T0:.0f}s", flush=True)
    first = {}
    for iz, Z in enumerate(Zs):
        items = sorted(((E[key][iz], key) for key in E if E[key][iz] < 0))
        rem = int(Z)
        for e, (l, k) in items:
            if rem <= 0: break
            if (l, k) not in first:
                first[(l, k)] = int(Z)
            rem -= cap(l)
    return first, E


def madelung_report(Zfirst, label):
    """Zfirst: dict (l,k) -> first Z.  Level N = n + l = k + 2l + 1.  Returns summary dict."""
    lv = []
    for (l, k), z in Zfirst.items():
        N = k + 2 * l + 1
        if N <= 12:
            lv.append((z, N, N - l, l))
    lv.sort()
    name = lambda N, n, l: f"{n}{LET[l]}"
    res = {"label": label}
    order = [(N, n, l, z) for z, N, n, l in lv]
    # first inversion: first element in z-order whose (N,n) key is smaller than a previously listed one
    inv = None
    for i in range(len(order)):
        for j in range(i):
            if (order[j][0], order[j][1]) > (order[i][0], order[i][1]):
                inv = (name(*order[i][:3]), order[i][3], name(*order[j][:3]), order[j][3]); break
        if inv: break
    res["first_inversion"] = inv
    inv10 = [(name(*o[:3]), o[3]) for i, o in enumerate(order) if o[0] <= 10 and any((p[0], p[1]) > (o[0], o[1]) and p[0] <= 10 for p in order[:i])]
    res["madelung_N<=10"] = (len(inv10) == 0)
    res["inversions_N<=10"] = inv10
    zz = {name(N, n, l): z for N, n, l, z in order}
    res["Z"] = {k: zz.get(k) for k in ("7h", "11s", "8g", "10s", "6h")}
    res["order_7h_11s_8g"] = bool(None not in (zz.get("7h"), zz.get("11s"), zz.get("8g")) and zz["7h"] < zz["11s"] < zz["8g"])
    return res, order


T0 = time.time()
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        # engine check against the paper's thresholds: pure TF, eps -> 0-: crossing G of 7h (k=1,l=5) and 11s (k=10,l=0)
        for (l, k, Gp) in ((0, 0, 1.059264881120), (2, 0, 15.529479667), (5, 1, 80.9409), (0, 10, 81.595), (4, 3, 82.389)):
            gs = np.linspace(Gp * 0.97, Gp * 1.03, 61)
            cn = count(np.full(gs.shape, l), gs, 0 * gs, 0, np.full(gs.shape, -1e-9))
            i = np.nonzero(cn >= k + 1)[0][0]
            print(f"l={l} k={k}: first G with count>k: {gs[i-1]:.4f}..{gs[i]:.4f}   (paper {Gp})", flush=True)
        print("time", time.time() - T0)
    if mode == "aufbau":
        Zmax = int(sys.argv[2]) if len(sys.argv) > 2 else 330
        fams = [("tf", 0.0), ("fa0", 0.0), ("latter", 1.0), ("fa", 1.0)] + [("interp", k) for k in (0.1, 0.25, 0.5, 0.75)]
        out = []
        for field, kap in fams:
            first, E = aufbau(field, kap, Zmax)
            res, order = madelung_report(first, f"{field} kappa={kap}")
            res["order"] = [(f"{n}{LET[l]}", z) for N, n, l, z in order]
            if field in ("tf", "fa0"):      # D0: first integer Z with E < 0 (the paper's threshold definition, integer Z)
                f0 = {key: int(np.nonzero(E[key] < 0)[0][0]) + 1 for key in E if (E[key] < 0).any()}
                r0, o0 = madelung_report(f0, "D0 " + res["label"])
                res["D0"] = {k: v for k, v in r0.items() if k != "label"}
                res["D0"]["order"] = [(f"{n}{LET[l]}", z) for N, n, l, z in o0]
            # D3: orbital energies (scaled units) of the margin levels at selected Z
            res["D3_E"] = {nm: {str(Z): float(E[key][Z - 1]) for Z in (100, 200, 300, 310, 320, 330) if Z <= Zmax}
                           for nm, key in (("7h", (5, 1)), ("11s", (0, 10)), ("8g", (4, 3)), ("10s", (0, 9)), ("6h", (5, 0)))}
            out.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "order"}), flush=True)
            print("   order:", " ".join(f"{a}:{z}" for a, z in res["order"]), flush=True)
        json.dump(out, open(os.path.join(HERE, "tf_robustness_aufbau.json"), "w"), indent=1)
    if mode == "cross":
        # D2: Z at which E_{k,l}(Z) = EPS (scaled units) for every level of N <= 12, bisection in Z with the Sturm count
        EPS = float(sys.argv[2]) if len(sys.argv) > 2 else -2e-3
        fams = [("tf", 0.0), ("fa0", 0.0), ("latter", 1.0), ("fa", 1.0)] + [("interp", k) for k in (0.1, 0.25, 0.5, 0.75)]
        levels = [(l, k) for l in range(6) for k in range(0, 12 - 2 * l)]
        allres = []
        for field, kap in fams:
            lg = np.array([l for l, k in levels], float); kg = np.array([k for l, k in levels])
            lo = np.full(len(levels), 0.3); hi = np.full(len(levels), 600.0)
            def cnt(Z):
                A, c, m = fieldpar(Z, field, kap)
                return count(lg, A + 0 * Z, c + 0 * Z, m, np.full(Z.shape, EPS))
            valid = cnt(lo) < kg + 1       # not already bound at Z = 0.3
            for _ in range(40):
                mid = np.sqrt(lo * hi)
                up = cnt(mid) >= kg + 1
                hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
            Zc = np.where(valid, np.sqrt(lo * hi), np.nan)
            first = {(l, k): z for (l, k), z in zip(levels, Zc) if not np.isnan(z)}
            nb = sorted(f"{k + l + 1}{LET[l]}" for (l, k), z in zip(levels, Zc) if np.isnan(z))
            lv = sorted((z, k + 2 * l + 1, k + l + 1, l) for (l, k), z in first.items())
            name = lambda n, l: f"{n}{LET[l]}"
            def firstinv(maxN):
                o = [x for x in lv if x[1] <= maxN]
                for i in range(len(o)):
                    for j in range(i):
                        if (o[j][1], o[j][2]) > (o[i][1], o[i][2]):
                            return (name(o[i][2], o[i][3]), round(o[i][0], 2), name(o[j][2], o[j][3]), round(o[j][0], 2))
                return None
            zz = {name(n, l): float(z) for z, N, n, l in lv}
            r = dict(field=field, kappa=kap, eps=EPS, Z={k: (round(zz[k], 2) if k in zz else None) for k in ("7h", "11s", "8g", "10s", "6h")},
                     order_7h_11s_8g=bool(all(k in zz for k in ("7h", "11s", "8g")) and zz["7h"] < zz["11s"] < zz["8g"]),
                     madelung_N_le_10_on_valid=firstinv(10) is None, first_inv_N_le_10=firstinv(10), first_inv_N_le_12=firstinv(12),
                     n_valid=len(first), already_bound=nb)
            allres.append(r)
            print(json.dumps(r), flush=True)
        json.dump(allres, open(os.path.join(HERE, f"tf_robustness_cross_{abs(EPS):g}.json"), "w"), indent=1)
