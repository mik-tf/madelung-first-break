#!/usr/bin/env python3
"""TF threshold table (paper 3).

For u'' = [l(l+1)/s^2 - G phi(s)/s] u, regular start, find the thresholds G(n,l), N = n+l <= 16,
l <= floor((N-1)/2).  Method notes (what is actually done):
 * phi and its local Taylor series come from tf_accurate.default() (60 digit solution, double copies).
 * Start: Frobenius series in t = sqrt(s), 60 terms, evaluated at s = 0.25 (inside the t-series region).
 * From 0.25 out: own Taylor stepping of u (order 22) on the same step grid as tf_accurate (h = 0.1 s),
   vectorised over many G at once.  Half-step run: every step split in two (phi re-centred exactly).
 * At S match to a sqrt(s) J_nu(c/s) + b sqrt(s) Y_nu(c/s), nu = l+1/2, c = 12 sqrt(G) (phi = 144/s^3).
   Threshold: b = 0.  The total node count (nodes in (0,S) plus nodes of the tail solution beyond S)
   jumps by one exactly at b = 0, so levels are found by multisection on the integer count to 1e-13.
 * Double precision numpy/scipy (not mpmath 30 digits): the quantity of interest is 1e-12 relative.
 * Error estimate: rerun with half step and with S doubled, take the larger change per level.
REVISED METHOD (2026-11-25, matching radius changed from the plan's S = 60): the count/scan/refine machinery is
used only at S = 200 to bracket each level.  Thresholds are then found by integrating out to S up to 1e6 in x = ln s
(DOP853, rtol 1e-13) with tf_accurate's own Coulson-March phi (valid beyond 200), matching to the 144/s^3 Bessel tail
only at large S, on a ladder 1e3 .. 1e6; root of b(G) from a 17-node Chebyshev interpolant in a +-3e-3 bracket.
Usage: nice -n 19 env OMP_NUM_THREADS=1 python3 tf_threshold_table.py
"""
import os, sys, csv, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import jv, yv, jvp, yvp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_accurate as TA

K = 22
NMAX = 16
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
tf = TA.default()
A = np.array([float(v) for v in tf.a[:80]])
S0 = 0.01
XJ = 0.25


def make_steps(S, nsub):
    out = []
    for (xs, h, c) in tf._steps_f:
        if xs >= S:
            break
        c = np.array(c[:K + 1])
        # full-order polynomial for exact re-centring
        cf = np.array(tf._steps_f[0][2]) if False else None
        out.append((xs, h, np.array(_full(xs, h))))
    return out


_cache = {}
def _full(xs, h):
    for (x, hh, c) in tf._steps_f:
        if x == xs:
            return c
    raise KeyError


def shift(c, d):
    """coefficients of p(eta + d) given p(eta) = sum c_k eta^k (exact binomial shift, double)."""
    n = len(c)
    out = np.zeros(n)
    for k in range(n):
        s = 0.0
        for j in range(k, n):
            s += math.comb(j, k) * c[j] * d ** (j - k)
        out[k] = s
    return out


_low = None
def low_steps():
    """phi Taylor steps (double) from S0 up to XJ, built from the t-series in mpmath."""
    global _low
    if _low is None:
        _low = []
        x = TA.mp.mpf(S0)
        while x < XJ - 1e-15:
            y, dp, _ = TA.eval_t(tf.a, TA.mp.sqrt(x))
            h = min(TA.mp.mpf('0.1') * x, TA.mp.mpf(XJ) - x)
            cc = TA.taylor_step_coeffs(x, y, dp, 40)
            _low.append((float(x), float(h), [float(v) for v in cc]))
            x = x + h
    return _low


def step_list(S, nsub):
    key = (S, nsub)
    if key in _cache:
        return _cache[key]
    out = []
    for (xs, h, c) in low_steps() + list(tf._steps_f):
        if xs >= S:
            break
        c = np.array(c)
        hh = min(h, S - xs)
        if nsub == 1:
            out.append((xs, hh, c[:K + 1].copy()))
        else:
            sub = hh / nsub
            for i in range(nsub):
                d = i * sub
                cc = c if i == 0 else shift(c[:40], d)
                out.append((xs + d, sub, np.array(cc[:K + 1])))
    _cache[key] = out
    return out


def start(l, G):
    """u, u' at s = S0 from the Frobenius series; G array."""
    M = 60
    c = np.zeros((M + 1,) + G.shape)
    c[0] = 1.0
    for k in range(2, M + 1):
        br = (l + 1 + k / 2) * (l + k / 2) - l * (l + 1)
        acc = 0.0
        for j in range(0, k - 1):
            if A[j] != 0.0:
                acc = acc - G * A[j] * c[k - 2 - j]
        c[k] = acc / br
    t = math.sqrt(S0)
    u = 0.0 * G
    du = 0.0 * G
    for k in range(M + 1):
        u = u + c[k] * t ** k
        du = du + c[k] * (l + 1 + k / 2) * t ** k
    return S0 ** (l + 1) * u, S0 ** l * du


def integrate(l, G, S, nsub=1):
    u, du = start(l, G)
    nodes = np.zeros(G.shape, dtype=int)
    ll = l * (l + 1)
    for (xs, h, c) in step_list(S, nsub):
        q = np.zeros((K + 1,) + G.shape)
        q[0] = ll
        for j in range(K + 1):
            q[j] = q[j] - G * (xs * c[j] + (c[j - 1] if j > 0 else 0.0))
        w = [u, du]
        for m in range(0, K - 1):
            rhs = 0.0
            for j in range(m + 1):
                rhs = rhs + q[j] * w[m - j]
            lower = 2 * xs * (m + 1) * m * w[m + 1] + m * (m - 1) * w[m]
            w.append((rhs - lower) / (xs * xs * (m + 2) * (m + 1)))
        W = np.array(w)
        prev = np.sign(u)
        for i in range(1, 9):
            eta = h * i / 8
            v = 0.0
            for k in range(K - 1, -1, -1):
                v = v * eta + W[k]
            sg = np.sign(v)
            nodes += (sg != prev) & (sg != 0)
            prev = sg
        eta = h
        u = 0.0
        du = 0.0
        for k in range(K, -1, -1):
            if k <= K - 1:
                u = u * eta + W[k]
        for k in range(K - 1, 0, -1):
            du = du * eta + k * W[k]
    return u, du, nodes


def tail(l, G, S, u, du):
    """returns b/|scale|, tail node count, size of next asymptotic term"""
    nu = l + 0.5
    c = 12 * np.sqrt(G)
    x = c / S
    s = S
    J, Y = jv(nu, x), yv(nu, x)
    Jp, Yp = jvp(nu, x), yvp(nu, x)
    f1 = math.sqrt(s) * J
    f2 = math.sqrt(s) * Y
    f1p = J / (2 * math.sqrt(s)) + math.sqrt(s) * Jp * (-c / s ** 2)
    f2p = Y / (2 * math.sqrt(s)) + math.sqrt(s) * Yp * (-c / s ** 2)
    W12 = f1 * f2p - f1p * f2
    a = (u * f2p - du * f2) / W12
    b = (f1 * du - f1p * u) / W12
    xs = x[None, :] * (np.arange(1, 61)[:, None] / 60.0)
    g = a[None, :] * jv(nu, xs) + b[None, :] * yv(nu, xs)
    sg = np.sign(g)
    seq = np.concatenate([(-np.sign(b))[None, :], sg], axis=0)  # x -> 0 first, then up to x_S
    cnt = ((seq[1:] != seq[:-1]) & (seq[1:] != 0) & (seq[:-1] != 0)).sum(axis=0)
    return b, a, cnt


def count(l, G, S, nsub=1):
    G = np.atleast_1d(np.asarray(G, dtype=float))
    u, du, nodes = integrate(l, G, S, nsub)
    b, a, tc = tail(l, G, S, u, du)
    return nodes + tc


def bracket_scan(l, Gmax=80.0, step=0.1, S=60.0):
    Gmax = 280.0 if l == 0 else (220.0 if l < 3 else 160.0)
    Gs = np.arange(0.05, Gmax, step)
    return Gs, count(l, Gs, S)


def refine(l, nr, lo, hi, S, nsub, tol=1e-13):
    for _ in range(40):
        Gs = np.linspace(lo, hi, 33)
        cn = count(l, Gs, S, nsub)
        idx = np.nonzero(cn >= nr + 1)[0]
        if len(idx) == 0 or idx[0] == 0:
            return None
        lo, hi = Gs[idx[0] - 1], Gs[idx[0]]
        if (hi - lo) / hi < tol:
            break
    return 0.5 * (lo + hi)


def selfcheck():
    # tail solution check: u = sqrt(s) J_nu(c/s) solves u'' = [l(l+1)/s^2 - 144 G/s^4] u
    l, G, s, e = 2, 20.0, 40.0, 1e-3
    c = 12 * math.sqrt(G)
    f = lambda z: math.sqrt(z) * jv(l + .5, c / z)
    d2 = (f(s + e) - 2 * f(s) + f(s - e)) / e ** 2
    rhs = (l * (l + 1) / s ** 2 - 144 * G / s ** 4) * f(s)
    print(f"tail solution check: {d2:.6e} vs {rhs:.6e}")
    # eps=0-like check not available; check hydrogenic limit is out of scope


from scipy.integrate import solve_ivp

LAMF = float(TA.LAM)
CF = np.array(tf._c_f)          # Coulson-March series coefficients, w(z) = sum CF[k] z^k, z = s^-LAM, phi = 144 s^-3 w
S1 = 200.0                      # end of the Taylor stage (tf_accurate's own phi, join near 200)
LADDER = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
NCH = 17                        # Chebyshev nodes per level
W0 = 3e-3                       # half width of the bracket around the S=200 estimate (relative)
RTOL = 1e-13


def far(l, Gv, u0, du0, Ss, rtol=RTOL):
    """integrate from S1 out in x = ln s with the full asymptotic phi; return u, u' at each S in Ss (array over Gv)."""
    ll = l * (l + 1)
    ur = np.zeros((len(Ss), len(Gv)))
    dur = np.zeros_like(ur)
    for i, G in enumerate(Gv):
        def rhs(x, y):
            s = math.exp(x)
            z = s ** (-LAMF)
            w = np.polyval(CF[::-1], z)
            return [y[1], y[1] + (ll - 144.0 * G * w / (s * s)) * y[0]]
        sc = max(abs(u0[i]), abs(du0[i]) * S1)
        sol = solve_ivp(rhs, (math.log(S1), math.log(Ss[-1])), [u0[i], du0[i] * S1], method="DOP853",
                        rtol=rtol, atol=1e-18 * sc, t_eval=[math.log(v) for v in Ss])
        assert sol.success and sol.y.shape[1] == len(Ss)
        ur[:, i] = sol.y[0]
        dur[:, i] = sol.y[1] / np.array(Ss)
    return ur, dur


def bvals(l, Gv, Ss, nsub=1, rtol=RTOL):
    u0, du0, _ = integrate(l, Gv, S1, nsub)
    ur, dur = far(l, Gv, u0, du0, Ss, rtol)
    out = np.zeros_like(ur)
    for k, S in enumerate(Ss):
        b, a, _ = tail(l, Gv, S, ur[k], dur[k])
        # normalise by the size of the solution (a-part and b-part scales) so b is O(1)
        out[k] = b / np.sqrt(a * a + b * b + 1e-300) if False else b
    return out


def roots_from(Gv, bv, G0):
    """Chebyshev interpolant of b(G) on the node set; root nearest G0."""
    x = (Gv - G0) / (W0 * G0)
    sc = np.max(np.abs(bv))
    p = np.polynomial.chebyshev.Chebyshev.fit(x, bv / sc, len(x) - 1, domain=[-1, 1])
    r = p.roots()
    r = r[np.abs(r.imag) < 1e-12].real
    r = r[np.abs(r) <= 1.0]
    if len(r) == 0:
        return None
    return G0 * (1 + W0 * r[np.argmin(np.abs(r))])


def level_roots(l, G0, Ss, nsub=1, rtol=RTOL):
    k = np.arange(NCH)
    xn = np.cos(np.pi * (k + 0.5) / NCH)
    Gv = G0 * (1 + W0 * xn)
    B = bvals(l, Gv, Ss, nsub, rtol)
    return [roots_from(Gv, B[j], G0) for j in range(len(Ss))]


def main():
    selfcheck()
    levels = []
    for l in range(0, 8):
        Gs, cn = bracket_scan(l, S=S1)
        for N in range(2 * l + 1, NMAX + 1):
            nr = N - 2 * l - 1
            idx = np.nonzero(cn >= nr + 1)[0]
            if len(idx) == 0:
                raise SystemExit(f"scan did not reach l={l} nr={nr}")
            lo, hi = Gs[max(idx[0] - 1, 0)], Gs[idx[0]]
            G200 = refine(l, nr, lo, hi, S1, 1)
            rs = level_roots(l, G200, LADDER)
            if any(r is None for r in rs):
                raise SystemExit(f"no root in bracket l={l} N={N}")
            rh = level_roots(l, G200, [1e5, 1e6], nsub=2)
            rt = level_roots(l, G200, [1e5, 1e6], rtol=1e-11)
            Gf = rs[-1]
            ch = abs(rs[-1] - rs[-2]) / Gf
            e_step = max(abs(rh[1] - Gf), 0) / Gf
            e_tol = abs(rt[1] - Gf) / Gf
            levels.append(dict(N=N, n=N - l, l=l, nr=nr, G200=G200, ladder=rs, G=Gf, e_last=ch, e_step=e_step, e_tol=e_tol,
                               err=max(ch, e_step, e_tol)))
        print(f"l={l} done ({time.time() - t0:.0f}s)", flush=True)
    levels.sort(key=lambda d: (d["N"], d["n"]))
    with open(os.path.join(HERE, "tf_threshold_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "n", "l", "nr", "G_final_S1e6", "G_S200_count"] + [f"G_S{int(s)}" for s in LADDER] +
                   ["change_last_two_S", "change_half_step", "change_ode_tol", "err_estimate"])
        for d in levels:
            w.writerow([d["N"], d["n"], d["l"], d["nr"], f"{d['G']:.12e}", f"{d['G200']:.12e}"] + [f"{r:.12e}" for r in d["ladder"]] +
                       [f"{d['e_last']:.3e}", f"{d['e_step']:.3e}", f"{d['e_tol']:.3e}", f"{d['err']:.3e}"])
    print("csv written", flush=True)


if __name__ == "__main__":
    main()
