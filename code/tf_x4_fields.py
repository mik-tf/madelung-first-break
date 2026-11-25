#!/usr/bin/env python3
"""X4 of paper 3: is N* s_inf common across three fields of the path family w(1+eps g_TF)?
Pass rule (fixed before any run, read from the paper, X4): for eps = 0.25, 0.5, 1 the products P(eps) = N*(eps) s_inf(eps)
are each within 15 percent of one common constant (taken as their mean). Kill: (max-min)/mean above 40 percent.
Fields, radial equation, nearest Tietz field and Numerov shooting are those of tf_path_check.py (sigma = ln(s)/2):
   y'' = (kap^2 - 2 Z W(sigma, eps)) y,  kap = 2l+1,  W = (1-eps) WT + eps WF.
s_inf(eps) = 2 Wmax / K0^2 - 1,  K0 = (1/pi) int sqrt(2 W) dsigma   (Bohr-Sommerfeld, same derivation as the paper;
for Tietz W = sech^2 gives 0).  N*(eps) = smallest N with  min_l Z_(N+1,l) < Z_(N,0)  (group N+1 starts below the top of group N),
Z_(N,l) = (N-2l)-th zero-energy coupling in sector l.  Sector 0: all roots from a sign-change scan of the Wronskian (k-th change is
n_r = k-1).  Other sectors: first 4 roots (n_r = 0..3) from a scan above the classical minimum kap^2/(2 Wmax).
Domain |sigma| <= min(12, 300/kap) (avoids overflow of the growing solutions).
Usage: python3 tf_x4_fields.py out.json h_inv NSEC0 LMAX eps [eps ...]
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_accurate
t0 = time.time()
_tf = tf_accurate.default()
phi = lambda s: _tf.phi(s)

def ints(a):
    t = lambda s: (1 + a * s) ** -2
    pp = quad(lambda s: s * phi(s) * t(s), 0, 12, epsabs=1e-14, epsrel=1e-13, limit=400)[0] \
        + quad(lambda s: s * phi(s) * t(s), 12, np.inf, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    tt = quad(lambda s: s * t(s) ** 2, 0, np.inf, epsabs=1e-14, epsrel=1e-13)[0]
    ff = quad(lambda s: s * phi(s) ** 2, 0, 12, epsabs=1e-14, epsrel=1e-13, limit=400)[0] \
        + quad(lambda s: s * phi(s) ** 2, 12, np.inf, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    A = pp / tt
    return ff - 2 * A * pp + A * A * tt, A

def setup(hinv, XD=12.0):
    res = minimize(lambda p: ints(p[0])[0], [0.5367], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-18))
    a = float(res.x[0]); A = ints(a)[1]
    h = 1.0 / hinv; M = int(round(XD / h))
    sig = np.arange(-M, M + 1) * h
    s = np.exp(2 * sig)
    WT = 4 * a * s / (1 + a * s) ** 2
    WF = (4 * a / A) * s * np.array([phi(v) for v in s])
    return h, M, sig, WT, WF, a, A

def D(Z, kap, W, h):
    """normalised Wronskian at the centre, W already sliced symmetric (length 2M+1)."""
    Z = np.atleast_1d(np.asarray(Z, float))
    n = len(W); M = (n - 1) // 2; kap2 = kap * kap; h2 = h * h / 12.0
    def sweep(order):
        y0 = np.ones_like(Z); y1 = np.full_like(Z, math.exp(kap * h))
        ys = {order[0]: y0, order[1]: y1}
        qm = kap2 - 2 * Z * W[order[0]]; qn = kap2 - 2 * Z * W[order[1]]
        ym, yn = y0, y1
        for k in range(1, len(order) - 1):
            qp = kap2 - 2 * Z * W[order[k + 1]]
            yp = (2 * (1 + 5 * h2 * qn) * yn - (1 - h2 * qm) * ym) / (1 - h2 * qp)
            ym, yn, qm, qn = yn, yp, qn, qp
            if order[k + 1] >= M - 2 and order[k + 1] <= M + 2: ys[order[k + 1]] = yp
        return ys
    left = sweep(list(range(0, M + 3)))
    right = sweep(list(range(n - 1, M - 3, -1)))
    dy = lambda ys: (-ys[M + 2] + 8 * ys[M + 1] - 8 * ys[M - 1] + ys[M - 2]) / (12 * h)
    nl = np.sqrt(sum(left[j] ** 2 for j in range(M - 2, M + 3)))
    nr = np.sqrt(sum(right[j] ** 2 for j in range(M - 2, M + 3)))
    return (dy(left) * right[M] - dy(right) * left[M]) / (nl * nr)

def scan_roots(kap, W, h, zlo, zhi, step, kmax):
    zs = np.arange(zlo, zhi, step)
    out = []
    for i in range(0, len(zs), 4000):
        out.append(D(zs[i:i + 4000], kap, W, h))
    d = np.concatenate(out); sg = np.sign(d)
    idx = np.nonzero(sg[1:] != sg[:-1])[0][:kmax]
    lo = zs[idx].copy(); hi = zs[idx + 1].copy(); flo = sg[idx].copy()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = np.sign(D(mid, kap, W, h))
        same = fm == flo
        lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
        if np.max(hi - lo) < 1e-11 * np.max(hi): break
    return list(0.5 * (lo + hi))

def field(eps, M, sig, WT, WF, h):
    W = (1 - eps) * WT + eps * WF
    K0 = quad(lambda x: 0, 0, 1)[0]
    # K0 = (1/pi) int sqrt(2W) dsigma by trapezoid on the full grid (W decays exponentially, domain 12)
    K0 = float(np.sum(np.sqrt(2 * W)) * h / math.pi)
    Wmax = float(W.max())
    return W, K0, Wmax, 2 * Wmax / K0 ** 2 - 1

def run(out, hinv, nsec0, lmax, epss):
    h, M, sig, WT, WF, a, A = setup(hinv)
    print(f"nearest Tietz a={a:.7f} A={A:.7f}; h=1/{hinv}; setup {time.time()-t0:.0f}s", flush=True)
    res = {}
    for eps in epss:
        W, K0, Wmax, sinf = field(eps, M, sig, WT, WF, h)
        print(f"eps={eps}: K0={K0:.8f} Wmax={Wmax:.8f} s_inf={sinf:.8f}", flush=True)
        # sector 0: scan from 0 to 1.25 (nsec0+0.5)^2 / K0^2
        zmax = 1.25 * (nsec0 + 0.5) ** 2 / K0 ** 2
        Z = {}
        r0 = scan_roots(1, W, h, 0.05, zmax, 0.25, nsec0)
        for k, z in enumerate(r0): Z[(k + 1, 0)] = z
        print(f"  sector 0: {len(r0)} roots ({time.time()-t0:.0f}s)", flush=True)
        for l in range(1, lmax + 1):
            kap = 2 * l + 1; X = min(12.0, 300.0 / kap); m = int(round(X / h)); m = min(m, M)
            Wl = W[M - m:M + m + 1]
            z0 = kap * kap / (2 * Wmax)
            rs = scan_roots(kap, Wl, h, 0.95 * z0, (1.05 + 12.0 / kap) * z0, 0.0005 * z0, 4)
            for j, z in enumerate(rs): Z[(2 * l + 1 + j, l)] = z
            print(f"  l={l}: {len(rs)} roots ({time.time()-t0:.0f}s)", flush=True)
        res[str(eps)] = dict(K0=K0, Wmax=Wmax, sinf=sinf, Z={f"{N},{l}": z for (N, l), z in Z.items()})
        json.dump(dict(hinv=hinv, a=a, A=A, res=res), open(out, "w"))
    return res

def analyse(res):
    summ = {}
    for e, r in res.items():
        Z = {tuple(int(v) for v in k.split(",")): z for k, z in r["Z"].items()}
        Ns = sorted(N for (N, l) in Z if l == 0)
        found = None; rows = []
        for N in Ns[:-1]:
            cand = {l: Z[(N + 1, l)] for l in range(0, (N) // 2 + 1) if (N + 1, l) in Z}
            if not cand: continue
            lbest = min(cand, key=cand.get)
            margin = cand[lbest] / Z[(N, 0)] - 1
            rows.append((N, margin, lbest, (N) // 2))
            if margin < 0 and found is None: found = N
        summ[e] = (found, r["sinf"], rows)
    return summ

if __name__ == "__main__":
    if sys.argv[1] == "analyse":
        for fn in sys.argv[2:]:
            d = json.load(open(fn)); s = analyse(d["res"])
            print(fn, "h=1/%d" % d["hinv"])
            for e, (N, sinf, rows) in s.items():
                print(f"  eps={e}: N*={N} s_inf={sinf:.6f} N* s_inf={"none" if N is None else format(N*sinf, ".4f")}")
                if N:
                    for (n, mg, lb, lt) in rows:
                        if abs(n - N) <= 2: print(f"     N={n}: Z_low(N+1)/Z_top(N)-1={mg:+.6f} lowest at l={lb} (top l={lt})")
    else:
        out = sys.argv[1]; hinv = int(sys.argv[2]); nsec0 = int(sys.argv[3]); lmax = int(sys.argv[4])
        run(out, hinv, nsec0, lmax, [float(v) for v in sys.argv[5:]])
        s = analyse(json.load(open(out))["res"])
        for e, (N, sinf, rows) in s.items():
            print(f"eps={e}: N*={N} s_inf={sinf:.6f} N* s_inf={"none" if N is None else format(N*sinf, ".4f")}")
