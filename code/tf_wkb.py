#!/usr/bin/env python3
"""paper 3: semiclassical threshold curve zeta(t), the limit s_inf, the 1/N law, the gap law,
and the link to the kappa of Bjerg-Fournais-Hearnshaw-Solovej (arXiv:2506.14227).  Light: about a minute.

Model (paper 3, eq. 2.1): u'' = [l(l+1)/s^2 - G phi(s)/s] u at energy zero, phi = TF function.
Langer form in x = ln s, u = sqrt(s) y:  y'' = [lam^2 - G f(x)] y,  f = s phi(s),  lam = l + 1/2.
Bohr-Sommerfeld at threshold:  sqrt(G) K(lam/sqrt(G)) = n_r + 1/2 = N + 1/2 - 2 lam,
  K(mu) = (1/pi) int (f - mu^2)_+^(1/2) dx.
Leading order, G = N^2 zeta(t), t = lam/N:  sqrt(zeta) K(t/sqrt(zeta)) = 1 - 2t.
  zeta(0) = 1/K(0)^2, zeta(1/2) = 1/(4 f_max), s_inf = 4 f_max / K(0)^2 - 1.
1/N law (derived in the paper):  s_N = s_inf + c_odd/N (N odd), s_inf + c_even/N (N even), + O(N^-2 log N),
  c_odd = (1+s_inf)(1/2 - beta), c_even = (1+s_inf)(5/2 - 3 beta), beta = sqrt(2 a / f_max), a = -f''(x_max).
BFHS25: kappa(lam) = (1/pi) int (Phi_1(r) - lam^2/r^2)_+^(1/2) dr = c K(lam/c), c = (3 pi/4)^(1/3) = sqrt(2 b).

Usage:  python3 tf_wkb.py     (reads tf_threshold_table_N30.csv, prints every number quoted in the paper)
"""
import os, sys, math, csv
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tf_accurate as TA

tf = TA.default()
phi = lambda s: float(tf.phi(s))
dphi = lambda s: float(tf.dphi(s))

# ---- spline of g(x) = ln f(x), f = s phi(s), on a wide grid (phi beyond 200 is tf_accurate's Coulson-March form)
XS = np.linspace(math.log(1e-9), math.log(1e7), 12001)
GX = np.array([x + math.log(phi(math.exp(x))) for x in XS])
SP = CubicSpline(XS, GX)

# ---- exact constants (direct phi evaluations)
sm = brentq(lambda s: phi(s) + s * dphi(s), 0.5, 20.0, xtol=1e-15)          # max of s phi
fmax = sm * phi(sm)
d2 = sm ** 2 * (2 * dphi(sm) + sm * phi(sm) ** 1.5 / math.sqrt(sm))          # d^2 f / dx^2 at the max
a = -d2
beta = math.sqrt(2 * a / fmax)
# K(0) = (1/pi) int_0^inf sqrt(phi/s) ds = (1/pi) int 2 sqrt(phi(u^2)) du  (s = u^2), split for accuracy
k0a = quad(lambda u: 2 * math.sqrt(phi(u * u)), 0, 10, epsabs=1e-13, epsrel=1e-13, limit=400)[0]
k0b = quad(lambda x: math.sqrt(math.exp(x) * phi(math.exp(x))), math.log(100), math.log(1e7), epsabs=1e-13, epsrel=1e-13, limit=400)[0]
k0b += 12.0 / 1e7                                                             # tail beyond 1e7 (phi ~ 144/s^3)
K0 = (k0a + k0b) / math.pi
s_inf = 4 * fmax / K0 ** 2 - 1
c = (3 * math.pi / 4) ** (1 / 3)
norm = quad(lambda u: 2 * u * u * phi(u * u) ** 1.5, 0, 10, limit=400)[0] \
     + quad(lambda x: math.exp(1.5 * x) * phi(math.exp(x)) ** 1.5, math.log(100), math.log(1e7), limit=400)[0]
print(f"s_max={sm:.12f} f_max={fmax:.12f} a=-f''(x_max)={a:.12f} beta={beta:.10f}")
print(f"K(0)={K0:.12f}  zeta(0)={1/K0**2:.12f}  zeta(1/2)={1/(4*fmax):.12f}  s_inf={s_inf:.10f}")
print(f"int s^(1/2) phi^(3/2) ds = {norm:.10f} (exact 1) -> int 4 lam kappa = 4 c^3/(3 pi) * that = {4*c**3/(3*math.pi)*norm:.10f}")
print(f"kappa(0)=c K(0)={c*K0:.10f}, kappa edge lam_c=c sqrt(f_max)={c*math.sqrt(fmax):.10f}, Madelung 6^(1/3)={6**(1/3):.10f}, edge {6**(1/3)/2:.10f}")
print(f"check (2 lam_c/kappa(0))^2 - 1 = {(2*c*math.sqrt(fmax)/(c*K0))**2-1:.10f}")
c_odd = (1 + s_inf) * (0.5 - beta)
c_even = (1 + s_inf) * (2.5 - 3 * beta)
print(f"predicted 1/N coefficients: c_odd={c_odd:.6f} c_even={c_even:.6f}  A_N*N -> {c_odd-c_even:.6f}")
print(f"predicted start-to-start gap: odd N: 2 beta/N = {2*beta:.6f}/N, even N: (4-2 beta)/N = {4-2*beta:.6f}/N")

XM = minimize_scalar(lambda x: -SP(x), bracket=(XS[np.argmax(GX)] - 0.01, XS[np.argmax(GX)], XS[np.argmax(GX)] + 0.01)).x
GXM = float(SP(XM))


def K(mu):
    """(1/pi) int (f - mu^2)_+^(1/2) dx via the spline (mu > 0)."""
    lk = 2 * math.log(mu)
    if lk >= GXM:
        return 0.0
    xa = brentq(lambda x: SP(x) - lk, XS[0], XM, xtol=1e-14)
    xb = brentq(lambda x: SP(x) - lk, XM, XS[-1], xtol=1e-14)
    return quad(lambda x: math.sqrt(max(math.exp(SP(x)) - mu * mu, 0.0)), xa, xb, epsabs=1e-13, epsrel=1e-12, limit=400)[0] / math.pi


# slope of K at 0: predicted -(1/p_core + 1/p_tail) = -(1 + 1/2) = -3/2
for mu in (1e-2, 3e-3, 1e-3):
    print(f"  (K(mu)-K(0))/mu at mu={mu:g}: {(K(mu)-K0)/mu:.5f}  (predicted -1.5)")
# edge: K(mu) ~ (f_max - mu^2)/sqrt(2a)  (linear edge, not square root)
for e in (1e-2, 1e-3):
    mu = math.sqrt(fmax * (1 - e))
    print(f"  edge: K / ((f_max-mu^2)/sqrt(2a)) at 1-mu^2/f_max={e:g}: {K(mu)/((fmax-mu*mu)/math.sqrt(2*a)):.5f}")


def zeta(t):
    """leading-order threshold curve: sqrt(z) K(t/sqrt(z)) = 1 - 2t."""
    if t <= 0:
        return 1 / K0 ** 2
    if t >= 0.5:
        return 1 / (4 * fmax)
    zmin = t * t / fmax * (1 + 1e-12)
    return brentq(lambda z: math.sqrt(z) * K(t / math.sqrt(z)) - (1 - 2 * t), zmin, 4 / K0 ** 2, xtol=1e-15)


print("zeta(t), t = 0..0.5:", " ".join(f"{t:.2f}:{zeta(t):.6f}" for t in np.linspace(0, 0.5, 11)))


def G_bs(N, l):
    """full Bohr-Sommerfeld threshold (Langer, Maslov 1/2), as in tf_threshold_table_N30.wkb."""
    lam = l + 0.5
    nr = N - 2 * l - 1
    F = lambda Gv: math.sqrt(Gv) * K(lam / math.sqrt(Gv)) - (nr + 0.5)
    gmin = lam * lam / fmax * (1 + 1e-12)
    return brentq(F, gmin, 4 * (N + 1) ** 2 / K0 ** 2, xtol=1e-12)


def lmax(N): return (N - 1) // 2


# ---- data
G = {}
with open(os.path.join(HERE, "tf_threshold_table_N30.csv")) as fh:
    for r in csv.DictReader(fh):
        G[(int(r["N"]), int(r["l"]))] = float(r["G_final_S1e6"])
NM = max(N for N, l in G)
s = {N: G[(N, 0)] / G[(N, lmax(N))] - 1 for N in range(3, NM + 1)}
gam = {N: G[(N + 1, lmax(N + 1))] / G[(N, lmax(N))] - 1 for N in range(3, NM)}   # start-to-start gap

# ---- WKB sequence (same quantisation), N up to 30 against data, and further out to show the limit
sw, gw = {}, {}
GW = {}
for N in list(range(3, 31)) + [40, 41, 60, 61, 100, 101, 200, 201, 400, 401]:
    GW[(N, 0)] = G_bs(N, 0)
    GW[(N, lmax(N))] = G_bs(N, lmax(N))
    sw[N] = GW[(N, 0)] / GW[(N, lmax(N))] - 1
for N in range(3, 30):
    gw[N] = GW[(N + 1, lmax(N + 1))] / GW[(N, lmax(N))] - 1

print("\nN  s_data  s_wkb  s_pred(1/N)  gap_data  gap_wkb  2/N   overlap(data: s>gap)")
for N in range(3, NM + 1):
    cN = c_odd if N % 2 else c_even
    pr = s_inf + cN / N
    gd = gam.get(N, float('nan')); gwN = gw.get(N, float('nan'))
    print(f"{N:2d} {s[N]:.5f} {sw[N]:.5f} {pr:.5f}  {gd:+.5f} {gwN:+.5f} {2/N:.5f}  {s[N] > gd if N in gam else ''}")
print("WKB s_N far out:", " ".join(f"{N}:{sw[N]:.5f}" for N in (40, 41, 60, 61, 100, 101, 200, 201, 400, 401)))
print("WKB N*(s_N > gap_N, first):", min(N for N in gw if sw[N] > gw[N]), " data N*:", min(N for N in gam if s[N] > gam[N]))

# ---- T7: Richardson (fixed before the comparison): least squares s_inf + c1/N + c2/N^2 per parity, 10 <= N <= 30
def fit(Ns, vals, k=2, logterm=False):
    cols = [np.ones(len(Ns))] + [np.array(Ns, float) ** -j for j in range(1, k + 1)]
    if logterm:
        cols.append(np.log(Ns) / np.array(Ns, float) ** 2)
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
    res = np.array(vals) - A @ coef
    return coef, res

rows = []
for name, k, lo, lg in (("primary: 1/N,1/N^2, N>=10", 2, 10, False),
                        ("1/N,1/N^2, N>=15", 2, 15, False),
                        ("1/N,1/N^2,1/N^3, N>=10", 3, 10, False),
                        ("1/N,1/N^2,logN/N^2, N>=10", 2, 10, True),
                        ("1/N only, N>=20", 1, 20, False)):
    out = []
    for par in (1, 0):
        Ns = [N for N in range(lo, NM + 1) if N % 2 == par]
        coef, res = fit(Ns, [s[N] for N in Ns], k, lg)
        out.append((coef, np.max(np.abs(res)), np.sqrt(np.mean(res ** 2))))
    sinf = 0.5 * (out[0][0][0] + out[1][0][0])
    rows.append((name, out[0][0][0], out[1][0][0], sinf, out[0][0][1], out[1][0][1], out[0][1], out[1][1]))
    print(f"{name:30s} s_inf odd={out[0][0][0]:.5f} even={out[1][0][0]:.5f} mean={sinf:.5f}  c1 odd={out[0][0][1]:+.4f} even={out[1][0][1]:+.4f}  max|res| odd={out[0][1]:.1e} even={out[1][1]:.1e}")
# classical three-point Richardson in 1/N on N, N/2-ish nodes: odd 7,15,29... use (10,20,30) even and (9,19,29) odd
def rich3(n1, n2, n3):
    Ns = [n1, n2, n3]
    A = np.vstack([np.ones(3), 1 / np.array(Ns, float), 1 / np.array(Ns, float) ** 2]).T
    return np.linalg.solve(A, [s[n] for n in Ns])[0]
r_e, r_o = rich3(10, 20, 30), rich3(9, 19, 29)
print(f"three-point Richardson: even (10,20,30) {r_e:.5f}, odd (9,19,29) {r_o:.5f}, mean {(r_e+r_o)/2:.5f}")
prim = rows[0][3]
print(f"T7: s_inf(WKB)={s_inf:.5f} vs primary Richardson {prim:.5f}: ratio-1 = {s_inf/prim-1:+.4f} -> {'PASS' if abs(s_inf/prim-1) <= 0.05 else 'FAIL'}")

# ---- residuals of the zero-parameter 1/N law and of a two-parameter fit with fixed s_inf
print("\nzero-parameter law s_inf + c/N: residual data - law (N, odd/even):")
print(" ".join(f"{N}:{s[N]-(s_inf+(c_odd if N%2 else c_even)/N):+.4f}" for N in range(10, NM + 1)))
for par in (1, 0):
    Ns = [N for N in range(10, NM + 1) if N % 2 == par]
    y = np.array([s[N] - s_inf for N in Ns])
    A = np.vstack([1 / np.array(Ns, float), 1 / np.array(Ns, float) ** 2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    print(f"fixed s_inf, fit c1/N + c2/N^2, {'odd' if par else 'even'}: c1={coef[0]:+.4f} (pred {c_odd if par else c_even:+.4f}) c2={coef[1]:+.3f} max|res|={np.max(np.abs(res)):.1e}")
# gap law residuals
print("gap data - 2/N, and data - parity law (2beta/N odd, (4-2beta)/N even):")
print(" ".join(f"{N}:{gam[N]-2/N:+.4f}/{gam[N]-((2*beta) if N%2 else (4-2*beta))/N:+.4f}" for N in range(10, NM)))
# alternation amplitude
print("A_N*N (pair p,p+1, N=p+1/2):", " ".join(f"{p},{p+1}:{(s[p]-s[p+1])*(p+0.5):.4f}" for p in range(9, NM, 2)), f" limit {c_odd-c_even:.4f}")
for p in (40, 60, 100, 200, 400):
    # pairs (p+1? ) use (p, p+1) with p even -> even-odd; report odd-even with odd p+1 and even p
    print(f"  WKB s_{p+1}-s_{p} times N: {(sw[p+1]-sw[p])*(p+0.5):.4f}")
