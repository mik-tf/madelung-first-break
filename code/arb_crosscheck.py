#!/usr/bin/env python3
"""Paper 3, rigour upgrade (T5/W6): independent cross-check of the bracket-end sign certificates with python-flint (Arb).

Independent of the mpmath code (tf_certificate.py, tf_threshold_certificate.py are NOT imported): own Taylor method with
arb balls (validated remainders from the (N+1)-th Taylor coefficient on an a priori Picard box), own start data at
t0 = 1e-9 from the integral form of the regular solution, own zero counting.  Only the mathematical statements of
the threshold certificate are reused (sandwich criteria for B, tail lemma for s >= S), and the bracket centres
G_num from tf_threshold_table.csv.

Variables (as in the note): s = t^2, y(t) = phi_TF(t^2); y' = t u, u' = 4 y^(3/2), y(0) = 1, u(0) = 2 B.
Schroedinger u''(s) = [L/s^2 - G phi/s] u, a(t) = u(t^2):  a' = b, b' = b/t + (4L/t^2 - 4 G y) a, L = l(l+1).
B is carried as a ball [B* - 1e-35, B* + 1e-35]; first the sandwich is re-checked in Arb: B* - 1e-35 hits zero
(u < 0 and u^2/4 > 2 y^(5/2)/t) and B* + 1e-35 turns up (u > 0), so the true B lies inside the ball.
Per level (n, l), k = n - l - 1, delta = 1e-4 (bracket ends G_num (1 -+ delta)):
  upper end: a has at least k + 1 certified zeros on (0, t_node]   -> sign +
  lower end: exactly k certified zeros on (0, S], S = 70^2, sign of a(S) = (-1)^k, and
             w(S) = u'/u(S) > a*/S (tail lemma, a* the lower root of x^2 - x = L - 144 G/S^2), and
             y(70) < 144/70^6 (tail hypothesis)                      -> sign -
PASS iff every bracket end gets the sign (+ upper, - lower) that the mpmath certificate got (hi_ok, lo_ok in
tf_overlap_certificate_N12.log) and the zero counts agree.
Usage: python3 arb_crosscheck.py [levels "n:l,n:l,..."] [ratio=0.01]
"""
import sys, os, re, csv, time
from flint import arb, ctx

ctx.prec = 400
NT = 36
UNIT = arb(0.5, 0.5)          # the ball [0, 1]
T0 = arb("1e-9")
T_S = 70
DELTA = arb("1e-4")
B_STAR = "-1.588071022611375312718684509423950109452746621674825616765677"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = "7:5,11:0,8:4,8:5,7:6,10:1,2:1,3:2,5:3"     # 7h 11s 8g 8h 7i 10p + 2p 3d 5f


class Ambiguous(Exception):
    pass


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def subset(x, y):
    return (x.lower() > y.lower()) and (x.upper() < y.upper())


def inflate(x, f=0.5):
    lo = x.lower(); hi = x.upper(); w = hi - lo
    eps = x.abs_upper() * arb("1e-30") + arb("1e-300")
    return (lo - f * w - eps).union(hi + f * w + eps)


def peval(c, h):
    r = c[-1]
    for k in range(len(c) - 2, -1, -1):
        r = r * h + c[k]
    return r


def tf_coeffs(tp, y0, u0, n):
    """Taylor coefficients y_0..y_n, u_0..u_n about tp of y' = t u, u' = 4 y^(3/2) (Miller recurrence for y^(3/2))."""
    y = [y0]; u = [u0]; w = [y0 * y0.sqrt()]
    p = arb(3) / 2
    for k in range(n):
        y.append((tp * u[k] + (u[k - 1] if k > 0 else arb(0))) / (k + 1))
        u.append(4 * w[k] / (k + 1))
        m = k + 1
        s = arb(0)
        for j in range(1, m + 1):
            s += (p * j - (m - j)) * y[j] * w[m - j]
        w.append(s / (m * y[0]))
    return y, u


def tf_apriori(y, u, h, T):
    Yb, Ub = y, u
    for _ in range(14):
        if not (Yb > 0):
            raise Ambiguous("TF a priori box not positive")
        nY = y + UNIT * h * T * Ub
        nU = u + UNIT * h * 4 * Yb * Yb.sqrt()
        iY = inflate(nY); iU = inflate(nU)
        if not (iY > 0):
            raise Ambiguous("TF inflated box not positive")
        mY = y + UNIT * h * T * iU
        mU = u + UNIT * h * 4 * iY * iY.sqrt()
        if subset(mY, iY) and subset(mU, iU):
            return iY, iU
        Yb, Ub = iY, iU
    raise Ambiguous("TF a priori box did not close")


class TF:
    """Validated TF solution y(t), stored per step (point and box Taylor coefficients)."""

    def __init__(self, B, mode, ratio, tmax=500):
        self.steps = []
        y = arb(1); u = 2 * B; tc = arb(0); self.result = None
        while True:
            h = T0 if not self.steps else ratio * tc
            T = tc.union(tc + h)
            Yb, Ub = tf_apriori(y, u, h, T)
            yc, uc = tf_coeffs(tc, y, u, NT)
            ybc, ubc = tf_coeffs(T, Yb, Ub, NT + 1)
            yn = peval(yc, h) + ybc[NT + 1] * h ** (NT + 1)
            un = peval(uc, h) + ubc[NT + 1] * h ** (NT + 1)
            self.steps.append(dict(tc=tc, h=h, T=T, yc=yc, ybc=ybc[:NT + 1], Yb=Yb, rem=ybc[NT + 1]))
            y, u, tc = yn, un, tc + h
            if not (y > 0):
                self.result = "nonpositive"; return
            lab = mode(y, u, tc)
            if lab:
                self.result = lab; self.tend = float(tc.mid()); return
            if tc > tmax:
                self.result = "tmax"; return

    def y_at(self, t):
        for st in self.steps:
            if st["tc"] <= t and t <= st["tc"] + st["h"]:
                hh = arb(t) - st["tc"]
                return peval(st["yc"], hh) + st["rem"] * hh ** (NT + 1)
        raise RuntimeError("t outside range")


def mode_lo(y, u, t):
    if u.upper() < 0:
        lhs = u.upper() ** 2 / 4
        rhs = 2 * y * y * y.sqrt() / t
        if (lhs - rhs) > 0:
            return "hits_zero"
    return None


def mode_hi(y, u, t):
    return "turns_up" if u.lower() > 0 else None


def series_t(T, n, kind):
    r = []
    for j in range(n + 1):
        sg = -1 if j % 2 else 1
        r.append(sg / T ** (j + 1) if kind == 1 else sg * (j + 1) / T ** (j + 2))
    return r


def ab_series(ycoef, tp, L, G, a0, b0, n):
    p = series_t(tp, n, 1); q = series_t(tp, n, 2)
    c = [4 * L * q[j] - 4 * G * ycoef[j] for j in range(n + 1)]
    a = [a0]; b = [b0]
    for k in range(n + 1):
        sb = arb(0)
        for j in range(k + 1):
            sb += p[j] * b[k - j] + c[j] * a[k - j]
        a.append(b[k] / (k + 1)); b.append(sb / (k + 1))
    return a, b


def lin_apriori(a, b, hh, T, Yb, L, G):
    cf = 4 * L / (T * T) - 4 * G * Yb
    Ab, Bb = a, b
    for _ in range(25):
        nA = a + UNIT * hh * Bb
        nB = b + UNIT * hh * (Bb / T + cf * Ab)
        iA = inflate(nA); iB = inflate(nB)
        mA = a + UNIT * hh * iB
        mB = b + UNIT * hh * (iB / T + cf * iA)
        if subset(mA, iA) and subset(mB, iB):
            return iA, iB
        Ab, Bb = iA, iB
    raise Ambiguous("linear a priori box did not close")


def start_data(tf, l, G):
    """a = 1, b/a at t0 = 1e-9 from the integral form: a = t^rho z, rho = 2l+2,
    z'' + (2 rho - 1)/t z' + 4 G y z = 0, z(0) = 1, z'(0) = 0, so z' = -4 G t^(1-2rho) int_0^t s^(2rho-1) y z ds."""
    st = tf.steps[0]
    t0 = st["h"]; Yb = st["Yb"]
    rho = 2 * l + 2
    ylo = Yb.lower(); yhi = Yb.upper()
    zlo = 1 - G * yhi * t0 ** 2 / rho          # z' >= -4 G yhi t/(2 rho) (z <= 1), integrated
    if not (zlo > 0):
        raise Ambiguous("start: zlo not positive")
    c = 4 * G * t0 / (2 * rho)
    xlo = c * ylo * zlo; xhi = c * yhi / zlo
    return arb(1), (rho / t0 - xlo.union(xhi))


def trajectory(tf, l, G, mode, target=None, tS=None):
    L = l * (l + 1)
    a, w0 = start_data(tf, l, G)
    b = w0 * a
    count = 0
    for st in tf.steps[1:]:
        h = st["h"]; tc = st["tc"]; T = st["T"]; last = False
        if mode == "at":
            if tc >= tS:
                break
            if tc + h >= tS:
                last = True; h = arb(tS) - tc
                T = tc.union(tc + h)
        Ab, Bb = lin_apriori(a, b, h, T, st["Yb"], L, G)
        ac, bc = ab_series(st["yc"], tc, L, G, a, b, NT)
        abc, bbc = ab_series(st["ybc"], st["T"], L, G, Ab, Bb, NT)
        an = peval(ac[:NT + 1], h) + abc[NT + 1] * h ** (NT + 1)
        bn = peval(bc[:NT + 1], h) + bbc[NT + 1] * h ** (NT + 1)
        hs = UNIT * h
        a_en = peval(ac[:NT + 1], hs) + abc[NT + 1] * hs ** (NT + 1)       # enclosure of a on the whole step
        b_en = peval(bc[:NT + 1], hs) + bbc[NT + 1] * hs ** (NT + 1)
        if sgn(a_en) == 0:
            if sgn(b_en) != 0 and sgn(a) != 0 and sgn(an) != 0:
                if sgn(a) != sgn(an):
                    count += 1
                else:
                    raise Ambiguous("equal end signs but enclosure straddles 0 at t=%g" % float(tc.mid()))
            else:
                raise Ambiguous("zero not isolated at t=%g" % float(tc.mid()))
        a, b = an, bn
        if mode == "count" and count >= target:
            return count, float((tc + h).mid()), a, b
        if last:
            return count, tS, a, b
    if mode == "count":
        return count, None, a, b
    raise RuntimeError("range exhausted")


def parse_mpmath_log():
    out = {}
    for ln in open(os.path.join(HERE, "tf_overlap_certificate_N12.log")):
        m = re.match(r"\s*(\d+) (\d+) (\d+) G_num (\S+) hi_ok (\w+) lo_ok (\w+).* c_hi (\d+) c_lo (\d+)", ln)
        if m:
            out[(int(m.group(2)), int(m.group(3)))] = (m.group(5) == "True", m.group(6) == "True",
                                                        int(m.group(7)), int(m.group(8)))
    return out


def build(ratio):
    Bc = arb(B_STAR); d = arb("1e-35").abs_upper()
    Bball = Bc + (-d).union(d)
    t0 = time.time()
    lo = TF(Bc - d, mode_lo, ratio); hi = TF(Bc + d, mode_hi, ratio)
    ok_sand = (lo.result == "hits_zero" and hi.result == "turns_up")
    print("sandwich (Arb): B*-1e-35 %s at t=%.1f, B*+1e-35 %s at t=%.1f  %s (%.0fs)" % (
        lo.result, getattr(lo, "tend", -1), hi.result, getattr(hi, "tend", -1), ok_sand, time.time() - t0), flush=True)
    tf = TF(Bball, lambda y, u, t: None, ratio, tmax=300)
    yS = tf.y_at(arb(T_S))
    psi = arb(144) / arb(T_S) ** 6
    tail_ok = bool(yS.upper() < psi)
    print("TF ball enclosure to t=%.0f (result %s); tail hypothesis y(70) < 144/70^6: %s  (y(70) %s, y/psi %s)  (%.0fs)" % (
        float(tf.steps[-1]["tc"].mid()), tf.result, tail_ok, yS.str(8), (yS / psi).str(8), time.time() - t0), flush=True)
    return tf, ok_sand and tail_ok


def level(tf, n, l, Gnum_str, expected):
    k = n - l - 1
    Gnum = arb(Gnum_str)
    Ghi = Gnum * (1 + DELTA); Glo = Gnum * (1 - DELTA)
    t0 = time.time()
    chi, tnode, _, _ = trajectory(tf, l, Ghi, "count", target=k + 1)
    sign_hi = +1 if tnode is not None else 0
    S = arb(T_S) ** 2
    clo, _, a, b = trajectory(tf, l, Glo, "at", tS=T_S)
    L = l * (l + 1)
    wS = b / (2 * arb(T_S) * a)
    disc = 1 + 4 * L - 576 * Glo / (S * S)
    astar = (1 - disc.sqrt()) / 2
    g = astar.upper() / S
    ok_cnt = (clo == k); ok_sign = (sgn(a) == (-1) ** k); ok_w = bool(wS.lower() > g.upper())
    sign_lo = -1 if (ok_cnt and ok_sign and ok_w) else 0
    exp_hi, exp_lo, e_chi, e_clo = expected
    agree = (sign_hi == +1) == exp_hi and (sign_lo == -1) == exp_lo and clo == e_clo
    print("level n=%d l=%d k=%d G_num %s | upper end: %d zeros reached at t<=%s sign %+d (mpmath hi_ok %s) | lower end: "
          "%d zeros (mpmath c_lo %d), sign(a_S) ok %s, w>g %s, sign %+d (mpmath lo_ok %s) | AGREE %s  %.0fs" % (
              n, l, k, Gnum_str[:12], k + 1, None if tnode is None else "%.1f" % tnode, sign_hi, exp_hi, clo, e_clo,
              ok_sign, ok_w, sign_lo, exp_lo, agree, time.time() - t0), flush=True)
    return agree and sign_hi == 1 and sign_lo == -1


def main():
    levels = [tuple(int(v) for v in x.split(":")) for x in (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).split(",")]
    ratio = arb(sys.argv[2]) if len(sys.argv) > 2 else arb("0.01")
    t00 = time.time()
    rows = {(int(r["n"]), int(r["l"])): r["G_final_S1e6"] for r in csv.DictReader(open(os.path.join(HERE, "tf_threshold_table.csv")))}
    exp = parse_mpmath_log()
    print("python-flint precision %d bits, Taylor order %d, ratio %s, delta 1e-4, levels %s" % (ctx.prec, NT, ratio.str(3), levels), flush=True)
    builds = {}
    def get(r):
        key = r.str(3)
        if key not in builds:
            builds[key] = build(r)
        return builds[key]
    tf, ok_base = get(ratio)
    res = []
    for (n, l) in levels:
        done = False
        for r in (ratio, ratio / 3, ratio / 10):       # finer relative step only where a zero cannot be isolated
            try:
                tfr, okr = get(r)
                if not okr:
                    raise Ambiguous("fine sandwich/tail invalid at ratio %s" % r.str(3))
                res.append(level(tfr, n, l, rows[(n, l)], exp[(n, l)]))
                done = True
                break
            except Ambiguous as e:
                print("level n=%d l=%d AMBIGUOUS at ratio %s: %s" % (n, l, r.str(3), e), flush=True)
        if not done:
            res.append(False)
    allok = ok_base and all(res)
    print("ARB CROSS-CHECK", "PASS" if allok else "FAIL", "levels %d/%d agree, sandwich+tail %s, total time %.0fs" % (
        sum(res), len(res), ok_base, time.time() - t00))


if __name__ == "__main__":
    main()
