#!/usr/bin/env python3
"""Paper 3: interval certificate that the Thomas-Fermi zero-energy thresholds G(n,l)
come in Madelung order for N = n + l <= NMAX (method and lemmas in the paper).


Setting: u'' = [L/s^2 - G phi(s)/s] u, L = l(l+1), regular at 0.  Z(G) = number of zeros of u on (0, inf),
non-decreasing in G.  G(n,l) = inf{G : Z(G) >= n - l}.
Bracket [G_lo, G_hi] of G(n,l):
  G_hi: Z(G_hi) >= n - l, shown by >= n - l certified sign changes of u inside the validated range, for the
        problem with the LOWER potential phi_lo (phi_lo < phi_TF, Sturm: phi_lo has fewer zeros).
  G_lo: Z(G_lo) <= n - l - 1, shown by (i) the problem with the UPPER potential phi_hi has exactly k = n-l-1
        zeros on (0,S] and a Riccati lower bound w_hi(S) >= g, (ii) the problem with phi_lo has >= k zeros on
        (0,S], hence the true Pruefer angle lies in (k pi, ...) at S with cot >= cot(angle of phi_hi), and
        (iii) the tail lemma: for s >= S, phi <= 144/s^3 and w >= a/s is a Riccati subsolution, so no zero after S.
All arithmetic is mpmath.iv (directed rounding).  Initial data at s0 come from two entire Coulomb-type series
(comparison problems), with an explicit tail bound.
Run: nice -n 19 env OMP_NUM_THREADS=1 python3 tf_threshold_certificate.py [NMAX] [delta]
"""
import sys, os, time, csv
import mpmath as mp
from mpmath import iv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_certificate as TC
from tf_certificate import recur, peval, ivpow15, B_STAR, DELTA

iv.prec = 640
mp.mp.dps = 100
NT = 50
RATIO = 0.03
TMIN = 0.03          # relative stepping from t = 0.03 (s = 9e-4) on; the first step is [0, TMIN*RATIO]
T_START = 0.03       # trajectories start at the first step end >= T_START
T_S = 70             # tail matching radius S = T_S^2 = 4900
BETA_ABS = 1.589     # |phi'(0)| bound used in the Coulomb lower comparison (B_lo, B_hi in (-1.5881, -1.5880))


class Ambiguous(Exception):
    pass


def ivb(x):
    return iv.mpf([min(x.a, x.b), max(x.a, x.b)])


class Sol:
    """Validated TF solution y(t) = phi(t^2) for a fixed B; like tf_certificate.Sol, but with relative steps from
    TMIN on and the full box Taylor series stored per step (needed for the Schroedinger remainder)."""

    def __init__(self, B, mode, tmax=2000.0):
        self.steps = []
        y = iv.mpf(1); u = iv.mpf(2) * iv.mpf(B)
        t = iv.mpf(0)
        self.result = None
        while True:
            t0f = float(t.a)
            h = RATIO * max(t0f, TMIN)
            hi = iv.mpf(h)
            while True:
                T = iv.mpf([t.a, (t + hi).b])
                Yb = y; Ub = u; ok = False
                for _ in range(8):
                    if Yb.a <= 0:
                        break
                    Ub_n = u + iv.mpf([0, 1]) * hi * 4 * ivpow15(Yb)
                    Yb_n = y + iv.mpf([0, 1]) * hi * T * Ub_n
                    iy = iv.mpf([Yb_n.a - abs(Yb_n.a - Yb_n.b) * 0.5, Yb_n.b + abs(Yb_n.a - Yb_n.b) * 0.5])
                    iu = iv.mpf([Ub_n.a - abs(Ub_n.a - Ub_n.b) * 0.5, Ub_n.b + abs(Ub_n.a - Ub_n.b) * 0.5])
                    if iy.a <= 0:
                        break
                    Ub2 = u + iv.mpf([0, 1]) * hi * 4 * ivpow15(iy)
                    Yb2 = y + iv.mpf([0, 1]) * hi * T * iu
                    Yb, Ub = iy, iu
                    if iy.a <= Yb2.a and Yb2.b <= iy.b and iu.a <= Ub2.a and Ub2.b <= iu.b:
                        ok = True
                        break
                if ok:
                    break
                h = h / 2
                hi = iv.mpf(h)
                if h < 1e-12:
                    self.result = 'picard_fail'
                    return
            yc, uc = recur(y, u, t, NT)
            ybc, ubc = recur(Yb, Ub, T, NT)
            ry, ru = ybc[NT + 1], ubc[NT + 1]
            self.steps.append(dict(t0f=t0f, h=h, hi=hi, yc=yc[:NT + 1], uc=uc[:NT + 1], ry=ry, ru=ru, t=t, T=T,
                                   ybc=ybc, Yb=Yb))
            y = peval(yc[:NT + 1], hi) + ry * hi ** (NT + 1)
            u = peval(uc[:NT + 1], hi) + ru * hi ** (NT + 1)
            t = t + hi
            if y.a <= 0:
                self.result = 'nonpositive'
                return
            lab = mode(y, u, t)
            if lab:
                self.result = lab
                self.tend = float(t.a)
                return
            if float(t.a) > tmax:
                self.result = 'tmax'
                return


def make_sandwich():
    Blo = B_STAR - DELTA; Bhi = B_STAR + DELTA

    def mode_lo(y, u, t):
        if u.b < 0:
            lhs = (u.b ** 2) / 4
            rhs = 2 * y * y * iv.sqrt(y) / t
            if (lhs - rhs).a > 0:
                return 'hits_zero'
        return None

    def mode_hi(y, u, t):
        return 'turns_up' if u.a > 0 else None

    return Sol(Blo, mode_lo), Sol(Bhi, mode_hi)


# ------------------------------------------------------------------ initial data from Coulomb comparison series
def coulomb_w(l, G, c, s0):
    """w = u'/u at s0 (interval, rigorous) for the regular solution of u'' = (L/s^2 - G/s + c) u, c >= 0,
    u = s^(l+1) sum a_j s^j, j(j+2l+1) a_j = -G a_(j-1) + c a_(j-2).
    Tail lemma.  Let m_j >= 0 solve the same recursion with +G, +c, M_j = m_j s0^j.  Then
    M_j <= (G s0/j) M_(j-1) + (c s0^2/j) M_(j-2) (as j + 2l + 1 >= j), so M_j <= ((G s0 + c s0^2)/j) max(M_(j-1), M_(j-2)),
    and for j > J with J >= 2 (G s0 + c s0^2) + 2 (the assert) the factor is at most 1/2 against the larger of the
    two previous terms.  With R = max(M_J, M_(J-1)): M_j <= R/2^(ceil((j-J)/2))-type decay, so
    sum_(j>J) M_j <= 2R and sum_(j>J) (l+1+j) M_j <= R (2 (l+J) + 9) <= 4 R (l+J+3), which is the bound used below
    (tail = 4 (M_J + M_(J-1)) >= 4R, times (l+1+J+2)).
    Also asserted: sum_(j>=1) m_j s0^j + tail < 1, so the series factor 1 + sum a_j s0^j >= 1 - that sum > 0 for the
    comparison solutions (and hence the true one, by the Wronskian start argument) has no zero on (0, s0]."""
    J = 40
    s0 = iv.mpf(s0)
    G = iv.mpf(G); c = iv.mpf(c)
    assert (2 * (G * s0 + c * s0 * s0) + 2).b < J      # ratio test of the tail lemma
    a = [iv.mpf(1)]
    m = [iv.mpf(1)]
    for j in range(1, J + 1):
        am1 = a[j - 1]; am2 = a[j - 2] if j >= 2 else iv.mpf(0)
        mm1 = m[j - 1]; mm2 = m[j - 2] if j >= 2 else iv.mpf(0)
        d = j * (j + 2 * l + 1)
        a.append((-G * am1 + c * am2) / d)
        m.append((G * mm1 + c * mm2) / d)
    num = iv.mpf(0); den = iv.mpf(0)
    for j in range(J + 1):
        num += (l + 1 + j) * a[j] * s0 ** j
        den += a[j] * s0 ** j
    tail = 4 * (m[J] * s0 ** J + m[J - 1] * s0 ** (J - 1))     # bound on |sum_{j>J} m_j s0^j| (geometric, ratio <= 1/2)
    num += iv.mpf([-1, 1]) * tail * (l + 1 + J + 2)
    den += iv.mpf([-1, 1]) * tail
    assert den.a > 0
    msum = iv.mpf(0)
    for j in range(1, J + 1):
        msum += m[j] * s0 ** j
    assert (msum + tail).b < 1, 'comparison solution may have a zero on (0, s0]'
    return num / den / s0


# ------------------------------------------------------------------ Schroedinger trajectory along a TF sandwich
def series_t(T, n, kind):
    """coefficients of 1/t (kind 1) or 1/t^2 (kind 2) about the interval/point T, up to order n."""
    r = []
    for j in range(n + 1):
        sgn = -1 if j % 2 else 1
        if kind == 1:
            r.append(sgn / T ** (j + 1))
        else:
            r.append(sgn * (j + 1) / T ** (j + 2))
    return r


def ab_series(ycoef, tp, L, G, a0, b0, n):
    """Taylor coefficients (a_k, b_k), k <= n+1, of a' = b, b' = b/t + (4L/t^2 - 4G y) a about tp."""
    p = series_t(tp, n, 1)
    q = series_t(tp, n, 2)
    c = [4 * L * q[j] - 4 * G * ycoef[j] for j in range(n + 1)]
    a = [a0]; b = [b0]
    for k in range(n + 1):
        sb = iv.mpf(0)
        for j in range(k + 1):
            sb += p[j] * b[k - j] + c[j] * a[k - j]
        a.append(b[k] / (k + 1))
        b.append(sb / (k + 1))
    return a, b


def sgn(x):
    if x.a > 0:
        return 1
    if x.b < 0:
        return -1
    return 0


def trajectory(sol, l, G, w0, mode, target=None, tS=None):
    """Integrate u (a) and b = du/dt, a(t_start) = 1, du/ds = w0 u, along sol.  Counts certified zeros of u.
    mode 'count': stop when `target` zeros are counted; return (count, t_node_upper).
    mode 'at': integrate to t = tS exactly; return (count, a, b)."""
    L = l * (l + 1)
    Gi = iv.mpf(G)
    i0 = next(i for i, s in enumerate(sol.steps) if s['t0f'] >= T_START)
    st = sol.steps[i0]
    t = st['t']
    a = iv.mpf(1)
    b = 2 * t * w0
    count = 0
    sa = 1
    for st in sol.steps[i0:]:
        h = st['hi']; hf = st['h']
        last = False
        if mode == 'at':
            if st['t0f'] >= tS:
                break
            if st['t0f'] + hf >= tS:
                last = True
                hh = iv.mpf(tS) - st['t']
                assert hh.a > 0 and hh.b <= h.b
            else:
                hh = h
        else:
            hh = h
        # a priori box over [0, hh] (linear ODE, Gronwall type)
        Tb = st['T']
        nu = iv.mpf(max(abs(a.a), abs(a.b), abs(b.a), abs(b.b)))   # upper ends of |a|, |b| (mp values)
        c1 = 1 / Tb.a
        c2 = 4 * L / (Tb.a ** 2) + 4 * Gi * st['Yb'].b
        # upper end explicitly: Python max on intervals can silently keep 1 when the comparison is undecided
        m = iv.mpf(max(mp.mpf(1), mp.mpf((c1 + c2).b)))
        hb = iv.mpf(hh.b)
        # Gronwall: |x(tau)|_inf <= |x0|_inf e^(m tau) for x = (a, b), m a row-sum bound of the coefficient matrix
        rho = iv.exp(hb * m) - 1
        Ab = a + iv.mpf([-1, 1]) * rho * nu
        Bb = b + iv.mpf([-1, 1]) * rho * nu
        ac, bc = ab_series(st['yc'], st['t'], L, Gi, a, b, NT)
        abc, bbc = ab_series(st['ybc'], Tb, L, Gi, Ab, Bb, NT)
        ra, rb = abc[NT + 1], bbc[NT + 1]
        acf = ac[:NT + 1]; bcf = bc[:NT + 1]
        a_new = peval(acf, hh) + ra * hh ** (NT + 1)
        b_new = peval(bcf, hh) + rb * hh ** (NT + 1)
        hs = iv.mpf([0, 1]) * hh
        a_en = peval(acf, hs) + ra * hs ** (NT + 1)
        b_en = peval(bcf, hs) + rb * hs ** (NT + 1)
        # zero classification on the step
        if sgn(a_en) != 0:
            pass
        else:
            if sgn(b_en) != 0 and sgn(a) != 0 and sgn(a_new) != 0:
                if sgn(a) != sgn(a_new):
                    count += 1
                else:
                    raise Ambiguous('equal end signs, monotone, but enclosure straddles 0 at t=%g' % st['t0f'])
            else:
                raise Ambiguous('zero not isolated at t=%g' % st['t0f'])
        a, b = a_new, b_new
        if mode == 'count' and count >= target:
            return count, float((st['t'] + h).b)
        if last:
            return count, a, b
    if mode == 'count':
        return count, None
    raise RuntimeError('range exhausted')


# ------------------------------------------------------------------ one level
def certify_level(sol_lo, sol_hi, n, l, Gnum, delta, rep):
    k = n - l - 1
    out = dict(n=n, l=l, N=n + l, G_num=Gnum)
    i_lo = next(i for i, s in enumerate(sol_lo.steps) if s['t0f'] >= T_START)
    i_hi = next(i for i, s in enumerate(sol_hi.steps) if s['t0f'] >= T_START)
    t_lo = sol_lo.steps[i_lo]['t']; t_hi = sol_hi.steps[i_hi]['t']
    assert t_lo.a == t_hi.a and t_lo.b == t_hi.b, 'hi and lo trajectories must start at the same s0'
    s0 = iv.mpf(t_lo) ** 2
    # ---- upper bracket
    Ghi = mp.mpf(Gnum) * (1 + mp.mpf(delta))
    Gh = iv.mpf(Ghi)
    w_low_angle = coulomb_w(l, Gh, Gh * BETA_ABS, s0)       # lower comparison angle  <-> larger w
    wl = iv.mpf(w_low_angle.b)
    cnt = trajectory(sol_lo, l, Gh, wl, 'count', target=k + 1)
    out['Ghi'] = Ghi
    out['hi_ok'] = cnt[1] is not None
    out['node_t_upper'] = cnt[1]
    out['node_s_upper'] = None if cnt[1] is None else cnt[1] ** 2
    # ---- lower bracket
    Glo = mp.mpf(Gnum) * (1 - mp.mpf(delta))
    Gl = iv.mpf(Glo)
    w_up_angle = coulomb_w(l, Gl, 0, s0)                    # upper comparison angle <-> smaller w
    wu = iv.mpf(w_up_angle.a)
    wl2 = iv.mpf(coulomb_w(l, Gl, Gl * BETA_ABS, s0).b)
    c_hi, a_hi, b_hi = trajectory(sol_hi, l, Gl, wu, 'at', tS=T_S)
    c_lo, a_lo, b_lo = trajectory(sol_lo, l, Gl, wl2, 'at', tS=T_S)
    S = iv.mpf(T_S) ** 2
    ssign = (-1) ** k
    w_hi = b_hi / (2 * iv.mpf(T_S) * a_hi)                   # du/ds / u at S
    L = l * (l + 1)
    disc = 1 + 4 * L - 576 * Gl / (S * S)
    a_star = (1 - iv.sqrt(disc)) / 2                          # lower root of a^2 - a = L - 144 G / S^2
    g = iv.mpf(a_star.b) / S                                 # g = a*/s, a* rounded up (still between the roots)
    ok_hi_count = (c_hi == k) and sgn(a_hi * ssign) == 1
    ok_lo_count = (c_lo >= k)
    ok_w = (w_hi.a > g.b)
    out.update(Glo=Glo, c_hi=c_hi, c_lo=c_lo, w_hi_S=(float(w_hi.a), float(w_hi.b)), g=float(g.b),
               lo_ok=bool(ok_hi_count and ok_lo_count and ok_w))
    return out


def main():
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    delta = sys.argv[2] if len(sys.argv) > 2 else '1e-3'
    here = os.path.dirname(os.path.abspath(__file__))
    rows = [r for r in csv.DictReader(open(os.path.join(here, 'tf_threshold_table.csv'))) if int(r['N']) <= NMAX]
    order = [(int(r['N']), int(r['n'])) for r in rows]
    assert order == sorted(order), 'rows not in Madelung order (N, then n ascending)'
    want = [(N, N - l) for N in range(1, NMAX + 1) for l in range((N - 1) // 2, -1, -1)]
    assert order == sorted(want) and all(int(r['N']) == int(r['n']) + int(r['l']) for r in rows), 'rows incomplete'
    t00 = time.time()
    print('NMAX', NMAX, 'delta', delta, 'levels', len(rows), flush=True)
    sol_lo, sol_hi = make_sandwich()
    print('sandwich: lo', sol_lo.result, 'hi', sol_hi.result, 'steps', len(sol_lo.steps), len(sol_hi.steps),
          'time %.0fs' % (time.time() - t00), flush=True)
    assert sol_lo.result == 'hits_zero' and sol_hi.result == 'turns_up'
    # tail lemma hypothesis: phi_hi(S) <= 144 / S^3 at S = T_S^2 (t = T_S):  y_hi(t) <= 144 / t^6
    def y_at(sol, tt):
        st = next(s for s in sol.steps if s['t0f'] <= tt < s['t0f'] + s['h'])
        hh = iv.mpf(tt) - st['t']
        assert hh.a >= 0 and hh.b <= st['hi'].b, 'y_at: point outside the step'
        return peval(st['yc'], hh) + st['ry'] * hh ** (NT + 1)
    yh = y_at(sol_hi, T_S)
    psi = iv.mpf(144) / iv.mpf(T_S) ** 6
    tail_ok = yh.b < psi.a
    print('tail hypothesis phi_hi(S) <= psi(S):', tail_ok, 'ratio phi_hi/psi in [%.6f, %.6f]' % (float((yh / psi).a), float((yh / psi).b)), flush=True)
    res = []
    for r in rows:
        t0 = time.time()
        n, l = int(r['n']), int(r['l'])
        try:
            o = certify_level(sol_lo, sol_hi, n, l, mp.mpf(r['G_final_S1e6']), delta, None)
        except Ambiguous as e:
            o = dict(n=n, l=l, N=n + l, G_num=mp.mpf(r['G_final_S1e6']), hi_ok=False, lo_ok=False, err=str(e))
        o['time'] = time.time() - t0
        res.append(o)
        print(o['N'], n, l, 'G_num %s' % mp.nstr(o['G_num'], 10), 'hi_ok', o.get('hi_ok'), 'lo_ok', o.get('lo_ok'),
              'node_s<=%s' % (None if o.get('node_s_upper') is None else '%.0f' % o['node_s_upper']),
              'c_hi', o.get('c_hi'), 'c_lo', o.get('c_lo'), 'w_hi', o.get('w_hi_S'), 'g', o.get('g'),
              o.get('err', ''), '%.0fs' % o['time'], flush=True)
    # verdict: Madelung order = (N, n) lexicographic = rows order of the csv; brackets disjoint and ordered
    allok = tail_ok and all(o['hi_ok'] and o['lo_ok'] for o in res)
    order_ok = allok
    if allok:
        for a, b in zip(res[:-1], res[1:]):
            if not (a['Ghi'] < b['Glo']):
                order_ok = False
                print('ORDER FAIL', a['n'], a['l'], b['n'], b['l'])
    print('brackets valid for all levels:', allok, ' disjoint and Madelung ordered:', order_ok)
    print('VERDICT', 'PASS' if order_ok else 'FAIL', 'N <= %d' % NMAX, 'total time %.0fs' % (time.time() - t00))


if __name__ == '__main__':
    main()
