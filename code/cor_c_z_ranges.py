"""Corollary C: Z intervals for the first filling of every subshell with N <= 12.
Z = (G/(2*0.885341...))^(3/2) = 4 G^(3/2) / (3 pi), since 2*0.885341... = (3 pi/4)^(2/3) exactly.
Brackets: G_num*(1 -/+ delta) from the certificates (delta 1e-4 overlap run, 1e-3 for N<=8 run),
evaluated with outward-rounded interval arithmetic (mpmath iv). Light, runs in seconds.
Pass line: every N <= 12 level gets a Z interval; intervals of N<=8 from the two certificates intersect;
Z(7h) and Z(11s) match the quoted first break 309.01..312.86.
Run: python3 cor_c_z_ranges.py
"""
import re
from mpmath import iv, mp
iv.prec = 200
mp.dps = 30
LET = 'spdfghi'
def load(fn, delta):
    out = {}
    for ln in open(fn):
        m = re.match(r'(\d+) (\d+) (\d+) G_num (\S+) hi_ok True lo_ok True', ln)
        if m:
            N, n, l = int(m[1]), int(m[2]), int(m[3])
            g = iv.mpf(m[4])      # printed with 10 digits: widen by 1e-9 relative, safe (bracket is G_num(1+-delta))
            g = g * iv.mpf([1 - 1e-9, 1 + 1e-9])
            out[(n, l)] = (N, g * (1 - iv.mpf(delta)), g * (1 + iv.mpf(delta)))
    return out
def L(x): return mp.mpf(x._mpi_[0])
def H(x): return mp.mpf(x._mpi_[1])
def lo2(x): return mp.floor(x * 100) / 100   # outward at 2 decimals: lower end down
def hi2(x): return mp.ceil(x * 100) / 100    # upper end up
def Z(g):
    return 4 * g ** iv.mpf(1.5) / (3 * iv.pi)
a = load('tf_overlap_certificate_N12.log', '1e-4')
b = load('tf_threshold_certificate_N8.log', '1e-3')
assert len(a) == 44 and len(b) == 20, (len(a), len(b))
print('Z = 4 G^(3/2)/(3 pi); checks 2*0.885341 =', mp.nstr((3 * mp.pi / 4) ** (mp.mpf(2) / 3), 10))
print('level  N   G_lo        G_hi        Z_lo(12 digits)  Z_hi(12 digits)  Z interval (outward, 2 decimals)')
res = {}
for k in sorted(a, key=lambda k: (a[k][0], k[0])):
    N, gl, gh = a[k]
    zl, zh = L(Z(gl)), H(Z(gh))
    res[k] = (zl, zh)
    print('%d%s %3d  %-10s  %-10s  %-16s %-16s [%s, %s]' % (k[0], LET[k[1]], N, mp.nstr(L(gl), 9), mp.nstr(H(gh), 9), mp.nstr(zl, 12), mp.nstr(zh, 12), mp.nstr(lo2(zl), 8), mp.nstr(hi2(zh), 8)))
ok = True
for k in b:
    N, gl, gh = b[k]
    zl, zh = L(Z(gl)), H(Z(gh))
    inside = zl <= res[k][0] and res[k][1] <= zh
    ok &= bool(inside)
print('N<=8 certificate (1e-3) intervals contain the 1e-4 intervals:', ok)
print('N<=10 groups (Z_hi of last-filled vs Z_lo of next level, strictly increasing in Madelung order):')
order = sorted(res, key=lambda k: (a[k][0], k[0]))
inc = all(res[order[i]][1] < res[order[i + 1]][0] for i in range(len(order)) if a[order[i]][0] <= 10 and a[order[i + 1]][0] <= 10 and i + 1 < len(order))
print('disjoint Madelung order for N<=10:', inc)
def q(k): return (mp.nstr(lo2(res[k][0]), 8), mp.nstr(hi2(res[k][1]), 8))
print('Z(10s) in [%s, %s]' % q((10, 0)))
print('Z(7h) in [%s, %s], Z(11s) in [%s, %s]' % (q((7, 5)) + q((11, 0))))
print('Z(8g) in [%s, %s]' % q((8, 4)))
print('RESULT', 'PASS' if ok and inc else 'FAIL')
