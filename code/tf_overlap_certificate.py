#!/usr/bin/env python3
"""Paper 3: certify the first group overlap N* = 11 of the TF zero-energy thresholds.
Imports tf_threshold_certificate (not modified).  .
Pass line: (a) within every group 9 <= N <= 12 the levels are in Madelung order (disjoint brackets);
(b) for every N <= 10 the highest level of group N lies below the lowest level of group N+1;
(c) G(11, l=0) > G(12, l=5) certifiably;
(d) only-inversion clause: 10p < 7h < 8g, 11s < 8g, 8g < 9f, 7i, 8h (levels 7i, 8h of group 13 are certified too,
with the monotonicity lemma in l and n_r of the note this gives: (11s, 7h) is the only inversion below G(8g)).
PASS iff (a), (b), (c), (d).
Run: nice -n 19 env OMP_NUM_THREADS=1 python3 tf_overlap_certificate.py [NMAX=12] [delta=1e-4]
Order of work: (c) pair first, then the rest (so a time limit still gives (c)).
"""
import sys, os, time, csv, json
import mpmath as mp
import tf_threshold_certificate as T
from tf_threshold_certificate import iv, Ambiguous, certify_level, make_sandwich, peval


def main():
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    delta = sys.argv[2] if len(sys.argv) > 2 else '1e-4'
    here = os.path.dirname(os.path.abspath(__file__))
    allrows = list(csv.DictReader(open(os.path.join(here, 'tf_threshold_table.csv'))))
    rows = [r for r in allrows if int(r['N']) <= NMAX]
    order0 = [(int(r['N']), int(r['n'])) for r in rows]
    assert order0 == sorted(order0), 'rows not in Madelung order (N, then n ascending)'
    want = [(N, N - l) for N in range(1, NMAX + 1) for l in range((N - 1) // 2, -1, -1)]
    assert order0 == sorted(want) and all(int(r['N']) == int(r['n']) + int(r['l']) for r in rows), 'rows incomplete'
    extra = [r for r in allrows if (int(r['N']), int(r['l'])) in ((13, 6), (13, 5))]   # 7i, 8h
    assert len(extra) == 2 and all(int(r['N']) == int(r['n']) + int(r['l']) for r in extra), 'extra rows missing'
    t00 = time.time()
    sols = {}
    def get_sols(ratio):
        # a finer relative step (ratio) is used only for levels where the coarse step cannot isolate a zero
        if ratio not in sols:
            T.RATIO = ratio
            sols[ratio] = make_sandwich()
            T.RATIO = 0.03
        return sols[ratio]
    sol_lo, sol_hi = get_sols(0.03)
    print('sandwich lo', sol_lo.result, 'tend %.1f' % sol_lo.tend, 'hi', sol_hi.result, 'tend %.1f' % sol_hi.tend,
          'range s <= %.4g' % (min(sol_lo.tend, sol_hi.tend) ** 2), 'time %.0fs' % (time.time() - t00), flush=True)
    assert sol_lo.result == 'hits_zero' and sol_hi.result == 'turns_up'
    def y_at(sol, tt):
        st = next(s for s in sol.steps if s['t0f'] <= tt < s['t0f'] + s['h'])
        hh = iv.mpf(tt) - st['t']
        assert hh.a >= 0 and hh.b <= st['hi'].b, 'y_at: point outside the step'
        return peval(st['yc'], hh) + st['ry'] * hh ** (T.NT + 1)
    def tail_of(sh):
        psi = iv.mpf(144) / iv.mpf(T.T_S) ** 6
        return y_at(sh, T.T_S).b < psi.a
    tail_ok = tail_of(sol_hi)
    print('tail hypothesis', tail_ok, flush=True)
    first = [r for r in rows if (int(r['N']), int(r['l'])) in ((11, 0), (12, 5))]
    order = first + [r for r in rows if r not in first] + extra
    res = {}
    for r in order:
        n, l = int(r['n']), int(r['l']); t0 = time.time()
        o = None
        for ratio in (0.03, 0.01, 0.003):
            try:
                sl, sh = get_sols(ratio)
                if ratio != 0.03:
                    print('fine sandwich ratio', ratio, 'lo', sl.result, 'tend %.2f' % sl.tend, 'hi', sh.result, 'tend %.2f' % sh.tend,
                          'range s <= %.4g' % (min(sl.tend, sh.tend) ** 2), 'tail', tail_of(sh), flush=True)
                if ratio != 0.03 and not (sl.result == 'hits_zero' and sh.result == 'turns_up' and tail_of(sh)):
                    raise RuntimeError('fine sandwich invalid')
                o = certify_level(sl, sh, n, l, mp.mpf(r['G_final_S1e6']), delta, None)
                o['ratio'] = ratio
                break
            except (Ambiguous, RuntimeError) as e:
                err = str(e)
        if o is None:
            o = dict(n=n, l=l, N=n + l, G_num=mp.mpf(r['G_final_S1e6']), hi_ok=False, lo_ok=False, err=err)
        o['time'] = time.time() - t0
        res[(int(r['N']), l)] = o
        ns = o.get('node_s_upper')
        print(o['N'], n, l, 'G_num', mp.nstr(o['G_num'], 10), 'hi_ok', o.get('hi_ok'), 'lo_ok', o.get('lo_ok'),
              'node_s<=%s' % (None if ns is None else '%.0f' % ns), 'c_hi', o.get('c_hi'), 'c_lo', o.get('c_lo'),
              'w_hi', o.get('w_hi_S'), 'g', o.get('g'), o.get('err', ''), 'ratio', o.get('ratio'), '%.0fs' % o['time'], flush=True)
        if (int(r['N']), l) == (12, 5) and (11, 0) in res and 'Glo' in res[(11, 0)] and 'Ghi' in res[(12, 5)]:
            a, b = res[(11, 0)], res[(12, 5)]
            ok = tail_ok and all(x['hi_ok'] and x['lo_ok'] for x in (a, b)) and b['Ghi'] < a['Glo']
            print('(c) G(11,0)_lo %s > G(12,5)_hi %s : %s' % (mp.nstr(a['Glo'], 12), mp.nstr(b['Ghi'], 12), 'PASS' if ok else 'FAIL'), flush=True)
    valid = tail_ok and len(res) == len(rows) + 2 and all(o['hi_ok'] and o['lo_ok'] for o in res.values())
    seq = [(int(r['N']), int(r['l'])) for r in rows]          # csv order = Madelung order (asserted above)
    def lt(x, y): return res[x]['Ghi'] < res[y]['Glo']
    # (a) consecutive levels in the same group, 9 <= N <= 12
    a_ok = valid and all(lt(x, y) for x, y in zip(seq[:-1], seq[1:]) if x[0] == y[0] and 9 <= x[0] <= NMAX)
    # (b) every member of group N lies below every member of group N+1, N <= 10 (non-adjacent groups by transitivity)
    b_ok = valid
    for N in range(1, min(10, NMAX - 1) + 1):
        grp = [k for k in res if k[0] == N]; nxt = [k for k in res if k[0] == N + 1]
        top = max(grp, key=lambda k: res[k]['Ghi']); bot = min(nxt, key=lambda k: res[k]['Glo'])
        good = (max(res[k]['Ghi'] for k in grp) < min(res[k]['Glo'] for k in nxt)) if valid else False
        print('(b) N=%d max Ghi (l=%d) %s < N=%d min Glo (l=%d) %s : %s' % (N, top[1], mp.nstr(res[top]['Ghi'], 10), N + 1, bot[1], mp.nstr(res[bot]['Glo'], 10), good))
        b_ok = b_ok and good
    c_ok = bool(valid and res[(12, 5)]['Ghi'] < res[(11, 0)]['Glo'])
    # (d) only-inversion clause (needs NMAX >= 12; the 13s, 12p, 11d, 10f, 9g part is (a) and the monotonicity lemma)
    d_ok = False
    if valid and NMAX >= 12:
        names = {(11, 1): '10p', (12, 5): '7h', (11, 0): '11s', (12, 4): '8g', (12, 3): '9f', (13, 6): '7i', (13, 5): '8h'}
        for k, nm in names.items():
            print('bracket', nm, k, '[%s, %s]' % (mp.nstr(res[k]['Glo'], 10), mp.nstr(res[k]['Ghi'], 10)))
        checks = [('10p below 7h', lt((11, 1), (12, 5))), ('11s below 8g', lt((11, 0), (12, 4))),
                  ('7h below 8g', lt((12, 5), (12, 4)))] + [('8g below ' + nm, lt((12, 4), k)) for k, nm in
                  (((12, 3), '9f'), ((13, 6), '7i'), ((13, 5), '8h'))]
        for nm, v in checks:
            print('(d) assert', nm, ':', v)
        assert lt((11, 1), (12, 5)), '10p below 7h'
        assert lt((11, 0), (12, 4)), '11s below 8g'
        assert lt((12, 5), (12, 4)), '7h below 8g'
        assert all(lt((12, 4), k) for k in ((12, 3), (13, 6), (13, 5))), '8g below 9f, 7i, 8h'
        d_ok = True
        print('(d) margins: Glo(7i)/Ghi(8g) = %s, Glo(8h)/Ghi(8g) = %s' % (mp.nstr(res[(13, 6)]['Glo'] / res[(12, 4)]['Ghi'], 8), mp.nstr(res[(13, 5)]['Glo'] / res[(12, 4)]['Ghi'], 8)))
    print('all brackets valid:', valid)
    print('RESULT (a)', a_ok, '(b)', b_ok, '(c)', c_ok, '(d)', d_ok, 'VERDICT', 'PASS' if (a_ok and b_ok and c_ok and d_ok) else 'FAIL',
          'N <= %d' % NMAX, 'delta', delta, 'total time %.0fs' % (time.time() - t00))


if __name__ == '__main__':
    main()
