#!/usr/bin/env python3
"""Run the checks of the paper and print PASS/FAIL per check. Run from the repository root:
    OMP_NUM_THREADS=1 nice -n 19 python3 code/run_all.py [--figures] [--certificate] [--overlap] [--table] [--path] [--arb] [--robustness]
Default: the fast checks that need no long computation (about 2 minutes on one core), run on the data and logs
shipped in code/. Opt-in heavy runs (each one process, one thread):
    --figures      rewrite figures/ with make_figures.py (seconds)
    --certificate  tf_threshold_certificate.py 8 1e-3     (Madelung order of thresholds, N <= 8; about 8 minutes)
    --overlap      tf_overlap_certificate.py 12 1e-4      (overlap certificate, N <= 12; about 2 hours)
    --table        tf_threshold_table_N30.py table 30     (rebuilds tf_threshold_table_N30.csv; hours; overwrites it)
    --path         tf_threshold_table_N30.py path 17      (path onsets eps*_N, N <= 17; about 1 hour)
    --arb          arb_crosscheck.py                      (second-library check; needs python-flint)
    --robustness   tf_robustness.py aufbau 330            (orderings under corrected one-body fields, Remark I; logs stored as
                   tf_robustness_aufbau.log and tf_robustness_cross_0.002.json; opt-in, not run by default)
The field-family runs (x4_*.json) are analysed from the stored files; to redo one, see the usage line of tf_x4_fields.py.
Needs Python 3, numpy, scipy, sympy, mpmath, matplotlib (and python-flint for --arb).
Exit status is nonzero if anything fails.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(HERE, script), *args], capture_output=True, text=True, cwd=HERE)
    return p.returncode, p.stdout + p.stderr


flags = set(sys.argv[1:])
rows = []

rc, out = run("cor_c_z_ranges.py")
rows.append(("T1  Z intervals of first filling from the certified brackets; first break 7h < 11s < 8g",
             rc == 0 and "RESULT PASS" in out))

rc, out = run("make_figures.py", "--check")
rows.append(("T2  figure layout check (no text over text, lines, arrows, patches; inside canvas)",
             rc == 0 and out.count(": PASS") >= 4))

rc, out = run("tf_threshold_table_N30.py", "tests")
# the script's Kelvin-parity line reports FAIL by design (the alternation is an endpoint effect, see the paper)
rows.append(("T3  table N <= 30: Madelung order in the pairs, largest gaps, overlap for every N = 11..29, endpoint effect",
             rc == 0 and all(k in out for k in ("T8 PASS", "T9 PASS", "X3 PASS", "X2 endpoint PASS"))))

rc, out = run("tf_wkb.py")
rows.append(("T4  Bohr-Sommerfeld limit s_inf = 0.25209 and the 1/N law against the table",
             rc == 0 and "s_inf=0.2520947107" in out and "-> PASS" in out))

rc, out = run("tf_x4_fields.py", "analyse", "x4_A_h512.json", "x4_B_h512.json")
rows.append(("T5  fields of the path family: first overlap and N* s_inf from the stored runs",
             rc == 0 and "N* s_inf=" in out and "Traceback" not in out))

if "--figures" in flags:
    rc, out = run("make_figures.py")
    rows.append(("F1  figures/ rewritten", rc == 0))
if "--certificate" in flags:
    rc, out = run("tf_threshold_certificate.py", "8", "1e-3")
    rows.append(("L1  interval certificate, thresholds in Madelung order, N <= 8", rc == 0 and "VERDICT PASS" in out))
if "--overlap" in flags:
    rc, out = run("tf_overlap_certificate.py", "12", "1e-4")
    rows.append(("L2  overlap certificate, N <= 12 (7h < 11s < 8g)", rc == 0 and "VERDICT PASS" in out))
if "--table" in flags:
    rc, out = run("tf_threshold_table_N30.py", "table", "30")
    rows.append(("L3  threshold table rebuilt, N <= 30", rc == 0))
if "--path" in flags:
    rc, out = run("tf_threshold_table_N30.py", "path", "17")
    rows.append(("L4  path onsets along (1-eps) A T + eps phi_TF, N <= 17", rc == 0 and "T10 PASS" in out))
if "--robustness" in flags:
    rc, out = run("tf_robustness.py", "aufbau", "330")
    rows.append(("R1  robustness of the orderings to corrected fields (see Remark I)", rc == 0))
if "--arb" in flags:
    rc, out = run("arb_crosscheck.py")
    rows.append(("L6  Arb cross-check of the bracket-end signs", rc == 0 and "ARB CROSS-CHECK PASS" in out))

width = max(len(r[0]) for r in rows)
print()
for name, ok_ in rows:
    print(f"{name.ljust(width)}  {'PASS' if ok_ else 'FAIL'}")
skipped = [f for f in ("--certificate", "--overlap", "--table", "--path", "--arb", "--robustness") if f not in flags]
if skipped:
    print("\nnot run (opt-in): " + " ".join(skipped))
allok = all(ok_ for _, ok_ in rows)
print("\nALL PASS" if allok else "\nSOME FAILED")
sys.exit(0 if allok else 1)
