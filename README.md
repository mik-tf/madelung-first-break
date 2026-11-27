# Where the n + l rule breaks in the Thomas-Fermi atom: certified zero-energy thresholds, Madelung order through n + l = 10, and the first overlap at n + l = 11

Author: Michaël Perreault

Version 1.1.1.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542256.svg)](https://doi.org/10.5281/zenodo.24542256)

In the Demkov-Ostrovsky (Tietz) field the one-electron levels with equal N=n+l reach zero energy together, and our earlier papers ("The Demkov-Ostrovsky Madelung model as the circle-invariant conformal Laplacian of the round 4-sphere", https://doi.org/10.5281/zenodo.24522163; "A first-order Madelung tie-break on the round four-sphere: Kelvin parity, convex order and the Thomas-Fermi field", https://doi.org/10.5281/zenodo.24533673) showed that a convex departure from it splits each group with lower n first. Here we study the Thomas-Fermi (TF) field, whose chemical potential is zero, so that a level reaching zero energy is a subshell that starts to fill. Theorem A: the zero-energy thresholds of all 30 levels with N<=10 are in exactly the Madelung order. Theorem B: the first break is a single transposition, 7h (N=12) is bound before 11s (N=11) near Z=310, and it is the only inversion below 8g. Both are proved with interval arithmetic at 640 bits, validated Taylor integration, Sturm comparison and an explicit tail lemma for the 144/s^3 asymptote. They concern the pure TF zero-energy model, the leading-order large-Z mean field of [BFHS25], giving, to our knowledge, the first certified answer to the question of how large an atom must be before the Madelung rule fails inside that model. A double-precision table of 240 levels up to N=30 shows overlaps of consecutive groups for every N from 11 to 29, a semiclassical law with beta=1.032 predicts the first overlap at N=11, and along the path from the Tietz to the TF field the first overlap appears at N=11 to 16. A (Z-1)/Z Fermi-Amaldi rescaling keeps the orderings, moving every threshold by about 3/2 in Z, whereas fields with a -1/r tail reorder levels from N=3 to 7, so the result is not a statement about real atoms; the subshell capacities remain an input.

Paper: [PDF](where-the-n-l-rule-breaks-in-the-thomas-fermi-atom-certified-zero-energy-thresholds-madelung-order-through-n-l-10-and-the-first-overlap-at-n-l-11.pdf), source in [paper.md](paper.md). Earlier papers: [The Demkov-Ostrovsky Madelung model as the circle-invariant conformal Laplacian of the round 4-sphere](https://doi.org/10.5281/zenodo.24522163) and [A first-order Madelung tie-break on the round four-sphere: Kelvin parity, convex order and the Thomas-Fermi field](https://doi.org/10.5281/zenodo.24533673).

## Licences

- Text (`paper.md`), figures and logs: CC BY 4.0, see [LICENSE-paper.md](LICENSE-paper.md).
- Code in `code/`: MIT, see [LICENSE](LICENSE).

## Re-run the checks

Python 3 with numpy, scipy, sympy, mpmath and matplotlib. From the repository root:

```
OMP_NUM_THREADS=1 nice -n 19 python3 code/run_all.py
```

It runs the fast checks on the data and logs shipped in `code/` and prints PASS or FAIL for each (about 2 minutes on one core). The heavy runs are opt-in flags: `--certificate` (thresholds in Madelung order, N <= 8, about 8 minutes), `--overlap` (the certificate for N <= 12 with the order 7h < 11s < 8g, about 2 hours), `--table` (rebuilds the N <= 30 table), `--path` (path onsets, N <= 17) and `--arb` (second-library cross-check, needs python-flint); `--figures` rewrites `figures/`. Each script also runs alone from the directory `code/`; the logs of the long runs are kept there.

## Cite as

Perreault, M. Where the n + l rule breaks in the Thomas-Fermi atom: certified zero-energy thresholds, Madelung order through n + l = 10, and the first overlap at n + l = 11, version 1.1.1. https://github.com/mik-tf/madelung-first-break (DOI: [10.5281/zenodo.24542256](https://doi.org/10.5281/zenodo.24542256) (all versions; v1.0 is [10.5281/zenodo.24542257](https://doi.org/10.5281/zenodo.24542257))). See [CITATION.cff](CITATION.cff).
