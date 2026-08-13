# Diffractor v2 workspace

Two packages, one validation chain:

    exact analytic  →  groundtruth (BOR-BEM)  →  diffractor
    (scalar ball,       Müller second-kind        (exact propagators +
     planar R+T=1)      solver, O(h²))             interface operators)

- `diffractor/` — the product, ordered by the nature of each thing:
  `geometry/` pure shape; `optics/` matter and its boundaries (Medium — a
  class, an index — and Interface: a geometric locus Σ + two media, nothing
  else); `scattering/` the interface response (scalar t_s, exact planar
  spectral operator); `propagation/` transport through space (exact ASM/RS1,
  gated paraxial, and the GENERAL energy-conservation factors in
  transport.py); `analysis/` measurements (OPL/OPD, demodulated pupils,
  energies).
- `groundtruth/` — reference solvers, never imported by diffractor.
- `benchmarks/golden/` — frozen session results (measured ovoid pupil, energy).
- `notebooks/` — the results, in the order the physics is built up: geometry,
  solver validation, the field of the stigmatic ovoid, and the models scored
  against the solver. Committed with their outputs; see `notebooks/README.md`.
- `tests/` — every claim ported from the validation session, as executable
  assertions. `python -m pytest tests/`

See PLAN.md for the phase roadmap and the interface-operator validation ladder.
