# Canonical cases (the validation ladder)

| # | Case                             | Reference                    | Status |
|---|----------------------------------|------------------------------|--------|
| 1 | Planar interface, oblique PW     | spectral t(k⊥), exact        | primitive implemented (`scattering/planar.py`); ladder test pending |
| 2 | Planar interface, point source   | RS1 + t(k⊥)                  | pending |
| 3 | Shallow spherical cap            | BEM + scalar ball            | pending |
| 4 | Stigmatic ovoid NA 0.3–0.75      | P = t_s·d2/d1 + BEM          | BLOCKED: the closed benchmark body is a resonator (±38 % swing with shroud length). Needs the open-surface formulation. `golden/shroud_test.npz`, `golden/ensemble.npz` |
| 5 | Full sphere (caustic regime)     | scalar ball exact            | pending |

Golden data conventions: pupils stored as (th, A_measured, A_predicted);
energies via 2π∫|A|²sinθdθ. Absolute scale: point source A0 = 1/4π. Never fit scales.


## Fase 1 note — why case 4 is blocked

Closing the ovoid to give BEM a closed surface creates a lossless dielectric
cavity.  The reverberant field it supports adds ~20 % to the energy through S2
and corrupts the pupil edge, where the direct field is weakest.  Averaging over
cavity modes stabilises the ENERGY (1.20 ± 0.01 × prediction) but not the
edge factor (converges to 0.48 vs 0.219 predicted).

Lesson recorded: mesh convergence was verified and was excellent — it just
converged to the answer for the wrong geometry.  Convergence of the
discretisation is not validation of the model.
