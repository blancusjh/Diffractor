# Notebooks

Four notebooks, in the order the physics is built up. Each one is
self-contained and can be read without running it: they are committed with
their outputs.

| | notebook | what it establishes |
|---|---|---|
| 1 | `01_stigmatic_geometry.ipynb` | the Cartesian oval; stigmatism, Snell and the ray tubes verified from the coordinates; the ray pupil $P=t_s d_2/d_1$ and the two ways of writing it |
| 2 | `02_solver_validation.ipynb` | Müller BOR-BEM against the exact scalar ball: $O(h^2)$ convergence, interior field, and the range of $k_2a$ where the solver is usable |
| 3 | `03_stigmatic_focus.ipynb` | the field of the ovoid: meridional sections, focal region, and the two reference spheres with amplitude and phase separated |
| 4 | `04_models_against_the_solver.ipynb` | the tangent-plane, Debye–Wolf and thin-element models scored against the solver |

Notebooks 1 and 2 compute everything they show, in seconds. Notebooks 3 and 4
read the frozen results in `benchmarks/golden/`; set `RECOMPUTE = True` in the
first cell of notebook 3 to rerun the solves instead (about ten minutes).

`nbstyle.py` holds the palette, the shared axes style and the loaders, so the
notebooks stay about physics.

## Running them

```
pip install -e diffractor -e groundtruth      # from the repository root
jupyter lab notebooks/
```

The notebooks expect the working directory to be `notebooks/`, which is what
Jupyter and `nbconvert --execute` both do by default.

## Regenerating the standalone figures

The four PNGs in `benchmarks/` come from the same stored results:

```
python benchmarks/compute_fields.py     # the solves, ~15 min
python benchmarks/fig_fields.py         # the figures, seconds
```
