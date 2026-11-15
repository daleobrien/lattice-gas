# Transport coefficients

[Back to the README](../README.md)

A lattice gas is only a fluid after averaging, and the viscosity and advection
coefficient it ends up with are properties of the rules, not free parameters.
The program measures both instead of quoting them:

```bash
./target/release/lgca --measure
```

This runs on whichever backend the simulation will, so the coefficients
reported are the ones the running code actually has. The two agree: `nu` 0.2989
on the CPU against 0.3080 on the GPU, `g` 0.4405 against 0.4361, both inside
the spread of the estimator across seeds.

* **Viscosity** comes from a shear wave seeded in a periodic box, whose
  amplitude decays as `exp(-nu k^2 t)`.
* **Advection factor `g`** comes from a transverse wave riding on a uniform
  stream. The pattern is advected at `g U`, not at `U`, so the drift of its
  phase measures `g` directly. A real fluid has `g = 1`; a lattice gas does
  not, because its equilibrium is not Galilean invariant.

Both are read off a single Fourier mode, averaged over several independent
realisations before fitting so that the lattice's own noise cancels rather than
biasing the fit. The Reynolds number is then `g U L / nu`.

`--scan-density` sweeps this across densities. The measured `g` follows the
FHP expression -- `(1/2)(1-2d)/(1-d)` for six directions, and `(7/12)` in
place of the `(1/2)` once rest particles dilute the momentum per unit mass --
to within about ten percent, and vanishes at half filling as it must. That is
a useful check on the lattice and the collision table together, since `g`
depends on the equilibrium distribution rather than on the choice of
collisions. `g` repeats to better than two percent between seeds; `nu` is the
noisier of the two, at roughly ten percent. Measurements below about 0.8
particles per cell should not be trusted, as there are too few particles to
resolve the mode.

Representative values with rest particles, alongside the FHP prediction
`g = (7/12)(1-2d)/(1-d)`:

| particles/cell | nu | g measured | g predicted | g/nu |
|---|---|---|---|---|
| 0.91 | 0.44 | 0.515 | 0.496 | 1.17 |
| 1.12 | 0.41 | 0.492 | 0.472 | 1.21 |
| 1.33 | 0.36 | 0.474 | 0.447 | 1.31 |
| 1.54 | 0.31 | 0.450 | 0.419 | 1.47 |
| 1.75 | 0.28 | 0.419 | 0.389 | 1.51 |
| 2.17 | 0.24 | 0.359 | 0.321 | 1.50 |
| 2.80 | 0.20 | 0.225 | 0.194 | 1.12 |

The Reynolds coefficient is flattest between about 1.5 and 1.8 particles per
cell, which is where the defaults sit.
