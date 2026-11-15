# The rules

[Back to the README](../README.md)

Six directions, plus optionally a seventh stationary state, give 64 or 128
possible cell states packed into one byte. The collision rule is generated
rather than hand-written: states are grouped by particle count and total
momentum, and a collision replaces a state with one drawn uniformly from its
own group. This conserves both invariants exactly, is symmetric (so
semi-detailed balance holds and the equilibrium is the expected Fermi-Dirac
distribution), and fires on every configuration that has an alternative --
which is what keeps the viscosity low.

The stationary state is the FHP-III model and is on by default. It roughly
halves the viscosity and so roughly doubles the Reynolds number reachable on a
given lattice. `--no-rest` gives the six-direction rule the book uses.

Obstacles are handled by bounce-back: particles that stream into a solid site
have their directions reversed and leave the way they came. The left edge is
re-seeded each step from an equilibrium distribution at the requested density
and speed, which is what maintains the flow; particles that leave the right
edge wrap around into that zone and are overwritten, making the outflow
effectively absorbing.
