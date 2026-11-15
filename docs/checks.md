# Checks

[Back to the README](../README.md)

`cargo test` covers the parts that fail quietly rather than loudly:

* Mass and momentum are conserved exactly -- as integers, not to within a
  tolerance -- both by every entry of the collision table and by 200 steps of
  a randomly filled periodic box.
* A lone particle, which has no collision partner, flies in a straight line in
  all six directions from rows of both parities. This is what pins down the
  neighbour offsets of the staggered lattice.
* A particle fired at a wall comes back the way it came.
* The equilibrium sampler delivers the velocity it is asked for, with and
  without stationary particles.
* The PNG writer's deflate output round-trips through a matching decoder.

The GPU path is a second implementation of the same rules, so it gets its own
layer of checks. Its collision circuit is not hand-written: it is emitted at
startup from the same enumeration of momentum classes the CPU lookup table is
built from, and a test runs the shader that ships over every state and every
draw to confirm it agrees with that enumeration. Propagation, walls, the
inflow boundary and the coarse-graining are each checked against the CPU
version. Two `#[ignore]`d tests then ask the question that matters -- whether it
is the same *fluid* -- by measuring viscosity and the advection factor on both
paths through an identical protocol from identical initial cells:

```bash
cargo test --release --lib -- --ignored --nocapture
```

`tests/golden.rs` adds the checks that matter when optimising rather than when
writing: that a fixed run still produces the same lattice bit for bit, that it
produces the same answer twice running, and that mass and momentum come out
exactly equal however the rows are divided between threads. A change to the
order of the random draws trips the first of those and none of the others,
which is the signal that the stream moved but the fluid did not.

Two things fall out of the physics rather than the tests: the advection factor
matches the FHP prediction across the whole density range, as [Transport coefficients](transport.md) describes,
and the wake sheds at a Strouhal number of about 0.11, which is the right range
for a plate held across the flow at this Reynolds number.
