# Making it faster

Measured on the machine in [`benches/baseline.txt`](benches/baseline.txt) --- Apple
M3 Pro, 12 CPU cores (6 performance + 6 efficiency), 18 GPU cores, 150 GB/s of
unified memory. Every number below is a measurement, either from
`cargo bench` or from a throwaway prototype written to settle one question.

## The finding

The obvious move on this machine is to reach for the GPU. That turns out to be
the third-best idea, and doing it first would have been a mistake.

The update is **bandwidth-bound**, and on an Apple part the GPU shares the same
memory controller as the CPU. A dispatch-per-step Metal port measures 0.23 ms
against 0.11 ms for a well-packed CPU kernel: the GPU is *twice as slow* until
the submission cost is amortised over a batch. What actually pays is the data
layout, and it pays on the CPU first.

The order is: coarse-graining (47--156x), bitplane packing (15--25x), and only
then the GPU (a further 4--10x, and mostly worth it for making bigger runs
possible rather than for making this run quicker).

## Where the time goes

The budget in [`BENCHMARKS.md`](BENCHMARKS.md) reproduces: for the default run,
4.7 minutes of wall time, spent like this.

| Phase | Calls | Each | Total | Share |
| --- | ---: | ---: | ---: | ---: |
| `Field::sample` | 8,001 | 22.6 ms | 181 s | **64%** |
| `Lattice::step` | 46,000 | 2.02 ms | 93 s | 33% |
| `transport::measure` at startup | 1 | 5.4 s | 5.4 s | 2% |
| Writing frames | 80 | 9.2 ms | 0.7 s | <1% |

One number frames the whole document. `Lattice::total_particles` reads the 2.6 MB
cell array in 45 us --- 58 GB/s, which is what the memory system will give a
single core. `Field::sample` reads *the same array, byte for byte* in 22.6 ms.
The 500x between them is implementation, not physics.

## Phase 1 --- coarse-graining

Half a day. No physics risk: this changes how the lattice is *read*, not how it
evolves.

`Field::sample` tests six bits and does three `f32` adds per cell, on one
thread, while also streaming a second 2.6 MB `Vec<bool>` for the solid flag.

1. **Fold the solid flag into bit 7 of the cell byte.** Bits 0--5 are the moving
   directions, bit 6 is the rest particle, bit 7 is free. This deletes a 2.6 MB
   array from *both* hot loops --- `stream_and_collide_row` already loads the
   cell byte for its rest bit, so testing solidity becomes free there.
2. **Replace the bit tests with a 256-entry `u64` lookup table.** Each entry
   packs four 16-bit fields that add without carrying into one another: mass,
   x-momentum, y-momentum and a solid counter. One load and one add per cell.
   Momentum accumulates as exact integers in `CX2`/`CY2` units and is converted
   once per block, which is also *more* accurate than summing 2,800 `f32`s.
3. **Thread it over block rows.** There are 64 of them at the default block size.

Measured on a prototype: **22.76 ms -> 0.482 ms on one thread, 0.146 ms on six.**

The same table serves `Lattice::mean_velocity` (23.3 ms) and `row_ux` /
`column_uy` in [`src/transport.rs`](src/transport.rs).

The 16-bit fields cap the block area at 8,191 cells; the default is 400.

**Effect on the default run: 4.7 min -> ~100 s (2.8x).**

## Phase 2 --- bitplane the update

About a week. This is the change that alters how the project feels to work on.

Store 64 cells per `u64`: seven planes for the six directions and the rest
particle, plus a solid plane. This is how the original FHP simulations ran, and
it changes the cost structure rather than shaving it.

* **Propagation becomes shifts.** Direction `d` is the whole plane shifted one
  bit with a carry from the neighbouring word, with row parity choosing the
  constants. The per-cell `rem_euclid` branch at the row ends disappears.
* **Collision becomes Boolean logic.** No table, no data-dependent load, no
  per-cell random draw --- one `u64` from the generator now feeds 64 cells
  instead of one, a 64x cut in generator work.
* **Walls become a select**: `(solid & reversed) | (!solid & collided)`.

Measured on a prototype at 2048x1280, with conservation verified exhaustively
over all 64 states: **10.97 ms -> 0.40 ms on one thread, 1.95 ms -> 0.108 ms on
six.**

Three things the prototype turned up that are otherwise found late:

* **Collision complexity is nearly free.** Sweeping extra Boolean work: 40 extra
  operations per word cost about 5%, 80 cost about 30%. Being bandwidth-bound
  means a rich rule set is affordable, which is what makes the next point
  workable.
* **The hard part is uniform-over-3 and uniform-over-5.** The rule is "uniform
  over the momentum class", and the classes have sizes {1, 2, 3, 5} --- hence
  the stride of 30. One random bit handles a pair; 3 and 5 need masked rejection
  sampling, looping a bounded number of rounds over an "unresolved lanes" mask.
  The previous point says that is affordable.
* **The serial inlet becomes the bottleneck.** `apply_inlet` costs about 38 us
  at 512 rows, so about 95 us at the production height. That is 6% of a step
  today and would be **44%** of a bitplane step. Generate the equilibrium as
  bitplanes through an AND/OR probability circuit --- `p` to eight bits of
  precision in about eight operations per word, 64 cells per draw.

**Re-tune the thread count afterwards rather than assuming it.** Today's
byte-per-cell kernel is compute-bound and 12 threads beat 6 (1.58 ms against
2.04 ms). The bitplane kernel is bandwidth-bound and it inverts --- 6 beat 12
(0.108 ms against 0.113 ms), because the efficiency cores add no bandwidth.

**The acceptance test already exists.** The golden checksums *will* move, which
[`BENCHMARKS.md`](BENCHMARKS.md) already licenses. What must not move is the
physics: `transport::measure` has to keep reporting nu = 0.2989 and g = 0.440
within noise. That is a quantitative check on a rewrite that changes every
random draw in the program. Keep the byte kernel as the reference
implementation.

**Effect on the default run: ~100 s -> ~10 s (about 30x overall).**

## Phase 3 --- Metal, if bigger runs are wanted

About a week, and optional.

The measurement that decides the design: **an empty GPU round trip costs
0.094 ms**, nearly as much as a whole six-thread bitplane step. One dispatch per
step measures 0.23 ms, which is twice as slow as the CPU. Batching fixes it:

| Steps per command buffer | ms/step at 2048x1280 |
| ---: | ---: |
| 1 | 0.230 |
| 10 | 0.031 |
| 100 | 0.011 |

So the GPU only pays if the program stops synchronising every step --- which
means **coarse-graining has to become a GPU kernel too**, accumulating into a
device-side field buffer that is read back only when a frame is written. Expect
0.02--0.03 ms/step in practice; the prototype kernel omits the neighbour shifts.

The real prize is size rather than speed at the current size. At 8192x5120
(41.9M cells) the GPU sustains 0.58 ms/step, a full run in under a minute.
Today's code would need over an hour for the same thing. That is the Re ~ 400
regime, which is currently out of reach.

On dependencies: the empty `Cargo.lock` is clearly deliberate, and the `metal`
crate pulls in `objc`, `block` and `foreign-types`. The prototyping above was
done with an 80-line `.m` file compiled by `clang`, so either a `build.rs` with
a small Objective-C shim or roughly 300 lines of `objc_msgSend` FFI keeps the
dependency list empty.

**Effect on the default run: ~10 s -> ~3 s (about 90--120x overall)**, at which
point PNG and SVG encoding and process startup dominate.

## Not a lever

Measured, and recorded here so the question does not have to be asked again.

* **Density tuning is exhausted.** `--scan-density` puts the peak of `g/nu` at
  1.556 near d = 0.31, against 1.474 at the default 0.22. That is 5.6% more
  Reynolds number per unit of obstacle size. Since total work scales as the cube
  of the obstacle size, the best case is about 15% less work. The default is
  well chosen. This was the most promising algorithmic lever and it is not
  there.
* **The collision table is not the bottleneck.** The six-direction table (384
  bytes) and the FHP-III table (3,840 bytes) run at the same speed, as
  [`BENCHMARKS.md`](BENCHMARKS.md) already suspected.

## Where to stop

Phase 1 is half a day for 2.8x with no physics risk, so it is worth doing
whatever else happens. Phase 2 is the one that matters: 4.7 minutes to 10
seconds is the difference between a run you schedule and a run you watch. Phase
3 is a genuine week for 3x at the current size, and is worth taking only for the
larger lattices it makes possible.
