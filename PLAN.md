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

## Phase 1 --- coarse-graining --- done

No physics risk: this changed how the lattice is *read*, not how it evolves.

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

The same table serves `Lattice::mean_velocity` and `row_ux` / `column_uy` in
[`src/transport.rs`](src/transport.rs). The 16-bit fields cap the block area at
8,191 cells; the default is 400.

### What it did

| Case | Before | After | |
| --- | ---: | ---: | ---: |
| `field/sample/2048x1280` | 22.6 ms | 0.192 ms | **118x** |
| `lattice/mean-velocity/2048x1280` | 23.1 ms | 0.507 ms | **46x** |
| `step/2048x1280/tN` | 2.02 ms | 1.521 ms | 1.33x |
| `lattice/total-particles/2048x1280` | 44.8 us | 45.4 us | --- |

The update got 33% faster for free: dropping the parallel `Vec<bool>` took a
2.6 MB stream out of its inner loop as well. `total_particles` had to start
masking `SOLID_BIT` off, which cost 24% until it was rewritten to popcount
eight cells per instruction, and it is now back where it started.

End to end, a 5,000-step run of the binary went from **28.3 s to 8.5 s**, and
the default run from 4.7 minutes to about 1.4. The status line went from 370M
to 1230M cell-updates/s.

### That it is the same simulation

`golden_output_is_unchanged` reported exactly the expected pattern, which is
what makes the change safe to believe:

* **Mass and momentum were identical in all five cases.** The simulation itself
  is untouched; not one particle moved differently.
* **The cell checksum moved only for the two cases with a plate.** The three
  periodic cases are byte-identical, which is what pins `SOLID_BIT` as the only
  difference in the array.
* **The field checksum moved everywhere**, from exact integer accumulation
  replacing the `f32` running sum.
* `physics_is_unchanged`, `stepping_is_deterministic` and
  `thread_count_does_not_change_conserved_quantities` all stayed green, the
  last of which covers the newly threaded `Field::sample`.

`Lattice::mean_velocity` still reports the same numbers to six decimals, so the
recorded `ux` and `uy` did not move at all. Decoding the output PNGs before and
after, the frames differ on 0.005% of channels by one part in 255 --- and the
new arithmetic is the more accurate of the two.

## Phase 2 --- bitplane the update --- next

About a week. This is the change that alters how the project feels to work on,
and with Phase 1 done the update is 91% of the run, so it is now the only thing
worth attacking.

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

### What the collision actually costs --- measured, and it changes the plan

The rule turned out to be far more tractable to *derive* than expected and far
more expensive to *run* than expected.

Enumerating the momentum classes gives only **eight non-singleton class types**
up to the twelve-element symmetry group: two of size 5, four of size 3, two of
size 2, covering 28 classes and 76 of the 128 states. A generator walks that
enumeration --- the same one `collision.rs` already does at runtime, so the two
cannot drift --- and emits the exact rule as Boolean logic over the seven
planes, given a fair bit, a 1-of-3 selector and a 1-of-5 selector. Rejection
sampling supplies the last two, leaving a lane unresolved (and so unchanged)
1.6% and 2.0% of the time, which is doubly stochastic and therefore still sound
physics. It verifies exhaustively: all 6,144 (state, draw) combinations come out
uniform over the class and conserving.

It costs 685 boolean operators after sharing the minterm prefix tree, and that
is the problem:

| At 2048x1280 | ms/step | against today |
| --- | ---: | ---: |
| today, byte per cell, 12 threads | 1.521 | --- |
| CPU bitplane, exact rule, 12 threads | 0.441 | 3.4x |
| GPU, exact rule, 100 steps per command buffer | **0.034** | **45x** |

**The CPU version is register-bound, not ALU-bound.** Widening the word from
one `u64` to two and then four made it *slower* --- 2.68, 2.89, 3.24 ms on one
thread --- because the live values spill. Sharing the minterm tree cut the
operator count 29% and bought almost nothing, because the compiler was already
doing that CSE. There is no obvious further factor of two on the CPU.

**The GPU barely noticed the real rule**: 0.011 ms/step with a toy FHP-I
collision, 0.034 ms with the exact one, propagation shifts included. It has the
registers and the ALUs that twelve CPU cores do not.

So the ordering in this document was wrong, and it was wrong for a reason worth
recording: **the GPU is the third-best idea only when the collision is cheap.**
The measurement that put it third was a bandwidth argument, and it was correct
for the rule it was measured with. The actual maximal rule is compute-bound,
and compute is precisely where the shared memory controller stops mattering.

**Effect on the default run: CPU bitplane takes ~1.4 min to ~25 s (3.3x).
Going to Metal with the same rule takes it to ~4 s (about 20x).**

## Phase 3 --- Metal --- in progress

Promoted ahead of the CPU bitplane rewrite on the strength of the measurement
above: same rule, 3.4x on twelve cores against 45x on the GPU.

### Landed

* [`src/metal.rs`](src/metal.rs) --- a hand-rolled Metal binding. Metal has no C
  API, but `objc_msgSend` is an ordinary C symbol, so the whole thing is a few
  typed casts of it. `Cargo.lock` is still empty. Buffers are shared rather
  than private, which on Apple silicon means the CPU and GPU address one
  allocation and the analysis passes need no transfer --- and it measures the
  same as private storage.
* [`src/gpu.rs`](src/gpu.rs) --- the lattice as bitplanes, 32 cells to a `uint`,
  seven planes plus a solid plane that sits outside the ping-pong because it
  never changes. Propagation is a shift with a carry from the neighbouring
  word, plus a fix-up for the one bit per row that wraps, so widths that are
  not a multiple of 32 work too.
* The collision circuit is **emitted at startup from `collision::classes`**,
  the same enumeration the CPU lookup table is built from. Neither is
  hand-written, so they cannot drift into describing different physics.

### Measured

| At 2048x1280 | ms/step | against the CPU |
| --- | ---: | ---: |
| CPU, byte per cell, 12 threads | 1.504 | --- |
| GPU, one step per command buffer | 0.201 | 7.5x |
| GPU, 100 steps per command buffer | **0.0345** | **43.6x** |

The two GPU rows are the same work submitted two ways. A round trip costs more
than seven steps do, which is the design constraint the rest of this phase has
to respect.

### Checked

Six tests, all of which run the shader that ships rather than a
transliteration of it:

* `the_shader_rule_matches_the_enumeration` runs the emitted circuit over every
  state and every draw --- 6,144 combinations for FHP-III, 3,072 for the
  six-direction model --- and checks each against the class it came from.
* `a_single_particle_travels_in_a_straight_line` covers all six directions,
  both row parities, and a width of 80 so that the ragged last word is
  exercised.
* `walls_reverse_particles`, `a_periodic_box_conserves_mass_and_momentum` over
  250 steps, `load_and_store_round_trip` on a width of 100, and a bulk
  comparison against the CPU.

### Still to do

* Coarse-graining as a GPU kernel, so a frame does not force a round trip every
  fifth step.
* The inlet re-seed, which is 44% of a step at this speed if it stays on the CPU.
* Driving `transport::measure` from the GPU path, and with it the acceptance
  test: nu and g must still come out at 0.2989 and 0.440.
* Wiring into `main.rs`, with the CPU path kept as the reference the golden
  tests run against.

### Notes from building it

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
