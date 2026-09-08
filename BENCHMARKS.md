# Benchmarks

A dependency-free benchmark suite lives in [`benches/speed.rs`](benches/speed.rs),
and the numbers it produced on the machine it was written on are recorded in
[`benches/baseline.txt`](benches/baseline.txt). `cargo bench` reads that file
and reports the change against it, so the loop for a performance change is:

```bash
cargo bench                       # where are we now, against the baseline?
# ... make the change ...
cargo test                        # did the answer change, and did it matter?
cargo bench                       # what moved, and by how much?
```

`tests/golden.rs` is the half of that loop that is easy to skip and expensive
to skip. It answers two separate questions:

* **Did the output change at all?** `golden_output_is_unchanged` checksums the
  cell array and the coarse-grained field after a fixed run from a fixed seed.
  Widening the collision loop to SIMD, or re-banding the threads, changes the
  order of the random draws and so changes these checksums. That is allowed --
  but it should be a decision, not a surprise.
* **Did the fluid change?** `physics_is_unchanged` checks mean velocity and
  particle count against tolerances wide enough that a completely different
  random stream sails through them. `stepping_is_deterministic` runs each case
  twice and demands the same answer, which is what catches a race introduced by
  letting threads touch more than their own rows. And because collisions
  conserve mass and momentum exactly,
  `thread_count_does_not_change_conserved_quantities` holds those two exactly
  equal across one, two, three, five and eight threads -- an invariant no
  reordering of the random draws can excuse breaking.

So the reading is: layer one alone failing means *you moved the random stream*,
and you should confirm you meant to and regenerate the table. Either of the
others failing means *you broke the simulation*. To regenerate after a
deliberate change:

```bash
cargo test --test golden -- --ignored --nocapture print_golden
```

and paste the printed block over `GOLDEN` in `tests/golden.rs`, saying in the
commit message why the stream moved. The whole file runs in about 1.5 s.

Only re-record the baseline once a change is settled and committed:

```bash
cargo bench -- --save-baseline
```

Other options:

```bash
cargo bench -- step field/sample  # only cases whose name contains these
cargo bench -- --quick            # rough numbers, about 4x faster
cargo bench -- --slow             # also the cases that take seconds each
cargo bench -- --threads 1        # pin the parallel cases to one thread
cargo bench -- --threshold 2      # flag smaller changes than the default 5%
cargo bench -- --list             # case names, and nothing else
cargo bench -- --help
```

Each case is warmed up for 120 ms, then timed over seven samples of about
150 ms each. The headline figure is the **fastest** sample, which is a much
steadier estimator on a laptop than the mean: noise only ever adds time. The
median is printed next to it, so a wide gap between the two is visible as
what it is -- a busy machine -- rather than being folded into the result.

## Where the time actually goes

The point of the suite is not the individual numbers but the budget they add
up to. For the default run -- 2048x1280 cells, 6,000 warmup steps then 40,000
recorded ones, a field sample every 5 steps, a frame every 500 -- the measured
per-call costs predict this:

| Phase | Calls | Each | Total | Share |
| --- | ---: | ---: | ---: | ---: |
| `GpuLattice::advance_sampling` | 46,000 | 0.048 ms | 2.2 s | **59%** |
| Writing frames (PNG + SVG) | 80 | 9.7 ms | 0.8 s | 21% |
| `transport::measure` at startup | 1 | 0.7 s | 0.7 s | 19% |
| `init_equilibrium` | 1 | 18 ms | 0.02 s | <1% |

The step figure is `step/gpu/2048x1280/batched+inlet+sample`, which is the
production configuration: the inflow boundary and a field sample every fifth
step, both on the GPU, batched so the program synchronises once a frame rather
than once a sample. The binary takes 3.1 s against the 3.7 s this predicts,
because a real run puts 500 steps in a command buffer where the benchmark case
puts 100, so it pays the submission a fifth as often.

On `--no-gpu` the same table reads 74 s for `Lattice::step`, 1.5 s for
`Field::sample` and 6.1 s for the startup measurement, and the run takes 1.4
minutes -- which is where this table stood one phase ago, when it said **the
update is 91% of the run**.

And a phase before that it read the other way round again: **coarse-graining
cost twice as much as simulating**, because `Field::sample` walked all 2.6M
cells on one thread, unpacking six direction bits per cell into `f32` adds,
while `Lattice::step` -- which does strictly more work per cell -- spread itself
over all twelve cores. It now looks the cell byte up in a table of packed
integer moments (`src/moments.rs`) and shares the block rows out over the same
threads the update uses, which took it from 22.6 ms to 0.18 ms.
`Lattice::mean_velocity` had the same shape and the same fix, 23.1 ms to
0.51 ms.

So the run is no longer dominated by any one thing, which is the point at which
to stop. `PLAN.md` has the history.

## What each case is for

**`step/*`** -- the fused propagate-and-collide update, which is the only thing
in the program that has to be fast.

* `512x512/t1` vs `2048x1280/t1`: the same kernel in and out of cache. The
  full-size update runs at 244 Mcell/s against 452 in cache, so 54% of the
  in-cache rate is what the memory system leaves it.
* `512x512/tN` vs `512x512/t1`: 3.3x from 12 threads.
* `2048x1280/tN`: the number that decides how long a real run takes.
* `2048x64/tN`: a short, wide lattice pays the per-step `thread::scope` cost
  over far fewer rows, so this is where thread-spawn overhead shows up.
* `t1/no-rest`: the book's six-direction model, whose collision table is half
  the size. That it is not measurably faster says the table is not the
  bottleneck.
* `tN/plate`: solid cells take the other arm of the inner branch.
* `tN/inlet`: the inlet re-seed is serial, so the gap to plain `512x512/tN`
  (178 -> 210 us) is the cost of the one part of `step` that does not scale.
  Those 32 us are for 512 rows; a full-height run pays about 80 us per step,
  which is small against the CPU update and is not small against the GPU one.

**`step/gpu/*`** -- the same update on the GPU, as bitplanes. The four cases
are cumulative, and the gaps between them are the interesting part:

* `one-per-submit` vs `batched`: 0.171 ms against 0.032, the same hundred steps
  submitted one command buffer each and all in one. A round trip costs more
  than five steps do, which is the constraint the whole GPU path is built
  around -- it is why coarse-graining had to become a kernel too.
* `batched+inlet`: 0.046 ms. The inflow boundary is its own dispatch, and costs
  0.013 ms a step almost regardless of how wide it is, because what is being
  paid for is the second dispatch rather than the work. `PLAN.md` has the
  measurements behind that choice; folding it into the step kernel instead cost
  0.025 ms.
* `batched+inlet+sample`: 0.048 ms, the production configuration. Twenty field
  samples across the hundred steps add 0.0025 ms a step, so coarse-graining has
  gone from twice the cost of simulating to five percent of it.

**`field/gpu/sample`, `lattice/gpu/total-particles`** -- the analysis passes on
the device, timed on their own and so paying a full round trip each: about 0.14
and 0.17 ms against a floor of 0.09 ms for an empty submit. Inside a batch the
sample costs 0.0025 ms. These two cases measure the synchronisation, which is
what makes them worth having -- they are the reason the run loop hands a whole
frame's worth of steps to `advance_sampling` rather than calling it per sample.

**`field/*`, `lattice/mean-velocity`, `lattice/total-particles`** -- the
CPU analysis passes, all three of which now read the cell array a word at a
time.
`total_particles` popcounts eight cells per instruction and manages 58.1
Gcell/s; `Field::sample` and `Lattice::mean_velocity` look each cell byte up in
the packed-moment tables in `src/moments.rs` and reach 15.0 and 5.1 Gcell/s.
They used to take the cell apart a bit at a time on a single thread, at 116 and
114 Mcell/s, and the 500x gap between those and `total_particles` was the whole
story of this section. Closing it is most of what took the default run from 4.7
minutes to 1.4. (`field/vorticity` and `field/mean-velocity` work on the 102x64
block grid rather than the lattice, which is why they are microseconds.)

**`render/*`** -- the hand-rolled PNG and SVG encoders. Both are small
absolutely, and only run once per frame.

**`setup/*`** -- once-per-process costs. `transport::measure` at size 64 stands
in for the size-256 measurement the binary actually does; the real one is
`--slow`.

## Reading the results

Run-to-run noise on this machine is around 1% for the single-threaded cases,
2-3% for the multi-threaded ones, and around 10% for `step/2048x64/tN`, which
gives twelve threads only five rows each and so is almost pure thread-scoping
overhead, at the mercy of the scheduler. The render cases
touch the filesystem and drift by a few percent too.

The suite flags a change of 5% or more (`--threshold` moves the line). Treat a
single flagged multi-threaded result as a hint and re-run it before believing
it; a change that matters will show up in the single-threaded cases as well,
and those are worth trusting at the 1-2% level.

## Recorded baseline

Apple M3 Pro, 12 cores, macOS, rustc 1.97.1, 2026-09-09. Best of seven
samples; throughput counts lattice cells updated or touched per second. The
`step/gpu/*batched*` cases each cover 100 steps, so divide by 100 for a
per-step figure.

| Case | Time | Throughput |
| --- | ---: | ---: |
| `step/512x512/t1` | 580.49 us | 451.6 Mcell/s |
| `step/512x512/tN` | 177.85 us | 1473.9 Mcell/s |
| `step/512x512/t1/no-rest` | 580.09 us | 451.9 Mcell/s |
| `step/512x512/tN/plate` | 178.00 us | 1472.7 Mcell/s |
| `step/512x512/tN/inlet` | 210.08 us | 1247.8 Mcell/s |
| `step/2048x1280/t1` | 10.751 ms | 243.8 Mcell/s |
| `step/2048x1280/tN` | 1.475 ms | 1776.7 Mcell/s |
| `step/2048x64/tN` | 127.01 us | 1032.0 Mcell/s |
| `step/gpu/2048x1280/one-per-submit` | 168.25 us | 15580.5 Mcell/s |
| `step/gpu/2048x1280/batched` | 3.228 ms | 81213.5 Mcell/s |
| `step/gpu/2048x1280/batched+inlet` | 4.520 ms | 58000.4 Mcell/s |
| `step/gpu/2048x1280/batched+inlet+sample` | 4.765 ms | 55009.3 Mcell/s |
| `field/gpu/sample/2048x1280` | 139.64 us | 18772.8 Mcell/s |
| `lattice/gpu/total-particles/2048x1280` | 165.16 us | 15871.8 Mcell/s |
| `field/sample/2048x1280` | 175.28 us | 14955.5 Mcell/s |
| `field/vorticity/2048x1280` | 9.16 us |  |
| `field/mean-velocity/2048x1280` | 4.85 us |  |
| `lattice/total-particles/2048x1280` | 45.08 us | 58148.6 Mcell/s |
| `lattice/mean-velocity/2048x1280` | 512.67 us | 5113.3 Mcell/s |
| `render/write-vorticity/png` | 6.661 ms | 54.9 Mpx/s |
| `render/write-arrows/svg` | 3.053 ms |  |
| `collision/build/rest` | 7.17 us |  |
| `collision/build/no-rest` | 3.19 us |  |
| `lattice/init-equilibrium/2048x1280` | 17.562 ms | 149.3 Mcell/s |
| `transport/measure/64` | 661.232 ms |  |

Numbers taken on a different machine are not comparable to these; re-record the
baseline before using it, and say in the commit message which machine it came
from.
