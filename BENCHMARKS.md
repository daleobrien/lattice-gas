# Benchmarks

A dependency-free benchmark suite lives in [`benches/speed.rs`](benches/speed.rs),
and the numbers it produced on the machine it was written on are recorded in
[`benches/baseline.txt`](benches/baseline.txt). `cargo bench` reads that file
and reports the change against it, so the loop for a performance change is:

```bash
cargo bench                       # where are we now, against the baseline?
# ... make the change ...
cargo test                        # the physics must still hold
cargo bench                       # what moved, and by how much?
```

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
per-call costs predict about 4.7 minutes of wall time, spent like this:

| Phase | Calls | Each | Total | Share |
| --- | ---: | ---: | ---: | ---: |
| `Field::sample` | 8,001 | 22.6 ms | 181 s | **64%** |
| `Lattice::step` | 46,000 | 2.02 ms | 93 s | 33% |
| `transport::measure` at startup | 1 | 5.4 s | 5.4 s | 2% |
| Writing frames (PNG + SVG) | 80 | 9.2 ms | 0.7 s | <1% |
| `init_equilibrium` | 1 | 29 ms | 0.03 s | <1% |

A 1,000-step run of the binary takes 13.9 s against the 12.4 s this predicts,
so the model is close enough to plan against.

The headline is that **coarse-graining costs twice as much as simulating**.
`Field::sample` walks all 2.6M cells single-threaded, unpacking six direction
bits per cell into `f32` adds, while `Lattice::step` -- which does strictly
more work per cell -- spreads itself over all twelve cores. It runs every fifth
step by default, so the 5:1 step advantage does not come close to paying for
the 12:1 parallelism gap. `Lattice::mean_velocity` (23.1 ms) has the same shape
and the same problem, though it is only called for the status line.

## What each case is for

**`step/*`** -- the fused propagate-and-collide update, which is the only thing
in the program that has to be fast.

* `512x512/t1` vs `2048x1280/t1`: the same kernel in and out of cache. The
  full-size update runs at 243 Mcell/s against 429 in cache, so 57% of the
  in-cache rate is what the memory system leaves it.
* `512x512/tN` vs `512x512/t1`: 3.2x from 12 threads.
* `2048x1280/tN`: the number that decides how long a real run takes.
* `2048x64/tN`: a short, wide lattice pays the per-step `thread::scope` cost
  over far fewer rows, so this is where thread-spawn overhead shows up.
* `t1/no-rest`: the book's six-direction model, whose collision table is half
  the size. That it is not measurably faster says the table is not the
  bottleneck.
* `tN/plate`: solid cells take the other arm of the inner branch.
* `tN/inlet`: the inlet re-seed is serial, so the gap to plain `512x512/tN`
  (190 -> 241 us) is the cost of the one part of `step` that does not scale.

**`field/*`, `lattice/mean-velocity`, `lattice/total-particles`** -- the
analysis passes. `total_particles` is a plain `count_ones` over the cell array
and manages 58.5 Gcell/s. `Field::sample` and `Lattice::mean_velocity` read the
same array, byte for byte, 500x slower; that gap is the whole story of this
section. (`field/vorticity` and `field/mean-velocity` work on the 102x64 block
grid rather than the lattice, which is why they are microseconds.)

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

Apple M3 Pro, 12 cores, macOS, rustc 1.97.1, 2026-09-08. Best of seven
samples; throughput counts lattice cells updated or touched per second.

| Case | Time | Throughput |
| --- | ---: | ---: |
| `step/512x512/t1` | 610.48 us | 429.4 Mcell/s |
| `step/512x512/tN` | 190.24 us | 1378.0 Mcell/s |
| `step/512x512/t1/no-rest` | 614.81 us | 426.4 Mcell/s |
| `step/512x512/tN/plate` | 189.78 us | 1381.3 Mcell/s |
| `step/512x512/tN/inlet` | 241.05 us | 1087.5 Mcell/s |
| `step/2048x1280/t1` | 10.805 ms | 242.6 Mcell/s |
| `step/2048x1280/tN` | 2.022 ms | 1296.7 Mcell/s |
| `step/2048x64/tN` | 124.01 us | 1056.9 Mcell/s |
| `field/sample/2048x1280` | 22.572 ms | 116.1 Mcell/s |
| `field/vorticity/2048x1280` | 8.99 us | |
| `field/mean-velocity/2048x1280` | 4.53 us | |
| `lattice/total-particles/2048x1280` | 44.80 us | 58515.8 Mcell/s |
| `lattice/mean-velocity/2048x1280` | 23.063 ms | 113.7 Mcell/s |
| `render/write-vorticity/png` | 6.244 ms | 58.5 Mpx/s |
| `render/write-arrows/svg` | 2.969 ms | |
| `collision/build/rest` | 9.02 us | |
| `collision/build/no-rest` | 3.48 us | |
| `lattice/init-equilibrium/2048x1280` | 29.100 ms | 90.1 Mcell/s |
| `transport/measure/64` | 261.43 ms | |

Numbers taken on a different machine are not comparable to these; re-record the
baseline before using it, and say in the commit message which machine it came
from.
