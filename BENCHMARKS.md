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
| `GpuLattice::advance_sampling` | 46,000 | 0.037 ms | 1.70 s | **81%** |
| `transport::measure` at startup | 1 | 0.30 s | 0.30 s | 14% |
| `init_equilibrium` and setup | 1 | 0.11 s | 0.11 s | 5% |
| Writing frames (PNG + SVG) | 80 | 9.5 ms | 0.76 s | *overlapped* |

The step figure is `step/gpu/2048x1280/batched+inlet+sample`, which is the
production configuration: the inflow boundary and a field sample every fifth
step, both on the GPU, batched so the program synchronises once a frame rather
than once a sample.

Frame encoding is in the table but not in the total, because it does not happen
on the critical path: the frames go to a pool of writer threads while the GPU
carries on stepping. The binary reports 1.7 s of simulation against the 1.70 s
the step alone predicts, so all 0.76 s of deflate and SVG formatting is hidden.
Wall time is 2.0 s including startup.

**What limits the step depends on how big the lattice is**, and the answer at
the production size is not what a bandwidth calculation suggests. It moves a
compulsory 4.92 MB a step --- seven planes and the solid plane read, seven
written --- which at 0.0330 ms works out at 152 GB/s, against 140 GB/s for a
bare streaming copy measured on this machine. That looks like the memory wall
and is not: 4.6 MB of buffers never leaves cache, so the comparison is against
a rate the step never has to sustain.

Two measurements say what is really going on. Replacing the collision with the
identity, so the kernel only propagates and writes, makes it **2.2 to 2.5 times
faster** at every size that fits in cache. And throughput *rises* with the
lattice --- 61 Gcell/s at 1024x1024, 79 at the production size, 81 at
2048x2048, 93 at 4096x2048 --- which is a kernel wanting more threads to hide
latency behind, not one starved of bandwidth.

Only the largest lattices are memory-bound. At 8192x5120 the buffers are 73 MB,
the step takes 0.587 ms, and that is 134 GB/s: the streaming rate, to within
the error on it.

Two things follow from that split, and both were measured rather than assumed.
The obstacle plane is one of the fifteen words a step moves per lattice word,
and at any size the obstacle is a few hundred words out of a million --- so the
step keeps a bitmap with one bit per word saying whether there is any obstacle
in it, and skips the load otherwise. Thirty-two threads share one load of the
bitmap and it stays in cache. That is worth about 4%, at both 2048x1280 and
8192x5120. It was *predicted* to be worth 7%, and is not, because the obstacle
plane never changes and so was largely cache-resident rather than being the
compulsory traffic the estimate assumed.

The other is coarse-graining, which costs 10% of a step at 8192x5120 against
7% at the production size --- again because at the larger size its reads add to
a memory-bound kernel instead of hiding behind a compute-bound one. Sampling
less often is the obvious answer and is not free; see `--sample-every` below.

So at the sizes a run actually uses, the step is **compute-bound**, and the
685-operator collision circuit is the cost. Its *shape* is not, though ---
emitting each state's minterm standalone instead of sharing a prefix tree
measures the same to half a percent, because the Metal compiler finds the
sharing either way. That is the same negative result LLVM gave on the CPU.

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

* `one-per-submit` vs `batched`: 0.158 ms against 0.033, the same hundred steps
  submitted one command buffer each and all in one. A round trip costs about
  four steps, which is the constraint the whole GPU path is built around -- it
  is why coarse-graining had to become a kernel too. Note that `batched` runs
  with no inlet, and pays 2.4% for the remapped thread index it does not use;
  the case below is the one to compare a real run against.
* `batched+inlet`: 0.034 ms. The inflow boundary lives inside the step kernel,
  and the thread index is handed out so that its words come first --- which
  costs 0.0014 ms a step, where the two obvious arrangements cost 0.013 and
  0.021. `PLAN.md` has the three measurements and why they differ; the short
  version is that both of the obvious ones charge for overhead rather than
  work, and neither varies with the width of the inlet.
* `batched+inlet+sample`: 0.037 ms, the production configuration. Twenty field
  samples across the hundred steps add 0.0026 ms a step, so coarse-graining has
  gone from twice the cost of simulating to seven percent of it.

**`step/gpu/256x256`** -- the size `transport::measure` works at, and the
reason it is a separate line in the budget above. 2,048 threads is eight times
short of filling this GPU, so a step there costs 6.8 us against 33 us for one
forty times the size: it is paying launch latency, not moving memory. Since it
is latency-bound, the two transport measurements simply run on two threads and
interleave, which halves them.

**`lattice/gpu/store`** -- unpacking the planes back to one byte per cell.
Nothing in a run needs it, but the transport measurement projects the lattice
onto a Fourier mode every few steps and comes through here. Taking a word at a
time rather than a cell at a time -- the eight planes that describe a cell also
describe the thirty-one beside it -- took it from 451 to 2,611 Mcell/s.

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

**The GPU cases drift with the machine's power state**, by more than the 5%
threshold, and for a long time after sustained load. After three back-to-back
25-second full-lattice runs every GPU case read 4-7% above its recorded
baseline, including cases whose code had not changed. So for anything on the
GPU, do not trust a comparison against the baseline file: measure the two
versions *interleaved*, in one sitting, and compare those. Doing exactly that
turned an apparent 22% regression into a real 4% improvement.

`step/gpu/2048x1280/one-per-submit` is the noisiest case in the suite and the
one to distrust: it submits a single small dispatch and waits, so it is almost
all submission latency, and it lands bimodally at either about 158 or about 191
us depending on what power state the GPU is in. Both readings are reproducible
several times in a row. Take a flag on that case as a hint about the machine
rather than about the code.

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
| `step/512x512/t1` | 568.72 us | 460.9 Mcell/s |
| `step/512x512/tN` | 186.36 us | 1406.6 Mcell/s |
| `step/512x512/t1/no-rest` | 566.53 us | 462.7 Mcell/s |
| `step/512x512/tN/plate` | 182.35 us | 1437.6 Mcell/s |
| `step/512x512/tN/inlet` | 219.48 us | 1194.4 Mcell/s |
| `step/2048x1280/t1` | 10.601 ms | 247.3 Mcell/s |
| `step/2048x1280/tN` | 1.490 ms | 1759.3 Mcell/s |
| `step/2048x64/tN` | 117.10 us | 1119.4 Mcell/s |
| `step/gpu/2048x1280/one-per-submit` | 183.83 us | 14260.5 Mcell/s |
| `step/gpu/2048x1280/batched` | 3.434 ms | 76339.7 Mcell/s |
| `step/gpu/256x256/batched` | 834.60 us | 7852.4 Mcell/s |
| `step/gpu/4096x2048/batched` | 11.109 ms | 75511.8 Mcell/s |
| `step/gpu/2048x1280/batched+inlet` | 3.591 ms | 73009.7 Mcell/s |
| `step/gpu/2048x1280/batched+inlet+sample` | 3.770 ms | 69532.0 Mcell/s |
| `field/gpu/sample/2048x1280` | 150.69 us | 17396.3 Mcell/s |
| `lattice/gpu/store/2048x1280` | 998.14 us | 2626.3 Mcell/s |
| `lattice/gpu/total-particles/2048x1280` | 165.75 us | 15815.9 Mcell/s |
| `field/sample/2048x1280` | 171.35 us | 15298.4 Mcell/s |
| `field/vorticity/2048x1280` | 8.90 us |  |
| `field/mean-velocity/2048x1280` | 4.75 us |  |
| `lattice/total-particles/2048x1280` | 44.31 us | 59166.1 Mcell/s |
| `lattice/mean-velocity/2048x1280` | 498.25 us | 5261.3 Mcell/s |
| `render/write-vorticity/png` | 6.490 ms | 56.3 Mpx/s |
| `render/write-arrows/svg` | 2.988 ms |  |
| `collision/build/rest` | 7.10 us |  |
| `collision/build/no-rest` | 3.16 us |  |
| `lattice/init-equilibrium/2048x1280` | 17.119 ms | 153.1 Mcell/s |
| `transport/measure/64` | 679.250 ms |  |

Numbers taken on a different machine are not comparable to these; re-record the
baseline before using it, and say in the commit message which machine it came
from.
