//! Benchmarks for the lattice gas.
//!
//! Run everything and compare against the recorded baseline:
//!
//!     cargo bench
//!
//! Run one group, or anything whose name contains a substring:
//!
//!     cargo bench -- step field/sample
//!
//! Record the current machine as the new baseline:
//!
//!     cargo bench -- --save-baseline
//!
//! The suite is deliberately dependency-free. Each case is warmed up, then
//! timed over several samples; the headline number is the *fastest* sample,
//! which on a busy laptop is a far steadier estimator than the mean. The
//! median is printed alongside so that a wide gap between the two (a noisy
//! machine) is visible rather than silently folded into the result.

use lattice_gas::collision::CollisionTable;
use lattice_gas::hex::SQRT3_2;
use lattice_gas::lattice::Lattice;
use lattice_gas::render::{self, Field};
use lattice_gas::transport;

use std::collections::HashMap;
use std::hint::black_box;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Lattice geometry used by the "production" cases: the binary's own defaults.
const BIG: (usize, usize) = (2048, 1280);
/// A smaller lattice, for cases where the point is the per-cell cost rather
/// than the cache behaviour of a full-size run.
const SMALL: (usize, usize) = (512, 512);

const DENSITY: f32 = 0.22;
const SPEED: f32 = 0.4;
const BLOCK: usize = 20;
const SEED: u64 = 0x5EED;

const DEFAULT_BASELINE: &str = "benches/baseline.txt";

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// What one iteration of a case gets through, for the throughput column.
#[derive(Clone, Copy)]
struct Unit {
    per_iter: f64,
    scale: f64,
    label: &'static str,
}

/// Cell updates, reported in millions per second.
fn cells(n: usize) -> Option<Unit> {
    Some(Unit { per_iter: n as f64, scale: 1e6, label: "Mcell/s" })
}

/// Bytes touched, reported in MiB per second.
fn pixels(n: usize) -> Option<Unit> {
    Some(Unit { per_iter: n as f64, scale: 1e6, label: "Mpx/s" })
}

struct Record {
    name: String,
    best_ns: f64,
    median_ns: f64,
    iters: u64,
    unit: Option<Unit>,
}

struct Opts {
    filters: Vec<String>,
    samples: usize,
    warmup: Duration,
    sample_time: Duration,
    save_baseline: bool,
    baseline: PathBuf,
    compare: bool,
    slow: bool,
    list: bool,
    threads: usize,
    /// Percentage change against the baseline worth pointing at.
    threshold: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            filters: Vec::new(),
            samples: 7,
            warmup: Duration::from_millis(120),
            sample_time: Duration::from_millis(150),
            save_baseline: false,
            baseline: PathBuf::from(DEFAULT_BASELINE),
            compare: true,
            slow: false,
            list: false,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            threshold: 5.0,
        }
    }
}

struct Runner {
    opts: Opts,
    records: Vec<Record>,
    baseline: HashMap<String, f64>,
    group: &'static str,
    group_shown: bool,
    tty: bool,
}

impl Runner {
    fn selected(&self, name: &str) -> bool {
        self.opts.filters.is_empty() || self.opts.filters.iter().any(|f| name.contains(f))
    }

    fn group(&mut self, name: &'static str) {
        self.group = name;
        self.group_shown = false;
    }

    /// Time `body`. `setup` runs once, outside the clock, and its result is
    /// handed to every iteration; state that evolves as the benchmark runs
    /// (a lattice being stepped, say) is exactly what we want to measure.
    fn bench<T, R>(
        &mut self,
        name: &str,
        unit: Option<Unit>,
        setup: impl FnOnce() -> T,
        mut body: impl FnMut(&mut T) -> R,
    ) {
        if !self.selected(name) {
            return;
        }
        if self.opts.list {
            println!("{name}");
            return;
        }
        if !self.group_shown {
            println!("\n[{}]", self.group);
            self.group_shown = true;
        }

        // A running case can take a few seconds, so say what is in flight --
        // but only when someone is watching, since a carriage return in a
        // redirected log just leaves both lines behind.
        if self.tty {
            print!("{name:<38} ...");
            let _ = std::io::stdout().flush();
        }

        let mut state = setup();

        // Warm up caches, branch predictors and the CPU's clock, and learn
        // roughly how long one iteration takes.
        let warm = Instant::now();
        let mut warm_iters = 0u64;
        while warm.elapsed() < self.opts.warmup {
            black_box(body(black_box(&mut state)));
            warm_iters += 1;
        }
        let per_iter = warm.elapsed().as_secs_f64() / warm_iters.max(1) as f64;
        let iters = ((self.opts.sample_time.as_secs_f64() / per_iter).round() as u64).max(1);

        let mut samples = Vec::with_capacity(self.opts.samples);
        for _ in 0..self.opts.samples {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(body(black_box(&mut state)));
            }
            samples.push(t.elapsed().as_secs_f64() / iters as f64 * 1e9);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let rec = Record {
            name: name.to_string(),
            best_ns: samples[0],
            median_ns: samples[samples.len() / 2],
            iters,
            unit,
        };
        if self.tty {
            print!("\r");
        }
        println!("{}", self.format(&rec));
        self.records.push(rec);
    }

    fn format(&self, r: &Record) -> String {
        let thr = match r.unit {
            Some(u) => format!(
                "{:>10.1} {}",
                u.per_iter / (r.best_ns / 1e9) / u.scale,
                u.label
            ),
            None => String::new(),
        };
        let delta = match self.baseline.get(&r.name) {
            Some(&base) => {
                let pct = (r.best_ns - base) / base * 100.0;
                let tag = if pct <= -self.opts.threshold {
                    "faster"
                } else if pct >= self.opts.threshold {
                    "SLOWER"
                } else {
                    ""
                };
                format!("  {:>+7.1}% {}", pct, tag)
            }
            None => String::new(),
        };
        format!(
            "{:<38} {:>11}  (median {:>11}, n={:<5}){:<20}{}",
            r.name,
            time(r.best_ns),
            time(r.median_ns),
            r.iters,
            thr,
            delta
        )
    }
}

fn time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} us", ns / 1e3)
    } else if ns < 1_000_000_000.0 {
        format!("{:.3} ms", ns / 1e6)
    } else {
        format!("{:.3} s", ns / 1e9)
    }
}

// ---------------------------------------------------------------------------
// Baseline file
// ---------------------------------------------------------------------------

/// Whatever we can cheaply learn about the machine, so that a baseline file
/// carries enough context to say whether it is comparable to the run in hand.
fn host_info() -> Vec<String> {
    fn cmd(program: &str, args: &[&str]) -> Option<String> {
        let out = std::process::Command::new(program).args(args).output().ok()?;
        let s = String::from_utf8(out.stdout).ok()?;
        let s = s.trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    }

    let mut v = Vec::new();
    let cpu = cmd("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| {
            cmd("sh", &["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
        })
        .unwrap_or_else(|| "unknown".into());
    v.push(format!(
        "cpu: {cpu} ({} cores, {}-{})",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        std::env::consts::OS,
        std::env::consts::ARCH,
    ));
    v.push(format!(
        "rustc: {}",
        cmd("rustc", &["--version"]).unwrap_or_else(|| "unknown".into())
    ));
    v
}

fn read_baseline(path: &Path) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split_whitespace();
        if let (Some(name), Some(ns)) = (f.next(), f.next()) {
            if let Ok(ns) = ns.parse::<f64>() {
                out.insert(name.to_string(), ns);
            }
        }
    }
    out
}

fn write_baseline(path: &Path, records: &[Record], opts: &Opts) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("# lattice-gas benchmark baseline: nanoseconds per iteration.\n");
    s.push_str("#\n");
    s.push_str("# `cargo bench` reads this file and reports the change against it.\n");
    s.push_str("# Re-record it with `cargo bench -- --save-baseline`, on an idle\n");
    s.push_str("# machine, and note in the commit message what the machine was:\n");
    s.push_str("# these numbers are only comparable to others taken on the same one.\n");
    s.push_str("#\n");
    for line in host_info() {
        s.push_str(&format!("# {line}\n"));
    }
    s.push_str(&format!("# threads used: {}\n", opts.threads));
    s.push_str("#\n# name                                        ns/iter    throughput\n");
    for r in records {
        let thr = match r.unit {
            Some(u) => format!(
                "{:.1} {}",
                u.per_iter / (r.best_ns / 1e9) / u.scale,
                u.label
            ),
            None => "-".to_string(),
        };
        s.push_str(&format!("{:<40} {:>14.1}    {}\n", r.name, r.best_ns, thr));
    }
    std::fs::write(path, s)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A lattice matching how the binary actually sets one up: uniform inflow
/// equilibrium everywhere, optionally with the default flat plate in the way.
fn lattice(
    (w, h): (usize, usize),
    threads: usize,
    rest: bool,
    inlet_cols: usize,
    plate: bool,
) -> Lattice {
    let mut lat = Lattice::new(w, h, DENSITY, (SPEED, 0.0), inlet_cols, rest, threads, SEED);
    if plate {
        let yc = h as f32 * SQRT3_2 / 2.0;
        // The binary's defaults, scaled to whatever lattice we were given.
        let size = h as f32 * 173.0 / 1280.0;
        let (x0, x1) = (w as f32 / 5.0, w as f32 / 5.0 + 4.0);
        let (y0, y1) = (yc - size / 2.0, yc + size / 2.0);
        lat.add_solid(move |x, y| x >= x0 && x <= x1 && y >= y0 && y <= y1);
    }
    lat.init_equilibrium();
    lat
}

/// A coarse-grained field carrying a real flow, for the output benchmarks.
fn field(dims: (usize, usize), threads: usize) -> (Lattice, Field) {
    let mut lat = lattice(dims, threads, true, 8, true);
    lat.advance(50);
    let mut f = Field::new(&lat, BLOCK, BLOCK);
    f.sample(&lat, 1.0);
    (lat, f)
}

fn scratch_dir() -> PathBuf {
    let d = std::env::temp_dir().join("lattice-gas-bench");
    std::fs::create_dir_all(&d).expect("cannot create scratch directory");
    d
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let baseline = if opts.compare && !opts.save_baseline {
        read_baseline(&opts.baseline)
    } else {
        HashMap::new()
    };
    let have_baseline = !baseline.is_empty();

    let nth = opts.threads;
    let list = opts.list;
    let save = opts.save_baseline;
    let slow = opts.slow;
    let baseline_path = opts.baseline.clone();

    let mut r = Runner {
        opts,
        records: Vec::new(),
        baseline,
        group: "",
        group_shown: false,
        tty: std::io::stdout().is_terminal(),
    };

    if !list {
        println!(
            "lattice-gas benchmarks -- {nth} threads{}\n",
            if have_baseline { ", comparing against the recorded baseline" } else { "" }
        );
    }

    // --- the core update ---------------------------------------------------
    //
    // Everything else in a run is bookkeeping around this loop.
    r.group("step");

    let (sw, sh) = SMALL;
    let (bw, bh) = BIG;

    r.bench(
        "step/512x512/t1",
        cells(sw * sh),
        || lattice(SMALL, 1, true, 0, false),
        |l| l.step(),
    );
    r.bench(
        "step/512x512/tN",
        cells(sw * sh),
        || lattice(SMALL, nth, true, 0, false),
        |l| l.step(),
    );
    // The book's original six-direction model: a 64-entry collision table
    // instead of 128, so the difference is the table's cache footprint.
    r.bench(
        "step/512x512/t1/no-rest",
        cells(sw * sh),
        || lattice(SMALL, 1, false, 0, false),
        |l| l.step(),
    );
    // A solid obstacle turns a predictable branch into an unpredictable one.
    r.bench(
        "step/512x512/tN/plate",
        cells(sw * sh),
        || lattice(SMALL, nth, true, 0, true),
        |l| l.step(),
    );
    // The inlet re-seed is serial, so it is pure overhead on top of the
    // parallel update; the gap to step/512x512/tN is its cost.
    r.bench(
        "step/512x512/tN/inlet",
        cells(sw * sh),
        || lattice(SMALL, nth, true, 8, false),
        |l| l.step(),
    );
    // Production geometry: 2.6M cells is well past any cache, so this is the
    // number that decides how long a real run takes.
    r.bench(
        "step/2048x1280/t1",
        cells(bw * bh),
        || lattice(BIG, 1, true, 0, false),
        |l| l.step(),
    );
    r.bench(
        "step/2048x1280/tN",
        cells(bw * bh),
        || lattice(BIG, nth, true, 0, false),
        |l| l.step(),
    );
    // Thread scoping happens once per step, so a short, wide lattice pays it
    // more often per cell than a tall one. This is where that shows up.
    r.bench(
        "step/2048x64/tN",
        cells(bw * 64),
        || lattice((bw, 64), nth, true, 0, false),
        |l| l.step(),
    );

    // --- the GPU path ------------------------------------------------------
    //
    // A round trip to the GPU costs more than a step does, so the two cases
    // below are the same work submitted two ways: one step per command buffer,
    // and a hundred. The gap between them is the whole design constraint.
    #[cfg(target_os = "macos")]
    {
        use lattice_gas::gpu::GpuLattice;
        if lattice_gas::metal::Device::new().is_some() {
            let build = |steps: u64| {
                move || {
                    let seed = lattice_gas::lattice::Lattice::new(
                        bw, bh, DENSITY, (SPEED, 0.0), 0, true, 1, SEED,
                    );
                    let mut seed_cells = seed;
                    seed_cells.init_equilibrium();
                    let mut g = GpuLattice::new(bw, bh, true, SEED).expect("gpu lattice");
                    g.load(&seed_cells.cells);
                    (g, steps)
                }
            };
            r.bench(
                "step/gpu/2048x1280/one-per-submit",
                cells(bw * bh),
                build(1),
                |(g, n)| g.advance(*n),
            );
            r.bench(
                "step/gpu/2048x1280/batched",
                cells(bw * bh * 100),
                build(100),
                |(g, n)| g.advance(*n),
            );
            // The size `transport::measure` works at. 2,048 threads is far too
            // few to fill this GPU, so the case exists to show what the step
            // costs when it is launch latency rather than memory bandwidth.
            r.bench(
                "step/gpu/256x256/batched",
                cells(256 * 256 * 100),
                || {
                    let mut g = GpuLattice::new(256, 256, true, SEED).expect("gpu lattice");
                    let mut seed = Lattice::new(256, 256, DENSITY, (SPEED, 0.0), 0, true, 1, SEED);
                    seed.init_equilibrium();
                    g.load(&seed.cells);
                    (g, 100u64)
                },
                |(g, n)| g.advance(*n),
            );
            // The two things a real run adds to a bare step: the inflow
            // boundary, which is folded into the step kernel, and a field
            // sample every fifth step. Both are meant to disappear into the
            // step; these two cases are how that claim gets checked.
            r.bench(
                "step/gpu/2048x1280/batched+inlet",
                cells(bw * bh * 100),
                || {
                    let (mut g, n) = build(100)();
                    g.set_inlet(8, DENSITY, (SPEED, 0.0), true);
                    (g, n)
                },
                |(g, n)| g.advance(*n),
            );
            r.bench(
                "step/gpu/2048x1280/batched+inlet+sample",
                cells(bw * bh * 100),
                || {
                    let (mut g, n) = build(100)();
                    g.set_inlet(8, DENSITY, (SPEED, 0.0), true);
                    g.attach_field(BLOCK, BLOCK);
                    (g, n)
                },
                |(g, n)| g.advance_sampling(*n, 5, 0.2),
            );
            r.bench(
                "field/gpu/sample/2048x1280",
                cells(bw * bh),
                || {
                    let (mut g, _) = build(1)();
                    g.attach_field(BLOCK, BLOCK);
                    g
                },
                |g| g.sample_field(0.2),
            );
            // Unpacking the planes back to one byte per cell. Nothing in a
            // run needs this, but `transport::measure` projects the lattice
            // onto a Fourier mode every few steps and went through here to
            // do it, so its cost decided the shape of that code.
            r.bench(
                "lattice/gpu/store/2048x1280",
                cells(bw * bh),
                || (build(1)().0, vec![0u8; bw * bh]),
                |(g, cells)| g.store(cells),
            );
            r.bench(
                "lattice/gpu/total-particles/2048x1280",
                cells(bw * bh),
                || build(1)().0,
                |g| {
                    black_box(g.total_particles());
                },
            );
        }
    }

    // --- coarse-graining and analysis --------------------------------------
    //
    // `Field::sample` runs every `--sample-every` steps (5 by default), so its
    // cost is multiplied by a fifth of the step count in a real run.
    r.group("field");

    r.bench(
        "field/sample/2048x1280",
        cells(bw * bh),
        || field(BIG, nth),
        |(lat, f)| f.sample(lat, 0.2),
    );
    r.bench(
        "field/vorticity/2048x1280",
        None,
        || field(BIG, nth).1,
        |f| f.vorticity(),
    );
    r.bench(
        "field/mean-velocity/2048x1280",
        None,
        || field(BIG, nth).1,
        |f| f.mean_velocity(),
    );
    r.bench(
        "lattice/total-particles/2048x1280",
        cells(bw * bh),
        || lattice(BIG, nth, true, 0, false),
        |l| l.total_particles(),
    );
    r.bench(
        "lattice/mean-velocity/2048x1280",
        cells(bw * bh),
        || lattice(BIG, nth, true, 0, false),
        |l| l.mean_velocity(),
    );

    // --- writing frames ----------------------------------------------------
    //
    // Once per `--frame-every` steps, but both encoders are hand-rolled and
    // neither has ever been looked at, so they are worth a number.
    r.group("render");

    {
        let scale = 8usize;
        let px = (bw / BLOCK) * scale;
        let py = (bh / BLOCK) * ((scale as f32 * SQRT3_2).round() as usize);
        let dir = scratch_dir();
        let png = dir.join("bench-vorticity.png");
        let svg = dir.join("bench-arrows.svg");

        r.bench(
            "render/write-vorticity/png",
            pixels(px * py),
            || (field(BIG, nth).1, png.clone()),
            |(f, path)| render::write_vorticity(path, f, scale, 0.01).expect("png write failed"),
        );
        r.bench(
            "render/write-arrows/svg",
            None,
            || (field(BIG, nth).1, svg.clone()),
            |(f, path)| render::write_arrows(path, f, 1.6, (SPEED, 0.0)).expect("svg write failed"),
        );
    }

    // --- one-off startup costs ---------------------------------------------
    r.group("setup");

    r.bench("collision/build/rest", None, || (), |_| CollisionTable::build(true));
    r.bench("collision/build/no-rest", None, || (), |_| CollisionTable::build(false));
    r.bench(
        "lattice/init-equilibrium/2048x1280",
        cells(bw * bh),
        || Lattice::new(bw, bh, DENSITY, (SPEED, 0.0), 8, true, nth, SEED),
        |l| l.init_equilibrium(),
    );
    // Every run measures its own viscosity before it starts. The binary uses
    // size 256; that costs seconds, so it is behind --slow.
    r.bench(
        "transport/measure/64",
        None,
        || (),
        |_| transport::measure(DENSITY, true, nth, SEED, 64, false),
    );
    if slow {
        r.bench(
            "transport/measure/256",
            None,
            || (),
            |_| transport::measure(DENSITY, true, nth, SEED, 256, false),
        );
    }

    if list {
        return;
    }

    if save {
        match write_baseline(&baseline_path, &r.records, &r.opts) {
            Ok(()) => println!("\nbaseline written to {}", baseline_path.display()),
            Err(e) => eprintln!("\ncould not write {}: {e}", baseline_path.display()),
        }
    } else if !have_baseline {
        println!(
            "\nno baseline at {} -- record one with `cargo bench -- --save-baseline`",
            baseline_path.display()
        );
    }
}

const USAGE: &str = "\
lattice-gas benchmarks

  cargo bench                        run everything, compare to the baseline
  cargo bench -- step                run only cases whose name contains 'step'
  cargo bench -- --save-baseline     record the current numbers as the baseline

  --save-baseline      overwrite the baseline file with this run's results
  --baseline PATH      use a different baseline file
  --no-baseline        do not compare against a baseline
  --quick              fewer, shorter samples: rough numbers, ~4x faster
  --slow               also run the cases that take seconds each
  --samples N          timed samples per case          (default 7)
  --sample-time MS     target duration of one sample   (default 150)
  --threads N          threads for the parallel cases  (default: all cores)
  --threshold PCT      flag changes at least this large (default 5)
  --list               print the case names and exit
  -h, --help
";

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            // Cargo passes this through to `harness = false` targets.
            "--bench" => {}
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--save-baseline" => o.save_baseline = true,
            "--baseline" => o.baseline = PathBuf::from(val()?),
            "--no-baseline" => o.compare = false,
            "--slow" => o.slow = true,
            "--list" => o.list = true,
            "--quick" => {
                o.samples = 3;
                o.warmup = Duration::from_millis(40);
                o.sample_time = Duration::from_millis(50);
            }
            "--samples" => o.samples = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sample-time" => {
                let ms: u64 = val()?.parse().map_err(|e| format!("{e}"))?;
                o.sample_time = Duration::from_millis(ms);
            }
            "--threads" => o.threads = val()?.parse().map_err(|e| format!("{e}"))?,
            "--threshold" => o.threshold = val()?.parse().map_err(|e| format!("{e}"))?,
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other => o.filters.push(other.to_string()),
        }
    }
    if o.samples == 0 {
        return Err("--samples must be at least 1".into());
    }
    Ok(o)
}
