//! Command-line driver for the hexagonal lattice gas. The simulation itself
//! lives in the `lattice_gas` library alongside this binary.

use lattice_gas::hex::SQRT3_2;
use lattice_gas::lattice::Lattice;
use lattice_gas::render::{self, Field};
use lattice_gas::sim::Sim;
use lattice_gas::transport;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone)]
struct Config {
    width: usize,
    height: usize,
    density: f32,
    speed: f32,
    steps: u64,
    warmup: u64,
    obstacle: Obstacle,
    obstacle_x: f32,
    size: f32,
    block: usize,
    out: PathBuf,
    frame_every: u64,
    sample_every: u64,
    alpha: f32,
    scale: usize,
    arrow_scale: f32,
    rest_particles: bool,
    threads: usize,
    gpu: bool,
    seed: u64,
    nu: Option<f32>,
    quiet: bool,
    preview: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Obstacle {
    Plate,
    Cylinder,
    None,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            width: 2048,
            height: 1280,
            density: 0.22,
            speed: 0.4,
            steps: 40_000,
            warmup: 6_000,
            obstacle: Obstacle::Plate,
            obstacle_x: 0.0, // filled in as width/5
            size: 173.0,
            block: 20,
            out: PathBuf::from("out"),
            frame_every: 500,
            sample_every: 5,
            alpha: 0.2,
            scale: 8,
            arrow_scale: 1.6,
            rest_particles: true,
            threads: 0,
            gpu: true,
            seed: 0x5EED,
            nu: None,
            quiet: false,
            preview: true,
        }
    }
}

const USAGE: &str = "\
lgca -- hexagonal lattice-gas fluid (NKS pp. 378-380)

The defaults run flow past a plate at a Reynolds number near 100: 2.6 million
cells in about 8 MB, a few minutes of wall time.

  --width N            lattice columns              (default 2048)
  --height N           lattice rows, must be even   (default 1280)
  --density F          occupancy per direction      (default 0.22)
  --speed F            inflow speed, 0..1           (default 0.4)
  --steps N            update steps                 (default 40000)
  --warmup N           steps before recording       (default 6000)
  --obstacle KIND      plate | cylinder | none      (default plate)
  --size F             plate length or diameter     (default 173)
  --obstacle-x F       obstacle position in x       (default width/5)
  --block N            coarse-graining block edge   (default 20)
  --out DIR            output directory             (default out)
  --frame-every N      steps between frames         (default 500)
  --sample-every N     steps between field samples  (default 5)
  --alpha F            time-average weight, 0..1    (default 0.2)
  --scale N            pixels per block             (default 8)
  --arrow-scale F      arrow length multiplier      (default 1.6)
  --rest               use rest particles, FHP-III   (default on)\n  --no-rest            six directions only, as in the book
  --threads N          worker threads               (default: all cores)
  --gpu                run the update on the GPU    (default: on if there is one)
  --no-gpu             run it on the CPU instead
  --seed N             random seed
  --nu F               assume this viscosity when reporting Reynolds number
  --measure            measure viscosity and advection factor, then exit
  --scan-density       measure transport coefficients across densities, then exit
  --no-preview         do not print the terminal view
  --quiet              only print the summary
  -h, --help
";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Run,
    Measure,
    ScanDensity,
}

fn parse_args() -> Result<(Config, Mode), String> {
    let mut c = Config::default();
    let mut mode = Mode::Run;
    let mut args = std::env::args().skip(1);
    let mut obstacle_x: Option<f32> = None;

    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| format!("{a} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--width" => c.width = val()?.parse().map_err(|e| format!("{e}"))?,
            "--height" => c.height = val()?.parse().map_err(|e| format!("{e}"))?,
            "--density" => c.density = val()?.parse().map_err(|e| format!("{e}"))?,
            "--speed" => c.speed = val()?.parse().map_err(|e| format!("{e}"))?,
            "--steps" => c.steps = val()?.parse().map_err(|e| format!("{e}"))?,
            "--warmup" => c.warmup = val()?.parse().map_err(|e| format!("{e}"))?,
            "--size" => c.size = val()?.parse().map_err(|e| format!("{e}"))?,
            "--obstacle-x" => obstacle_x = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--block" => c.block = val()?.parse().map_err(|e| format!("{e}"))?,
            "--out" => c.out = PathBuf::from(val()?),
            "--frame-every" => c.frame_every = val()?.parse().map_err(|e| format!("{e}"))?,
            "--sample-every" => c.sample_every = val()?.parse().map_err(|e| format!("{e}"))?,
            "--alpha" => c.alpha = val()?.parse().map_err(|e| format!("{e}"))?,
            "--scale" => c.scale = val()?.parse().map_err(|e| format!("{e}"))?,
            "--arrow-scale" => c.arrow_scale = val()?.parse().map_err(|e| format!("{e}"))?,
            "--threads" => c.threads = val()?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => c.seed = val()?.parse().map_err(|e| format!("{e}"))?,
            "--nu" => c.nu = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--gpu" => c.gpu = true,
            "--no-gpu" => c.gpu = false,
            "--rest" => c.rest_particles = true,
            "--no-rest" => c.rest_particles = false,
            "--measure" | "--measure-viscosity" => mode = Mode::Measure,
            "--scan-density" => mode = Mode::ScanDensity,
            "--no-preview" => c.preview = false,
            "--quiet" => {
                c.quiet = true;
                c.preview = false;
            }
            "--obstacle" => {
                c.obstacle = match val()?.as_str() {
                    "plate" => Obstacle::Plate,
                    "cylinder" => Obstacle::Cylinder,
                    "none" => Obstacle::None,
                    other => return Err(format!("unknown obstacle {other}")),
                }
            }
            other => return Err(format!("unknown option {other}")),
        }
    }

    if c.height % 2 != 0 {
        return Err("--height must be even".into());
    }
    if !(0.0..=0.5).contains(&c.density) {
        return Err("--density must be in 0..0.5".into());
    }
    c.obstacle_x = obstacle_x.unwrap_or(c.width as f32 / 5.0);
    Ok((c, mode))
}

/// One frame's worth of output, on its way to a writer thread.
struct Frame {
    field: Field,
    png: PathBuf,
    svg: PathBuf,
    scale: usize,
    arrow_scale: f32,
    mean: (f32, f32),
}

/// Frames are written on threads of their own.
///
/// Encoding one costs about 10 ms --- 6.7 of PNG deflate, 3.1 of SVG
/// formatting --- against 19 ms of stepping between frames, so doing it inline
/// left the GPU idle for a third of the run. The frames are independent and a
/// copy of the field is 100 kB, so they can simply be handed off. The queue is
/// bounded, so a run that outpaces its writers waits for them rather than
/// growing without limit.
struct Frames {
    tx: Option<mpsc::SyncSender<Frame>>,
    workers: Vec<thread::JoinHandle<()>>,
    failure: Arc<Mutex<Option<String>>>,
}

fn write_frame(f: &Frame) -> std::io::Result<()> {
    let vort = f.field.vorticity();
    let wmax = {
        let mut v: Vec<f32> = vort
            .iter()
            .enumerate()
            .filter(|(k, _)| f.field.solid[*k] < 0.12)
            .map(|(_, x)| x.abs())
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() * 98 / 100).copied().unwrap_or(1e-3).max(1e-5)
    };
    render::write_vorticity(&f.png, &f.field, f.scale, wmax)?;
    render::write_arrows(&f.svg, &f.field, f.arrow_scale, f.mean)
}

impl Frames {
    fn new(workers: usize) -> Frames {
        let (tx, rx) = mpsc::sync_channel::<Frame>(workers);
        let rx = Arc::new(Mutex::new(rx));
        let failure = Arc::new(Mutex::new(None));
        let workers = (0..workers)
            .map(|_| {
                let (rx, failure) = (Arc::clone(&rx), Arc::clone(&failure));
                thread::spawn(move || loop {
                    // The lock is held only across `recv`, so the workers take
                    // turns picking a frame up and then run in parallel.
                    let job = rx.lock().unwrap().recv();
                    let Ok(job) = job else { return };
                    if let Err(e) = write_frame(&job) {
                        let mut slot = failure.lock().unwrap();
                        slot.get_or_insert_with(|| format!("{}: {e}", job.png.display()));
                    }
                })
            })
            .collect();
        Frames { tx: Some(tx), workers, failure }
    }

    /// Queue a frame, and report the first write that failed. Errors surface a
    /// frame or two late, which is soon enough to stop a run whose output
    /// directory has filled up.
    fn write(&self, frame: Frame) -> Result<(), String> {
        let _ = self.tx.as_ref().expect("writers still running").send(frame);
        match &*self.failure.lock().unwrap() {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Wait for everything queued to reach the disk.
    fn finish(mut self) -> Result<(), String> {
        drop(self.tx.take());
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        match &*self.failure.lock().unwrap() {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
}

fn threads(c: &Config) -> usize {
    if c.threads > 0 {
        c.threads
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    }
}

fn main() {
    let (c, mode) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    match mode {
        Mode::Measure => {
            let t = Instant::now();
            let tr = transport::measure(c.density, c.rest_particles, threads(&c), c.seed, 256, c.gpu);
            println!(
                "density {:.4} per direction ({:.2} particles per cell), rest particles: {}\n\
                 kinematic viscosity  nu = {:.4}\n\
                 advection factor      g = {:.4}   (a real fluid has g = 1)\n\
                 Reynolds coefficient g/nu = {:.3}, so Re = {:.3} * speed * size\n\
                 at speed {:.2}, Re = 100 needs an obstacle about {:.0} cells across\n\
                 measured in {:.1}s",
                c.density,
                c.density * if c.rest_particles { 7.0 } else { 6.0 },
                c.rest_particles,
                tr.nu,
                tr.g,
                tr.g / tr.nu,
                tr.g / tr.nu,
                c.speed,
                100.0 / tr.reynolds(c.speed, 1.0),
                t.elapsed().as_secs_f32()
            );
        }
        Mode::ScanDensity => {
            println!("  d      particles/cell     nu       g      g/nu");
            let mut d = 0.04;
            while d <= 0.46 {
                let tr = transport::measure(d, c.rest_particles, threads(&c), c.seed, 256, c.gpu);
                println!(
                    "{:6.3}   {:8.2}        {:7.4}  {:6.3}  {:7.3}",
                    d,
                    d * if c.rest_particles { 7.0 } else { 6.0 },
                    tr.nu,
                    tr.g,
                    tr.g / tr.nu
                );
                d += 0.03;
            }
        }
        Mode::Run => run(c),
    }
}

fn run(c: Config) {
    let nth = threads(&c);
    let mut lat = Lattice::new(
        c.width,
        c.height,
        c.density,
        (c.speed, 0.0),
        8,
        c.rest_particles,
        nth,
        c.seed,
    );

    let yc = c.height as f32 * SQRT3_2 / 2.0;
    match c.obstacle {
        Obstacle::Plate => {
            let (x0, x1) = (c.obstacle_x, c.obstacle_x + 4.0);
            let (y0, y1) = (yc - c.size / 2.0, yc + c.size / 2.0);
            lat.add_solid(move |x, y| x >= x0 && x <= x1 && y >= y0 && y <= y1);
        }
        Obstacle::Cylinder => {
            let r = c.size / 2.0;
            let cx = c.obstacle_x;
            lat.add_solid(move |x, y| (x - cx).powi(2) + (y - yc).powi(2) <= r * r);
        }
        Obstacle::None => {}
    }
    lat.init_equilibrium();

    std::fs::create_dir_all(&c.out).expect("cannot create output directory");

    if !c.quiet {
        println!("measuring transport coefficients...");
    }
    let mut tr = transport::measure(c.density, c.rest_particles, nth, c.seed, 256, c.gpu);
    if let Some(nu) = c.nu {
        tr.nu = nu;
    }
    let re = tr.reynolds(c.speed, c.size);

    let mut field = Field::new(&lat, c.block, c.block);
    let (table_states, table_active, table_stride) =
        (lat.table.n_states, lat.table.active_states(), lat.table.stride());
    let (mut sim, running_on) = Sim::new(lat, c.gpu);
    sim.attach_field(&field);

    if !c.quiet {
        println!(
            "lattice {} x {} = {:.1}M cells, running on the {running_on}\n\
             density {:.4}/direction ({:.2} particles per cell), inflow speed {:.2}\n\
             collision table: {} states, {} of them with alternatives, stride {}\n\
             nu = {:.4}, g = {:.3}  ->  Reynolds number ~ {:.0} on L = {:.0}\n",
            c.width,
            c.height,
            (c.width * c.height) as f32 / 1e6,
            c.density,
            c.density * if c.rest_particles { 7.0 } else { 6.0 },
            c.speed,
            table_states,
            table_active,
            table_stride,
            tr.nu,
            tr.g,
            re,
            c.size
        );
    }

    let start = Instant::now();
    let mass0 = sim.total_particles();

    if c.warmup > 0 {
        sim.advance(c.warmup);
    }
    sim.sample(&mut field, 1.0);

    // Four writers cover a `--frame-every` down to about 100 steps; past that
    // the run waits on them, which is the right way round.
    let frames = Frames::new(nth.clamp(1, 4));
    let mut frame = 0usize;
    let mut step = 0u64;
    while step < c.steps {
        // Run to the next frame in one go. On the GPU that is one command
        // buffer and one synchronisation; sampling every fifth step from the
        // CPU side would cost more than the steps themselves.
        let to_frame = c.frame_every - step % c.frame_every;
        let chunk = to_frame.min(c.steps - step);
        sim.advance_sampling(chunk, c.sample_every, c.alpha, &mut field);
        step += chunk;

        let mean = field.mean_velocity();
        let png = c.out.join(format!("vorticity-{frame:04}.png"));
        let svg = c.out.join(format!("arrows-{frame:04}.svg"));
        if let Err(e) = frames.write(Frame {
            field: field.clone(),
            png: png.clone(),
            svg,
            scale: c.scale,
            arrow_scale: c.arrow_scale,
            mean,
        }) {
            eprintln!("error: writing a frame failed: {e}");
            std::process::exit(1);
        }

        if !c.quiet {
            let rate = (c.width * c.height) as f64 * step as f64
                / start.elapsed().as_secs_f64()
                / 1e6;
            let cells_per = if c.rest_particles { 7.0 } else { 6.0 };
            println!(
                "step {step:>7}/{}  mean u = ({:+.3}, {:+.3})  density = {:.3}/dir  \
                 {rate:.0}M cell-updates/s  -> {}",
                c.steps,
                mean.0,
                mean.1,
                sim.total_particles() as f32 / (c.width * c.height) as f32 / cells_per,
                png.display()
            );
            if c.preview {
                print!("{}", render::ascii_preview(&field, 100, 22));
            }
        }
        frame += 1;
    }

    let mass1 = sim.total_particles();
    if let Err(e) = frames.finish() {
        eprintln!("error: writing a frame failed: {e}");
        std::process::exit(1);
    }
    if !c.quiet {
        println!(
            "\ndone in {:.1}s. {frame} frames in {}. particle count {} -> {} ({}).",
            start.elapsed().as_secs_f32(),
            c.out.display(),
            mass0,
            mass1,
            if mass0 == mass1 {
                "exactly conserved"
            } else {
                "changed at the inflow boundary, as expected"
            }
        );
    }
}
