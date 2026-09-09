//! Command-line driver for the hexagonal lattice gas. The simulation itself
//! lives in the `lattice_gas` library alongside this binary.

use lattice_gas::hex::SQRT3_2;
use lattice_gas::lattice::Lattice;
use lattice_gas::render::{self, Field};
use lattice_gas::sim::Sim;
use lattice_gas::transport;
use std::io::{IsTerminal, Write};
use std::os::fd::AsFd;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
    alpha: Option<f32>,
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
            alpha: None,
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
  --alpha F            time-average weight, 0..1    (default: sample-every/25)
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
            "--alpha" => c.alpha = Some(val()?.parse().map_err(|e| format!("{e}"))?),
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

/// The per-frame status display: a line of numbers and, unless it is turned
/// off, the ascii view of the flow.
///
/// On a terminal the block is drawn over the previous one, so a long run holds
/// still on one screenful instead of scrolling a few hundred views past. Piped
/// to a file or a log, where cursor movement would only leave escape codes
/// behind, the blocks are printed one after another as they always were.
struct Progress {
    in_place: bool,
    /// Terminal size as (rows, cols), when we could find it out.
    size: Option<(usize, usize)>,
    asked: Instant,
    /// Rows the last block covered, which is how far back up to go.
    drawn: usize,
    /// How long the closing summary will be, which decides how many rows to
    /// hold open for it.
    summary: usize,
}

impl Progress {
    fn new(summary: usize) -> Progress {
        let in_place = std::io::stdout().is_terminal();
        Progress {
            in_place,
            size: if in_place { terminal_size() } else { None },
            asked: Instant::now(),
            drawn: 0,
            summary,
        }
    }

    /// Blank rows to keep below the view. The summary is written over the bar
    /// at the end, so the bar's own row covers all of a summary that fits on
    /// one line, and only the rows it wraps onto need holding open.
    fn reserved(&self) -> usize {
        match self.size {
            Some((_, cols)) => self.summary.div_ceil(cols.max(1)).saturating_sub(1),
            None => 1,
        }
    }

    /// The size to draw the ascii view at: the 100 x 22 it has always been,
    /// less whatever it takes to leave the window a step line and the reserved
    /// rows. Redrawing in place only works while the whole block fits on
    /// screen, so a small window gets a small view rather than a scrolling one.
    ///
    /// This is where a resized window is noticed, one frame ahead of the draw
    /// that follows it, so the view and the rows held under it agree.
    fn view(&mut self) -> (usize, usize) {
        // Ask again every so often rather than every frame, which for a small
        // `--frame-every` would have us forking `stty` in a loop.
        if self.in_place && self.asked.elapsed() > Duration::from_secs(2) {
            self.size = terminal_size();
            self.asked = Instant::now();
        }
        match self.size {
            Some((rows, cols)) => (
                cols.saturating_sub(1).min(100),
                rows.saturating_sub(self.reserved() + 2).min(22),
            ),
            None => (100, 22),
        }
    }

    fn draw(&mut self, mut block: String) {
        if !block.ends_with('\n') {
            block.push('\n');
        }
        let mut out = std::io::stdout().lock();
        if !self.in_place {
            let _ = out.write_all(block.as_bytes());
            let _ = out.flush();
            return;
        }

        if let Some((_, cols)) = self.size {
            block = clip(&block, cols);
        }
        if self.drawn > 0 {
            // Back to the top of the last block, then clear to the bottom of
            // the screen so a shorter block leaves no tail behind.
            let _ = write!(out, "\x1b[{}A\x1b[J", self.drawn);
        }
        self.drawn = block.bytes().filter(|b| *b == b'\n').count();
        let _ = out.write_all(block.as_bytes());
        // Take the rows a wrapping summary will spill onto now, while the view
        // can still be redrawn, and step back over them. Closing the display
        // then costs no scrolling, so the last view stays where the run left it
        // instead of sliding up out of place.
        let reserved = self.reserved();
        if reserved > 0 {
            let _ = write!(out, "{}\x1b[{reserved}A", "\n".repeat(reserved));
        }
        let _ = out.flush();
    }

    /// Close the display by writing the summary where the bar was, leaving the
    /// run's last view above it rather than a bar stopped at 100%.
    fn finish(&mut self, summary: &str) {
        let mut out = std::io::stdout().lock();
        if self.in_place && self.drawn > 0 {
            // Up onto the bar and clear from there down; the view stays put.
            let _ = write!(out, "\x1b[1A\x1b[J");
        } else {
            // Nothing to write over, so keep the blank line that used to
            // separate the summary from the frames above it.
            let _ = writeln!(out);
        }
        let _ = writeln!(out, "{summary}");
        let _ = out.flush();
    }
}

/// The step counter as a bar filled to the fraction of the run that is done,
/// drawn to the width the view above it uses.
fn bar(step: u64, steps: u64, cols: usize) -> String {
    // Both numbers are padded to the width they end at, so nothing beside the
    // bar shuffles sideways as the run goes on.
    let digits = steps.to_string().len();
    let count = format!("{:>3}%  step {step:>digits$}/{steps}", 100 * step / steps.max(1));
    // Whatever the brackets and the count beside them leave.
    let width = cols.saturating_sub(count.len() + 4);
    let filled = (width as u64 * step / steps.max(1)) as usize;
    format!("[{}{}]  {count}\n", "#".repeat(filled), ".".repeat(width - filled))
}

/// Cut every line to the width of the window. A line that runs past the right
/// margin wraps onto a second row, and the redraw counts rows, not lines.
fn clip(block: &str, cols: usize) -> String {
    let mut out = String::with_capacity(block.len());
    for line in block.lines() {
        out.extend(line.chars().take(cols));
        out.push('\n');
    }
    out
}

/// Ask the terminal how big it is, as (rows, cols). The standard library has
/// no way to, so this asks `stty`, which reads its input: hand it our own
/// output, which is the terminal we are drawing on. /dev/tty would be the
/// obvious thing to give it and is the wrong one, being whatever terminal
/// started us rather than wherever stdout now goes.
fn terminal_size() -> Option<(usize, usize)> {
    let screen = std::io::stdout().as_fd().try_clone_to_owned().ok()?;
    let out = std::process::Command::new("stty")
        .arg("size")
        .stdin(std::process::Stdio::from(screen))
        .output()
        .ok()?;
    let text = std::str::from_utf8(&out.stdout).ok()?;
    let mut n = text.split_whitespace();
    let rows: usize = n.next()?.parse().ok()?;
    let cols: usize = n.next()?.parse().ok()?;
    (rows > 0 && cols > 0).then_some((rows, cols))
}

/// How many steps the running time average looks back over.
///
/// `--alpha` is a weight *per sample*, so raising `--sample-every` without
/// touching it lengthens the average in steps and smears the vortices as they
/// advect. Deriving one from the other holds the window fixed at the 25 steps
/// the defaults have always used, which is what makes a longer sampling
/// interval a real saving rather than a trade against the picture.
///
/// What it does trade against is noise: the average holds roughly
/// `(1 / alpha) * bx * by` cells, so a longer interval needs a bigger block to
/// stay as quiet. The defaults are 5 steps and a 20-cell block, or 2,000
/// cells; at a 32-cell block, `--sample-every 12` matches that.
const AVERAGE_STEPS: f32 = 25.0;

fn alpha(c: &Config) -> f32 {
    c.alpha
        .unwrap_or_else(|| (c.sample_every as f32 / AVERAGE_STEPS).clamp(0.02, 1.0))
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

    let alpha = alpha(&c);
    let start = Instant::now();
    let mass0 = sim.total_particles();

    if c.warmup > 0 {
        sim.advance(c.warmup);
    }
    sim.sample(&mut field, 1.0);

    // The line the run closes with. Writing it here as well as at the end
    // gives the display something to measure, so it can hold that many rows
    // open under the view and the summary can land without pushing it up.
    let summary = |secs: f64, rate: f64, frames: u64, mass1: u64| {
        format!(
            "done in {secs:.1}s at {rate:.0}M cell-updates/s. {frames} frames in {}. \
             particle count {mass0} -> {mass1}{}.",
            c.out.display(),
            if mass0 == mass1 { ", exactly conserved" } else { "" }
        )
    };

    // Four writers cover a `--frame-every` down to about 100 steps; past that
    // the run waits on them, which is the right way round.
    let frames = Frames::new(nth.clamp(1, 4));
    // Times and rates no run will exceed, and the wordier of the two endings,
    // so the measurement is an upper bound on the real one and the view never
    // ends up a row short.
    let mut progress = Progress::new(
        summary(9999.9, 99999.0, c.steps.div_ceil(c.frame_every), mass0).len(),
    );
    let mut frame = 0usize;
    let mut step = 0u64;
    while step < c.steps {
        // Run to the next frame in one go. On the GPU that is one command
        // buffer and one synchronisation; sampling every fifth step from the
        // CPU side would cost more than the steps themselves.
        let to_frame = c.frame_every - step % c.frame_every;
        let chunk = to_frame.min(c.steps - step);
        sim.advance_sampling(chunk, c.sample_every, alpha, &mut field);
        step += chunk;

        let mean = field.mean_velocity();
        let png = c.out.join(format!("vorticity-{frame:04}.png"));
        let svg = c.out.join(format!("arrows-{frame:04}.svg"));
        if let Err(e) = frames.write(Frame {
            field: field.clone(),
            png,
            svg,
            scale: c.scale,
            arrow_scale: c.arrow_scale,
            mean,
        }) {
            eprintln!("error: writing a frame failed: {e}");
            std::process::exit(1);
        }

        if !c.quiet {
            let (cols, rows) = progress.view();
            let mut block = String::new();
            if c.preview {
                block.push_str(&render::ascii_preview(&field, cols, rows));
            }
            block.push_str(&bar(step, c.steps, cols));
            progress.draw(block);
        }
        frame += 1;
    }

    let mass1 = sim.total_particles();
    if let Err(e) = frames.finish() {
        eprintln!("error: writing a frame failed: {e}");
        std::process::exit(1);
    }
    if !c.quiet {
        let secs = start.elapsed().as_secs_f64();
        let rate = (c.width * c.height) as f64 * c.steps as f64 / secs / 1e6;
        progress.finish(&summary(secs, rate, frame as u64, mass1));
    }
}
