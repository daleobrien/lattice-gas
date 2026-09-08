//! Measuring the lattice gas's own transport coefficients.
//!
//! A lattice gas is not the Navier-Stokes equation; it only behaves like one
//! after averaging, with a viscosity and an advection coefficient that fall
//! out of the microscopic rules. Both are measured here rather than quoted, so
//! the Reynolds number the program reports is a property of the code that is
//! actually running --- which is why the measurement runs on whichever backend
//! the run itself will use.
//!
//! Both measurements watch a single Fourier mode. The mode is projected onto
//! `cos` and `sin`, giving a vector whose length is the amplitude and whose
//! angle is the phase. Amplitude decays at `nu k^2`; in a moving stream the
//! phase drifts at `g U k`. Because the lattice is noisy, several independent
//! realisations are averaged *before* the fit -- the signal is identical
//! across them and adds, while the fluctuations are zero-mean and cancel.

use crate::hex::SQRT3_2;
use crate::lattice::{Equilibrium, Lattice};
use crate::moments;
use crate::rng::Rng;
use crate::sim::Sim;
use std::f64::consts::PI;

/// Steps discarded before fitting, to let non-hydrodynamic modes die away.
const TRANSIENT: u64 = 200;
const MIN_INTERVAL: u64 = 4;
const SAMPLES: usize = 20;
const REPS: u64 = 4;
const MAX_WINDOW: u64 = 30_000;

pub struct Transport {
    /// Kinematic viscosity, in lattice units.
    pub nu: f32,
    /// Coefficient of the advection term. A real fluid has g = 1; a lattice
    /// gas does not, because its equilibrium is not Galilean invariant.
    pub g: f32,
}

impl Transport {
    /// Reynolds number for a flow of speed `u` past an obstacle of size `l`.
    pub fn reynolds(&self, u: f32, l: f32) -> f32 {
        self.g * u * l / self.nu
    }
}

/// Column-wise mean velocity along y, weighted by the particles present.
///
/// Walked a row at a time rather than a column at a time: one packed
/// accumulator per column turns what was a stride-`w` crawl through the whole
/// lattice per column into a single sequential pass over it.
pub(crate) fn column_uy(cells: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut mass = vec![0i64; w];
    let mut py2 = vec![0i64; w];
    let mut acc = vec![0u64; w];

    let mut y0 = 0;
    while y0 < h {
        // Flush before a 16-bit field can overflow.
        let y1 = (y0 + moments::MAX_BLOCK).min(h);
        acc.iter_mut().for_each(|a| *a = 0);
        for y in y0..y1 {
            let row = &cells[y * w..(y + 1) * w];
            for (a, &c) in acc.iter_mut().zip(row) {
                *a += moments::WITH_WALLS[c as usize];
            }
        }
        for x in 0..w {
            let m = moments::unpack(acc[x], y1 - y0);
            mass[x] += m.mass;
            py2[x] += m.py2;
        }
        y0 = y1;
    }

    (0..w)
        .map(|x| {
            if mass[x] > 0 {
                py2[x] as f32 * SQRT3_2 / mass[x] as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Row-wise mean velocity along x.
pub(crate) fn row_ux(cells: &[u8], w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .map(|y| moments::sum(&moments::WITH_WALLS, &cells[y * w..(y + 1) * w]).velocity().0)
        .collect()
}

fn fit_slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// One realisation: the initial cells and the seed to run them from.
struct Start {
    cells: Vec<u8>,
    seed: u64,
}

/// Sample the mode every `interval` steps and return the complex trajectory.
fn trajectory<P>(
    sim: &mut Sim,
    start: &Start,
    project: &P,
    interval: u64,
    samples: usize,
) -> Vec<(f64, f64)>
where
    P: Fn(&[u8]) -> (f64, f64),
{
    sim.restart(&start.cells, start.seed);
    sim.advance(TRANSIENT);
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        out.push(project(sim.cells()));
        sim.advance(interval);
    }
    out
}

/// Time for the mode to fall to a third of its initial amplitude, so the fit
/// always lands where signal still dominates the lattice's own noise.
fn decay_window<P>(sim: &mut Sim, start: &Start, project: &P) -> u64
where
    P: Fn(&[u8]) -> (f64, f64),
{
    sim.restart(&start.cells, start.seed);
    sim.advance(TRANSIENT);
    let (a, b) = project(sim.cells());
    let a0 = (a * a + b * b).sqrt().max(1e-12);
    let block = 25u64;
    let mut t = 0u64;
    while t < MAX_WINDOW {
        sim.advance(block);
        t += block;
        let (a, b) = project(sim.cells());
        if (a * a + b * b).sqrt() < a0 / 3.0 {
            break;
        }
    }
    t.clamp(SAMPLES as u64 * MIN_INTERVAL, MAX_WINDOW)
}

/// Average the trajectories of independent realisations point by point.
fn averaged<B, P>(sim: &mut Sim, start: B, project: P) -> (Vec<f64>, Vec<(f64, f64)>)
where
    B: Fn(u64) -> Start,
    P: Fn(&[u8]) -> (f64, f64),
{
    let window = decay_window(sim, &start(0), &project);
    let interval = (window / SAMPLES as u64).max(MIN_INTERVAL);

    let mut acc = vec![(0.0f64, 0.0f64); SAMPLES];
    for rep in 0..REPS {
        let tr = trajectory(sim, &start(rep), &project, interval, SAMPLES);
        for (a, t) in acc.iter_mut().zip(tr) {
            a.0 += t.0 / REPS as f64;
            a.1 += t.1 / REPS as f64;
        }
    }
    let times = (0..SAMPLES).map(|i| (i as u64 * interval) as f64).collect();
    (times, acc)
}

/// A quiescent periodic box of the right size, on the right processor. The
/// realisations reuse it rather than building one each: on the GPU, compiling
/// the shader costs more than a short run does.
fn box_of(size: usize, density: f32, rest: bool, threads: usize, gpu: bool) -> Sim {
    let lat = Lattice::new(size, size, density, (0.0, 0.0), 0, rest, threads, 1);
    Sim::new(lat, gpu).0
}

/// Seed a transverse shear wave in a quiescent periodic box and watch its
/// amplitude decay as exp(-nu k^2 t).
pub fn viscosity(density: f32, rest: bool, threads: usize, seed: u64, size: usize, gpu: bool) -> f32 {
    let (w, h) = (size, size);
    let k = 2.0 * std::f32::consts::PI / (h as f32 * SQRT3_2);
    let amp = 0.05;

    let start = move |rep: u64| {
        let s = seed ^ (rep.wrapping_mul(0x9E37_79B9) << 20);
        let mut rng = Rng::new(s ^ 0xBEEF);
        let mut cells = vec![0u8; w * h];
        for y in 0..h {
            let ux = amp * (k * y as f32 * SQRT3_2).sin();
            let eq = Equilibrium::new(density, ux, 0.0, rest);
            for x in 0..w {
                cells[y * w + x] = eq.sample(&mut rng);
            }
        }
        Start { cells, seed: s }
    };

    let project = move |cells: &[u8]| {
        let u = row_ux(cells, w, h);
        let (mut a, mut b) = (0.0f64, 0.0f64);
        for y in 0..h {
            let p = k * y as f32 * SQRT3_2;
            a += (u[y] * p.cos()) as f64;
            b += (u[y] * p.sin()) as f64;
        }
        (2.0 * a / h as f64, 2.0 * b / h as f64)
    };

    let mut sim = box_of(size, density, rest, threads, gpu);
    let (times, mode) = averaged(&mut sim, start, project);
    let pts: Vec<(f64, f64)> = times
        .iter()
        .zip(&mode)
        .map(|(t, (a, b))| (*t, ((a * a + b * b).sqrt().max(1e-12)).ln()))
        .collect();
    (-fit_slope(&pts) / (k * k) as f64) as f32
}

/// Superimpose a transverse wave on a uniform stream. The pattern is advected
/// at `g * U`, not at `U`, and the phase drift measures `g` directly.
pub fn advection_factor(
    density: f32,
    rest: bool,
    threads: usize,
    seed: u64,
    size: usize,
    gpu: bool,
) -> f32 {
    let (w, h) = (size, size);
    let u0 = 0.25f32;
    let k = 2.0 * std::f32::consts::PI / w as f32;
    let amp = 0.05;

    let start = move |rep: u64| {
        let s = seed ^ (rep.wrapping_mul(0x9E37_79B9) << 20) ^ 0x1357;
        let mut rng = Rng::new(s ^ 0xF00D);
        let cols: Vec<Equilibrium> = (0..w)
            .map(|x| Equilibrium::new(density, u0, amp * (k * x as f32).sin(), rest))
            .collect();
        let mut cells = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                cells[y * w + x] = cols[x].sample(&mut rng);
            }
        }
        Start { cells, seed: s }
    };

    let project = move |cells: &[u8]| {
        let u = column_uy(cells, w, h);
        let (mut a, mut b) = (0.0f64, 0.0f64);
        for x in 0..w {
            let p = k * x as f32;
            a += (u[x] * p.cos()) as f64;
            b += (u[x] * p.sin()) as f64;
        }
        (2.0 * a / w as f64, 2.0 * b / w as f64)
    };

    // The realised mean speed, not the requested one, is what advects the mode.
    let u_actual = moments::sum(&moments::WITH_WALLS, &start(0).cells).velocity().0;

    let mut sim = box_of(size, density, rest, threads, gpu);
    let (times, mode) = averaged(&mut sim, start, project);

    // Unwrap the phase, which advances by -k * g * u0 per unit time.
    let mut unwrapped = 0.0f64;
    let mut prev = mode[0].0.atan2(mode[0].1);
    let mut pts = Vec::with_capacity(mode.len());
    for (t, (a, b)) in times.iter().zip(&mode) {
        let ph = a.atan2(*b);
        let mut delta = ph - prev;
        while delta > PI {
            delta -= 2.0 * PI;
        }
        while delta < -PI {
            delta += 2.0 * PI;
        }
        unwrapped += delta;
        prev = ph;
        pts.push((*t, unwrapped));
    }

    (-fit_slope(&pts) / (k * u_actual) as f64) as f32
}

pub fn measure(
    density: f32,
    rest: bool,
    threads: usize,
    seed: u64,
    size: usize,
    gpu: bool,
) -> Transport {
    Transport {
        nu: viscosity(density, rest, threads, seed, size, gpu),
        g: advection_factor(density, rest, threads, seed, size, gpu),
    }
}
