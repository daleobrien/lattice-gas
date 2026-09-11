//! Coarse-graining and output.
//!
//! A single cell is either occupied or not, so the raw lattice looks like
//! noise. The fluid appears only after averaging: the book uses blocks of
//! 20x20 cells, and we do the same, optionally with a running time average on
//! top to quieten the residual fluctuation.

use crate::hex::SQRT3_2;
use crate::lattice::Lattice;
use crate::moments;
use std::fmt::Write as _;
use std::path::Path;

/// A block counts as obstacle for display purposes above this solid fraction.
/// A thin plate only ever fills a fraction of a coarse block, so the threshold
/// has to sit below that.
const SOLID: f32 = 0.12;

/// Cloneable so a frame can be handed to a writer thread and the simulation
/// can carry on into the next one. At the default block size this is 100 kB.
#[derive(Clone)]
pub struct Field {
    pub bw: usize,
    pub bh: usize,
    pub bx: usize,
    pub by: usize,
    pub ux: Vec<f32>,
    pub uy: Vec<f32>,
    pub rho: Vec<f32>,
    /// Fraction of each block occupied by the obstacle.
    pub solid: Vec<f32>,
}

impl Field {
    pub fn new(lat: &Lattice, bx: usize, by: usize) -> Self {
        let bw = lat.w / bx;
        let bh = lat.h / by;
        Field {
            bw,
            bh,
            bx,
            by,
            ux: vec![0.0; bw * bh],
            uy: vec![0.0; bw * bh],
            rho: vec![0.0; bw * bh],
            solid: vec![0.0; bw * bh],
        }
    }

    /// Accumulate the current lattice state. `alpha` of 1.0 is a pure spatial
    /// average of this instant; smaller values blend with the running history.
    ///
    /// A block is summed through `moments::FLUID`, one table load and one
    /// integer add per cell, and the block rows are shared out over the same
    /// worker threads the update uses. Each block's sum is exact and depends
    /// only on its own cells, so the thread count cannot change the answer.
    ///
    /// Obstacle sites are left out of the mass and momentum --- a block that is
    /// half wall reports the velocity of the half that is fluid --- but their
    /// share of the block is recorded in `solid`.
    pub fn sample(&mut self, lat: &Lattice, alpha: f32) {
        assert!(
            self.bx * self.by <= moments::MAX_BLOCK,
            "a {} x {} block holds {} cells, more than the {} a packed sum can \
             accumulate without its fields carrying into one another",
            self.bx,
            self.by,
            self.bx * self.by,
            moments::MAX_BLOCK
        );
        let (bw, bh, bx, by) = (self.bw, self.bh, self.bx, self.by);
        if bw == 0 || bh == 0 {
            return;
        }
        let (w, cells) = (lat.w, &lat.cells[..]);
        let n = (bx * by) as f32;
        let band = {
            let nthreads = lat.threads().clamp(1, bh);
            (bh + nthreads - 1) / nthreads
        };
        let stripe = band * bw;

        std::thread::scope(|scope| {
            for (b, (((ux, uy), rho), solid)) in self
                .ux
                .chunks_mut(stripe)
                .zip(self.uy.chunks_mut(stripe))
                .zip(self.rho.chunks_mut(stripe))
                .zip(self.solid.chunks_mut(stripe))
                .enumerate()
            {
                scope.spawn(move || {
                    let mut acc = vec![0u64; bw];
                    for r in 0..ux.len() / bw {
                        acc.iter_mut().for_each(|a| *a = 0);
                        let jb = b * band + r;
                        for y in jb * by..(jb + 1) * by {
                            let row = &cells[y * w..y * w + bw * bx];
                            for (ib, a) in acc.iter_mut().enumerate() {
                                moments::accumulate(
                                    &moments::FLUID,
                                    &row[ib * bx..(ib + 1) * bx],
                                    a,
                                );
                            }
                        }
                        for (ib, &a) in acc.iter().enumerate() {
                            let m = moments::unpack(a, bx * by);
                            let (vx, vy) = m.velocity();
                            let k = r * bw + ib;
                            ux[k] += alpha * (vx - ux[k]);
                            uy[k] += alpha * (vy - uy[k]);
                            rho[k] += alpha * (m.mass as f32 / n - rho[k]);
                            solid[k] = m.solid as f32 / n;
                        }
                    }
                });
            }
        });
    }

    /// Vorticity on the block grid, in units of inverse lattice time.
    pub fn vorticity(&self) -> Vec<f32> {
        let dx = self.bx as f32;
        let dy = self.by as f32 * SQRT3_2;
        let mut w = vec![0.0; self.bw * self.bh];
        for j in 0..self.bh {
            for i in 0..self.bw {
                let ip = (i + 1).min(self.bw - 1);
                let im = i.saturating_sub(1);
                let jp = if j + 1 < self.bh { j + 1 } else { 0 };
                let jm = if j == 0 { self.bh - 1 } else { j - 1 };
                let dvdx = (self.uy[j * self.bw + ip] - self.uy[j * self.bw + im])
                    / ((ip - im) as f32 * dx);
                let dudy = (self.ux[jp * self.bw + i] - self.ux[jm * self.bw + i]) / (2.0 * dy);
                w[j * self.bw + i] = dvdx - dudy;
            }
        }
        w
    }

    /// Mean velocity over the fluid part of the field.
    pub fn mean_velocity(&self) -> (f32, f32) {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
        for k in 0..self.ux.len() {
            if self.solid[k] < 0.5 {
                sx += self.ux[k];
                sy += self.uy[k];
                n += 1.0;
            }
        }
        if n == 0.0 {
            (0.0, 0.0)
        } else {
            (sx / n, sy / n)
        }
    }
}

fn diverging(t: f32) -> (u8, u8, u8) {
    // t in -1..1: blue for negative, red for positive, near-white at zero.
    let t = t.clamp(-1.0, 1.0);
    let (r, g, b) = if t >= 0.0 {
        (1.0, 1.0 - 0.75 * t, 1.0 - 0.9 * t)
    } else {
        (1.0 + 0.9 * t, 1.0 + 0.6 * t, 1.0)
    };
    (
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

/// Vorticity map: the view that makes the vortex street obvious.
pub fn write_vorticity(path: &Path, f: &Field, scale: usize, wmax: f32) -> std::io::Result<()> {
    let vort = f.vorticity();
    let px = scale.max(1);
    let py = ((px as f32 * SQRT3_2 * f.by as f32 / f.bx as f32).round() as usize).max(1);
    let (iw, ih) = (f.bw * px, f.bh * py);
    let mut img = vec![0u8; iw * ih * 3];

    for j in 0..f.bh {
        for i in 0..f.bw {
            let k = j * f.bw + i;
            let (r, g, b) = if f.solid[k] > SOLID {
                (30, 30, 35)
            } else {
                diverging(vort[k] / wmax)
            };
            for dy in 0..py {
                // Image rows run downward; the lattice y axis runs upward.
                let iy = ih - 1 - (j * py + dy);
                for dx in 0..px {
                    let o = (iy * iw + i * px + dx) * 3;
                    img[o] = r;
                    img[o + 1] = g;
                    img[o + 2] = b;
                }
            }
        }
    }
    crate::png::write_rgb(path, iw, ih, &img)
}

/// Arrow plot in the style of the book's figure. `frame` is subtracted from
/// every vector, so passing the mean flow shows the fluid at rest with the
/// obstacle moving through it.
pub fn write_arrows(path: &Path, f: &Field, scale: f32, frame: (f32, f32)) -> std::io::Result<()> {
    let w = f.bw as f32 * f.bx as f32;
    let h = f.bh as f32 * f.by as f32 * SQRT3_2;
    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{}\" height=\"{}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n\
         <g stroke=\"#1a1a1a\" stroke-width=\"1.1\" fill=\"none\" stroke-linecap=\"round\">\n",
        w.round() as i32,
        h.round() as i32
    ));

    let arrow = f.bx as f32 * scale;
    for j in 0..f.bh {
        for i in 0..f.bw {
            let k = j * f.bw + i;
            if f.solid[k] > SOLID {
                continue;
            }
            let cx = (i as f32 + 0.5) * f.bx as f32;
            // Flip to SVG's downward y axis.
            let cy = h - (j as f32 + 0.5) * f.by as f32 * SQRT3_2;
            let vx = (f.ux[k] - frame.0) * arrow;
            let vy = -(f.uy[k] - frame.1) * arrow;
            let len = (vx * vx + vy * vy).sqrt();
            if len < 0.35 {
                continue;
            }
            let (x0, y0) = (cx - vx * 0.5, cy - vy * 0.5);
            let (x1, y1) = (cx + vx * 0.5, cy + vy * 0.5);
            let (hx, hy) = (vx / len, vy / len);
            let head = (len * 0.35).min(f.bx as f32 * 0.4);
            s.push_str(&format!(
                "<path d=\"M{x0:.1} {y0:.1}L{x1:.1} {y1:.1}m{:.1} {:.1}L{x1:.1} {y1:.1}l{:.1} {:.1}\"/>\n",
                -head * (hx + hy * 0.5),
                -head * (hy - hx * 0.5),
                -head * (hx - hy * 0.5),
                -head * (hy + hx * 0.5),
            ));
        }
    }

    s.push_str("</g>\n<g fill=\"#1a1a1a\">\n");
    for j in 0..f.bh {
        for i in 0..f.bw {
            if f.solid[j * f.bw + i] > SOLID {
                let x = i as f32 * f.bx as f32;
                let y = h - (j + 1) as f32 * f.by as f32 * SQRT3_2;
                s.push_str(&format!(
                    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\"/>\n",
                    f.bx as f32,
                    f.by as f32 * SQRT3_2
                ));
            }
        }
    }
    s.push_str("</g>\n</svg>\n");
    std::fs::write(path, s)
}

/// The size to draw the terminal view at, in character cells, fitted into a
/// window `cols` x `rows` in size.
///
/// Two things bound it. The lattice has an aspect ratio --- it is `bw * bx`
/// wide and `bh * by * SQRT3_2` tall --- and stretching the flow to fill a
/// window would be a lie about the geometry. And the block grid is all the
/// detail there is, so a view wider than `bw` only doubles up columns and
/// comes out looking coarser than the one that fits.
///
/// In colour a character cell holds two square sub-rows; without it one row,
/// on a cell about twice as tall as it is wide. Either way a row covers two
/// units of height per unit of width, which is why the same arithmetic serves
/// both.
pub fn preview_size(f: &Field, cols: usize, rows: usize, colour: bool) -> (usize, usize) {
    if cols == 0 || rows == 0 || f.bw == 0 || f.bh == 0 {
        return (0, 0);
    }
    let tall = f.bh as f32 * f.by as f32 * SQRT3_2 / (f.bw as f32 * f.bx as f32);
    let deep = if colour { f.bh.div_ceil(2) } else { f.bh };
    let mut c = cols.min(f.bw);
    let mut r = ((c as f32 * tall / 2.0).round() as usize).clamp(1, deep);
    if r > rows {
        // Too tall for the window, so the height is what is really available
        // and the width comes back to match it.
        r = rows;
        c = (((r * 2) as f32 / tall).round() as usize).clamp(1, cols.min(f.bw));
    }
    (c, r)
}

/// Terminal view of the vorticity, for watching a run without leaving the
/// shell.
///
/// Two sub-rows are packed into every character cell: an upper half block,
/// `\u{2580}`, painted in the colour of the row above sits on a background in
/// the colour of the row below. That buys the view twice the vertical
/// resolution for the same number of terminal rows, and puts the pixels at
/// roughly square aspect instead of the two-to-one a character cell has.
///
/// Colour carries the sign the way the PNGs do --- one rotation blue, the
/// other red --- which is what makes a vortex street read as a street rather
/// than a row of blobs. Without colour the same sampling is drawn with the
/// block-element shades, which lose the sign but ramp far more evenly than
/// punctuation ever did.
pub fn preview(f: &Field, cols: usize, rows: usize, colour: bool) -> String {
    if cols == 0 || rows == 0 || f.bw == 0 || f.bh == 0 {
        return String::new();
    }
    let vort = f.vorticity();
    // Scale to the strongest vorticity in the fluid; the obstacle's own edge
    // is excluded, being a discontinuity rather than a feature of the flow.
    let mut peak = 1e-9f32;
    for (k, v) in vort.iter().enumerate() {
        if f.solid[k] < SOLID {
            peak = peak.max(v.abs());
        }
    }
    if colour {
        colour_preview(f, &vort, peak, cols, rows)
    } else {
        shaded_preview(f, &vort, peak, cols, rows)
    }
}

/// The block holding sub-row `s` of `sub` and column `c` of `cols`, nearest
/// neighbour. Row 0 is the top of the view and the top of the lattice.
fn sample(f: &Field, c: usize, cols: usize, s: usize, sub: usize) -> usize {
    let i = (c * f.bw / cols).min(f.bw - 1);
    let j = f.bh - 1 - (s * f.bh / sub).min(f.bh - 1);
    j * f.bw + i
}

/// Vorticity mapped to -1..1, with a gamma that lifts the mid-tones. The peak
/// is a single cell somewhere; left linear, everything else sits in the bottom
/// of the ramp and the view comes out nearly blank.
fn shade(v: f32, peak: f32) -> f32 {
    let t = (v / peak).clamp(-1.0, 1.0);
    // A lattice gas is noisy by construction, and a tenth of the peak is about
    // what the residual fluctuation reaches. Cutting that away first keeps the
    // speckle dim and leaves the whole ramp for the flow.
    const FLOOR: f32 = 0.10;
    (((t.abs() - FLOOR).max(0.0) / (1.0 - FLOOR)).powf(0.7)).copysign(t)
}

fn colour_preview(f: &Field, vort: &[f32], peak: f32, cols: usize, rows: usize) -> String {
    let sub = rows * 2;
    let mut out = String::with_capacity(rows * cols * 20);
    for r in 0..rows {
        // Colours are only re-stated when they change, which for the flat
        // stretches of a field cuts the escape codes --- and so the bytes the
        // terminal has to chew through each frame --- by most of themselves.
        let mut last: Option<(u8, u8)> = None;
        for c in 0..cols {
            let top = terminal_colour(f, vort, peak, sample(f, c, cols, r * 2, sub));
            let bot = terminal_colour(f, vort, peak, sample(f, c, cols, r * 2 + 1, sub));
            match last {
                Some((t, b)) if t == top && b == bot => {}
                Some((t, _)) if t == top => {
                    let _ = write!(out, "\x1b[48;5;{bot}m");
                }
                Some((_, b)) if b == bot => {
                    let _ = write!(out, "\x1b[38;5;{top}m");
                }
                _ => {
                    let _ = write!(out, "\x1b[38;5;{top};48;5;{bot}m");
                }
            }
            last = Some((top, bot));
            out.push('\u{2580}');
        }
        out.push_str("\x1b[0m\n");
    }
    out
}

/// One block as a 256-colour index: the obstacle in grey, the fluid on the
/// same diverging scale the vorticity PNGs use, but running out of black so
/// that still water is the terminal's own background rather than a wall of
/// white.
fn terminal_colour(f: &Field, vort: &[f32], peak: f32, k: usize) -> u8 {
    if f.solid[k] > SOLID {
        return 244;
    }
    let m = shade(vort[k], peak);
    // Six levels a channel is not much, so hue does the work colour depth
    // cannot: each sign climbs black -> saturated -> bright -> near-white,
    // which reads as a gradient where a single channel would flatten out at
    // the top of its ramp.
    let (a, b, c) = (
        (m.abs() * 2.2).min(1.0),
        ((m.abs() - 0.45) / 0.55).clamp(0.0, 1.0),
        ((m.abs() - 0.80) / 0.20).clamp(0.0, 1.0) * 0.7,
    );
    if m >= 0.0 {
        cube(a, b * 0.95, c)
    } else {
        cube(c, b * 0.9, a)
    }
}

/// Nearest entry of xterm's 6x6x6 colour cube. 256 colours rather than 24-bit
/// because plenty of terminals in daily use --- Terminal.app among them ---
/// have no truecolor, and a field of vorticity has no need of more than 216
/// shades anyway.
fn cube(r: f32, g: f32, b: f32) -> u8 {
    // The cube's levels are not evenly spaced: 0, then 95 and 40 apart after.
    fn level(v: f32) -> u8 {
        let v = (v.clamp(0.0, 1.0) * 255.0).round();
        if v < 48.0 {
            0
        } else {
            (((v - 55.0) / 40.0).round().clamp(1.0, 5.0)) as u8
        }
    }
    16 + 36 * level(r) + 6 * level(g) + level(b)
}

/// The colourless view: block-element shades by magnitude, and a hatch for the
/// obstacle so it reads as a wall rather than as the strongest vorticity on
/// screen.
fn shaded_preview(f: &Field, vort: &[f32], peak: f32, cols: usize, rows: usize) -> String {
    let ramp = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];
    let mut out = String::with_capacity(rows * (cols + 1) * 3);
    for r in 0..rows {
        for c in 0..cols {
            let k = sample(f, c, cols, r, rows);
            if f.solid[k] > SOLID {
                out.push('\u{259e}');
            } else {
                let t = shade(vort[k], peak).abs() * (ramp.len() - 1) as f32;
                out.push(ramp[(t as usize).min(ramp.len() - 1)]);
            }
        }
        out.push('\n');
    }
    out
}
