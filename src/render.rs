//! Coarse-graining and output.
//!
//! A single cell is either occupied or not, so the raw lattice looks like
//! noise. The fluid appears only after averaging: the book uses blocks of
//! 20x20 cells, and we do the same, optionally with a running time average on
//! top to quieten the residual fluctuation.

use crate::hex::{CXF, CYF, NDIR, REST_BIT, SQRT3_2};
use crate::lattice::Lattice;
use std::path::Path;

/// A block counts as obstacle for display purposes above this solid fraction.
/// A thin plate only ever fills a fraction of a coarse block, so the threshold
/// has to sit below that.
const SOLID: f32 = 0.12;

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
    pub fn sample(&mut self, lat: &Lattice, alpha: f32) {
        let n = (self.bx * self.by) as f32;
        for by in 0..self.bh {
            for bx in 0..self.bw {
                let (mut px, mut py, mut mass, mut solid) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for y in by * self.by..(by + 1) * self.by {
                    let row = y * lat.w;
                    for x in bx * self.bx..(bx + 1) * self.bx {
                        let i = row + x;
                        if lat.solid[i] {
                            solid += 1.0;
                            continue;
                        }
                        let c = lat.cells[i];
                        for d in 0..NDIR {
                            if c & (1 << d) != 0 {
                                px += CXF[d];
                                py += CYF[d];
                                mass += 1.0;
                            }
                        }
                        if c & REST_BIT != 0 {
                            mass += 1.0;
                        }
                    }
                }
                let k = by * self.bw + bx;
                let (vx, vy) = if mass > 0.0 { (px / mass, py / mass) } else { (0.0, 0.0) };
                self.ux[k] += alpha * (vx - self.ux[k]);
                self.uy[k] += alpha * (vy - self.uy[k]);
                self.rho[k] += alpha * (mass / n - self.rho[k]);
                self.solid[k] = solid / n;
            }
        }
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

/// Rough terminal view of the vorticity, for watching a run without leaving
/// the shell.
pub fn ascii_preview(f: &Field, cols: usize, rows: usize) -> String {
    let vort = f.vorticity();
    let mut peak = 1e-9f32;
    for (k, v) in vort.iter().enumerate() {
        if f.solid[k] < SOLID {
            peak = peak.max(v.abs());
        }
    }
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let mut out = String::new();
    for r in 0..rows {
        let j = f.bh - 1 - r * f.bh / rows;
        for c in 0..cols {
            let i = (c * f.bw / cols).min(f.bw - 1);
            let k = j * f.bw + i;
            if f.solid[k] > SOLID {
                out.push('8');
            } else {
                let t = (vort[k].abs() / peak * (ramp.len() - 1) as f32) as usize;
                out.push(ramp[t.min(ramp.len() - 1)]);
            }
        }
        out.push('\n');
    }
    out
}
