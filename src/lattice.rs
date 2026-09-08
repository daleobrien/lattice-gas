//! The lattice itself: propagation, collision, walls and the inflow boundary.

use crate::collision::CollisionTable;
use crate::hex::*;
use crate::moments;
use crate::rng::Rng;

pub struct Lattice {
    pub w: usize,
    pub h: usize,
    /// One byte per site: six moving bits, a rest bit, and `SOLID_BIT` on top
    /// for the sites an obstacle occupies.
    pub cells: Vec<u8>,
    scratch: Vec<u8>,
    pub table: CollisionTable,
    /// Mean occupancy of a single direction, in `0..1`.
    pub density: f32,
    pub inflow: (f32, f32),
    /// Columns at the left edge that are re-seeded from equilibrium each step.
    pub inlet_cols: usize,
    pub steps: u64,
    threads: usize,
    seed: u64,
}

/// Occupancy probabilities for a cell at the given density and mean velocity.
/// This is the usual first-order expansion of the equilibrium distribution;
/// the coefficient 2 is `D / c^2` for the two-dimensional unit-speed lattice.
///
/// Velocity is momentum per particle, and a rest particle carries mass but no
/// momentum. The moving directions therefore have to be biased harder --- by
/// the ratio of total mass to moving mass --- for the cell to actually come
/// out at the requested velocity.
///
/// The probabilities depend only on the macroscopic state, never on the site,
/// so they are computed once here and every site is then a few integer
/// compares. Sampling is what the inlet re-seed spends its time on, and it is
/// bound by the generator's dependency chain rather than by the arithmetic:
/// each draw has to finish before the next can start. Hence the packing below,
/// which takes two samples from one step of the generator instead of one.
pub struct Equilibrium {
    /// `ceil(p * 2^24)` per direction: a 24-bit draw `k` is a hit exactly when
    /// `k < t`.
    thresh: [u32; NDIR],
    /// `None` when the model has no rest particles, so the extra draw is
    /// skipped entirely rather than being made and discarded.
    rest: Option<u32>,
}

/// `p` as a 24-bit threshold, so that integer `k < threshold(p)` agrees with
/// the float `k as f32 / 2^24 < p` for every `k` a draw can produce.
///
/// `k / 2^24` is exact in `f32` (24-bit mantissa) and so is `p * 2^24` (it only
/// moves the exponent), so the two comparisons are over the same reals and
/// `ceil` is the exact integer boundary. Out-of-range `p` clamps to the same
/// always/never behaviour the float compare had, and a NaN `p` saturates to 0,
/// matching `k < NaN` being false.
#[inline]
fn threshold(p: f32) -> u32 {
    const ONE: f32 = 16_777_216.0; // 2^24
    let t = (p * ONE).ceil();
    if t >= ONE {
        ONE as u32
    } else if t > 0.0 {
        t as u32
    } else {
        0
    }
}

impl Equilibrium {
    pub fn new(density: f32, ux: f32, uy: f32, rest: bool) -> Self {
        let bias = if rest { 2.0 * 7.0 / 6.0 } else { 2.0 };
        let mut thresh = [0u32; NDIR];
        for d in 0..NDIR {
            thresh[d] = threshold(density * (1.0 + bias * (CXF[d] * ux + CYF[d] * uy)));
        }
        Equilibrium {
            thresh,
            rest: if rest { Some(threshold(density)) } else { None },
        }
    }

    /// The 24-bit thresholds themselves, rest particle last. The GPU builds
    /// its inlet from these rather than from its own copy of the expansion, so
    /// that both paths seed the inflow from the same distribution.
    pub fn thresholds(&self) -> [u32; NDIR + 1] {
        let mut t = [0u32; NDIR + 1];
        t[..NDIR].copy_from_slice(&self.thresh);
        t[NDIR] = self.rest.unwrap_or(0);
        t
    }

    /// One cell state. Six directions come from three steps of the generator,
    /// two 24-bit samples each.
    ///
    /// The samples are taken from bits 40..64 and 16..40. The top slice is the
    /// same one `Rng::next_f32` uses; the second is placed above the low
    /// sixteen bits, which are the weakest part of a multiply's output and the
    /// reason `next_u32` returns the high half in the first place.
    #[inline(always)]
    pub fn sample(&self, rng: &mut Rng) -> u8 {
        let mut s = 0u8;
        for d in (0..NDIR).step_by(2) {
            let r = rng.next_u64();
            s |= ((((r >> 40) as u32) < self.thresh[d]) as u8) << d;
            s |= (((((r >> 16) as u32) & 0xFF_FFFF) < self.thresh[d + 1]) as u8) << (d + 1);
        }
        if let Some(t) = self.rest {
            if ((rng.next_u64() >> 40) as u32) < t {
                s |= REST_BIT;
            }
        }
        s
    }
}

/// Sample a single cell without keeping the distribution around. Building the
/// table costs more than the draw does, so use `Equilibrium` directly for
/// anything that fills more than a handful of cells.
#[inline]
pub fn equilibrium_sample(density: f32, ux: f32, uy: f32, rest: bool, rng: &mut Rng) -> u8 {
    Equilibrium::new(density, ux, uy, rest).sample(rng)
}

impl Lattice {
    pub fn new(
        w: usize,
        h: usize,
        density: f32,
        inflow: (f32, f32),
        inlet_cols: usize,
        rest_particles: bool,
        threads: usize,
        seed: u64,
    ) -> Self {
        assert!(h % 2 == 0, "row count must be even for the lattice to wrap cleanly in y");
        assert!(w > 4 && h > 4, "lattice is too small");
        Lattice {
            w,
            h,
            cells: vec![0; w * h],
            scratch: vec![0; w * h],
            table: CollisionTable::build(rest_particles),
            density,
            inflow,
            inlet_cols,
            steps: 0,
            threads: threads.max(1),
            seed,
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    /// Is this site part of an obstacle?
    #[inline]
    pub fn is_solid(&self, i: usize) -> bool {
        self.cells[i] & SOLID_BIT != 0
    }

    /// Mark one site as obstacle, discarding whatever was standing on it.
    #[inline]
    pub fn set_solid(&mut self, i: usize) {
        self.cells[i] = SOLID_BIT;
    }

    /// Worker threads this lattice was built to use. The analysis passes read
    /// it so they spread themselves the same way the update does.
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The seed this lattice was built with, so that a second implementation
    /// of the same run can be given the same one.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Put the lattice back to a given state and random seed. The obstacles,
    /// the collision table and the inflow are whatever they already were, so
    /// this is a restart of the same experiment rather than a new one.
    pub fn restart(&mut self, cells: &[u8], seed: u64) {
        assert_eq!(cells.len(), self.cells.len(), "wrong number of cells");
        self.cells.copy_from_slice(cells);
        self.seed = seed;
        self.steps = 0;
    }

    /// Fill the whole domain with the inflow equilibrium so the run starts
    /// from uniform flow rather than from rest.
    pub fn init_equilibrium(&mut self) {
        let mut rng = Rng::new(self.seed ^ 0xA5A5_1234);
        let (ux, uy) = self.inflow;
        let eq = Equilibrium::new(self.density, ux, uy, self.table.rest_particles);
        for i in 0..self.cells.len() {
            self.cells[i] = if self.cells[i] & SOLID_BIT != 0 {
                SOLID_BIT
            } else {
                eq.sample(&mut rng)
            };
        }
    }

    pub fn add_solid<F: Fn(f32, f32) -> bool>(&mut self, inside: F) {
        for y in 0..self.h {
            for x in 0..self.w {
                let (px, py) = site_position(x, y);
                if inside(px, py) {
                    let i = self.idx(x, y);
                    self.set_solid(i);
                }
            }
        }
    }

    pub fn advance(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn step(&mut self) {
        let (w, h) = (self.w, self.h);
        let old: &[u8] = &self.cells;
        let table = &self.table;
        let nthreads = self.threads.min(h);
        let base_seed = self.seed ^ self.steps.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        // Rows per thread, rounded up.
        let band = (h + nthreads - 1) / nthreads;
        let out = &mut self.scratch;

        std::thread::scope(|scope| {
            for (b, chunk) in out.chunks_mut(band * w).enumerate() {
                let y0 = b * band;
                let mut rng = Rng::new(base_seed ^ ((b as u64 + 1) << 32));
                scope.spawn(move || {
                    let rows = chunk.len() / w;
                    for r in 0..rows {
                        let y = y0 + r;
                        stream_and_collide_row(
                            old,
                            &mut chunk[r * w..(r + 1) * w],
                            table,
                            &mut rng,
                            y,
                            w,
                            h,
                        );
                    }
                });
            }
        });

        std::mem::swap(&mut self.cells, &mut self.scratch);
        self.steps += 1;
        self.apply_inlet();
    }

    /// Re-seed the leftmost columns from the equilibrium distribution. This is
    /// what maintains the mean flow; particles that leave the right-hand end
    /// wrap around into this zone and are overwritten, so the outflow is
    /// effectively absorbing.
    fn apply_inlet(&mut self) {
        if self.inlet_cols == 0 {
            return;
        }
        let (ux, uy) = self.inflow;
        let eq = Equilibrium::new(self.density, ux, uy, self.table.rest_particles);
        let mut rng = Rng::new(self.seed ^ self.steps.wrapping_mul(0xD1B5_4A32_D192_ED03));
        for y in 0..self.h {
            for x in 0..self.inlet_cols.min(self.w) {
                let i = y * self.w + x;
                if self.cells[i] & SOLID_BIT == 0 {
                    self.cells[i] = eq.sample(&mut rng);
                }
            }
        }
    }

    /// Momentum per particle over the whole lattice.
    ///
    /// This counts the particles standing on obstacle sites too, which are the
    /// ones a wall is in the middle of turning round. That is what it has
    /// always done; `Field::sample` takes the other view and leaves them out.
    pub fn mean_velocity(&self) -> (f32, f32) {
        moments::sum(&moments::WITH_WALLS, &self.cells).velocity()
    }

    /// Particles on the lattice, obstacle sites included. `SOLID_BIT` has to be
    /// masked off first, so this counts eight cells per popcount rather than
    /// paying for the mask once per byte.
    pub fn total_particles(&self) -> u64 {
        const MASK: u64 = 0x7F7F_7F7F_7F7F_7F7F;
        let mut words = self.cells.chunks_exact(8);
        let mut n = 0u64;
        for w in words.by_ref() {
            let bits = u64::from_le_bytes(w.try_into().unwrap());
            n += (bits & MASK).count_ones() as u64;
        }
        n + words
            .remainder()
            .iter()
            .map(|c| (c & STATE_MASK).count_ones() as u64)
            .sum::<u64>()
    }

}

/// One row of the fused propagate-then-collide update.
///
/// Propagation is done as a gather: the bit for direction `d` at site `c` is
/// whatever sat in direction `d` at the site one step *behind* `c`, which is
/// `c`'s neighbour in the opposite direction. Gathering (rather than
/// scattering) keeps each output row independent, so rows parallelise freely.
#[inline]
#[allow(clippy::too_many_arguments)]
fn stream_and_collide_row(
    old: &[u8],
    out: &mut [u8],
    table: &CollisionTable,
    rng: &mut Rng,
    y: usize,
    w: usize,
    h: usize,
) {
    let dxtab = if y & 1 == 1 { &DX_ODD } else { &DX_EVEN };

    let mut sdx = [0i32; NDIR];
    let mut srow = [0usize; NDIR];
    for d in 0..NDIR {
        let back = OPP[d];
        sdx[d] = dxtab[back];
        let sy = (y as i32 + DY[back]).rem_euclid(h as i32) as usize;
        srow[d] = sy * w;
    }

    let row = y * w;
    for x in 0..w {
        // The one load covers both the rest bit and whether this is a wall.
        let here = old[row + x];
        let mut state = here & REST_BIT;
        for d in 0..NDIR {
            let sx = if x == 0 || x == w - 1 {
                (x as i32 + sdx[d]).rem_euclid(w as i32) as usize
            } else {
                (x as i32 + sdx[d]) as usize
            };
            state |= old[srow[d] + sx] & (1 << d);
        }

        out[x] = if here & SOLID_BIT != 0 {
            SOLID_BIT | reverse(state)
        } else {
            table.apply(state, rng.next_u32())
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn totals(lat: &Lattice) -> (i64, i64, i64) {
        let (mut mass, mut px, mut py) = (0i64, 0i64, 0i64);
        for &c in &lat.cells {
            for d in 0..NDIR {
                if c & (1 << d) != 0 {
                    mass += 1;
                    px += CX2[d] as i64;
                    py += CY2[d] as i64;
                }
            }
            if c & REST_BIT != 0 {
                mass += 1;
            }
        }
        (mass, px, py)
    }

    fn random_box(rest: bool) -> Lattice {
        let mut lat = Lattice::new(32, 32, 0.2, (0.0, 0.0), 0, rest, 1, 12345);
        let mut rng = Rng::new(7);
        for c in lat.cells.iter_mut() {
            *c = (rng.next_u32() as u8) & if rest { STATE_MASK } else { MOVING_MASK };
        }
        lat
    }

    #[test]
    fn periodic_box_conserves_mass_and_momentum() {
        for &rest in &[false, true] {
            let mut lat = random_box(rest);
            let before = totals(&lat);
            lat.advance(200);
            assert_eq!(before, totals(&lat), "rest particles = {rest}");
        }
    }

    /// A lone particle has no collision partner, so it must fly in a straight
    /// line. This pins down the neighbour offsets for both row parities.
    #[test]
    fn single_particle_travels_in_a_straight_line() {
        for d in 0..NDIR {
            for start_row in [10usize, 11] {
                let mut lat = Lattice::new(48, 48, 0.2, (0.0, 0.0), 0, false, 1, 1);
                let (x0, y0) = (24usize, start_row);
                let i = lat.idx(x0, y0);
                lat.cells[i] = 1 << d;

                let steps = 8;
                lat.advance(steps as u64);

                let occupied: Vec<usize> = (0..lat.cells.len())
                    .filter(|&i| lat.cells[i] & STATE_MASK != 0)
                    .collect();
                assert_eq!(occupied.len(), 1, "direction {d}: particle was lost");
                assert_eq!(lat.cells[occupied[0]], 1 << d, "direction {d}: turned");

                let (x, y) = (occupied[0] % lat.w, occupied[0] / lat.w);
                let (px, py) = site_position(x, y);
                let (sx, sy) = site_position(x0, y0);
                assert!(
                    (px - (sx + steps as f32 * CXF[d])).abs() < 1e-3
                        && (py - (sy + steps as f32 * CYF[d])).abs() < 1e-3,
                    "direction {d} from row {start_row}: went to ({px}, {py}), \
                     expected ({}, {})",
                    sx + steps as f32 * CXF[d],
                    sy + steps as f32 * CYF[d]
                );
            }
        }
    }

    /// A particle that runs into a wall comes back the way it came.
    #[test]
    fn walls_reverse_particles() {
        let mut lat = Lattice::new(48, 48, 0.2, (0.0, 0.0), 0, false, 1, 1);
        let (x0, y0) = (20usize, 24usize);
        let wall = lat.idx(x0 + 3, y0);
        lat.set_solid(wall);
        let start = lat.idx(x0, y0);
        lat.cells[start] = 1; // heading east

        lat.advance(6);

        let occupied: Vec<usize> = (0..lat.cells.len())
            .filter(|&i| lat.cells[i] & STATE_MASK != 0)
            .collect();
        assert_eq!(occupied.len(), 1);
        assert_eq!(
            lat.cells[occupied[0]] & STATE_MASK,
            1 << 3,
            "should now head west"
        );
        // Three steps to reach the wall site, where it is turned round, then
        // three steps back: it ends up exactly where it started.
        assert_eq!(occupied[0], start);
    }

    /// Every direction must come out at its requested rate. Directions are
    /// sampled in pairs from one step of the generator --- even ones from bits
    /// 40..64, odd ones from 16..40 --- so a weak slice would show up here as
    /// a systematic split between the two.
    #[test]
    fn every_direction_hits_its_requested_rate() {
        const N: usize = 1_000_000;
        for &(density, ux, uy) in &[(0.22f32, 0.3f32, 0.0f32), (0.5, 0.0, 0.2)] {
            for &rest in &[false, true] {
                let bias: f32 = if rest { 2.0 * 7.0 / 6.0 } else { 2.0 };
                let eq = Equilibrium::new(density, ux, uy, rest);
                let mut rng = Rng::new(0xC0FFEE);
                let mut hits = [0u64; NDIR];
                for _ in 0..N {
                    let s = eq.sample(&mut rng);
                    for d in 0..NDIR {
                        hits[d] += ((s >> d) & 1) as u64;
                    }
                }
                for d in 0..NDIR {
                    let want = (density * (1.0 + bias * (CXF[d] * ux + CYF[d] * uy))) as f64;
                    let got = hits[d] as f64 / N as f64;
                    let sigma = (want * (1.0 - want) / N as f64).sqrt();
                    assert!(
                        (got - want).abs() < 5.0 * sigma,
                        "direction {d}, rest = {rest}, u = ({ux}, {uy}): rate {got:.6} \
                         against a requested {want:.6}, {:.1} sigma out",
                        (got - want).abs() / sigma
                    );
                }
            }
        }
    }

    /// Taking two samples from one step of the generator is only sound if the
    /// two stay independent; drawing them from overlapping bits would not.
    #[test]
    fn paired_directions_are_uncorrelated() {
        const N: usize = 1_000_000;
        let eq = Equilibrium::new(0.5, 0.0, 0.0, false);
        let mut rng = Rng::new(0xBEEF);
        let mut joint = [[0u64; NDIR]; NDIR];
        let mut single = [0u64; NDIR];
        for _ in 0..N {
            let s = eq.sample(&mut rng);
            for a in 0..NDIR {
                single[a] += ((s >> a) & 1) as u64;
                for b in 0..NDIR {
                    joint[a][b] += ((s >> a) & (s >> b) & 1) as u64;
                }
            }
        }
        for a in 0..NDIR {
            for b in (a + 1)..NDIR {
                let (pa, pb) = (single[a] as f64 / N as f64, single[b] as f64 / N as f64);
                let pab = joint[a][b] as f64 / N as f64;
                let corr =
                    (pab - pa * pb) / ((pa * (1.0 - pa)).sqrt() * (pb * (1.0 - pb)).sqrt());
                assert!(
                    corr.abs() < 0.01,
                    "directions {a} and {b} are correlated at {corr:.5}"
                );
            }
        }
    }

    /// The equilibrium sampler must deliver the velocity it is asked for, with
    /// or without stationary particles.
    #[test]
    fn equilibrium_hits_the_requested_velocity() {
        for &rest in &[false, true] {
            let mut rng = Rng::new(99);
            let (target, density) = (0.3f32, 0.2f32);
            let (mut px, mut mass) = (0.0f64, 0.0f64);
            for _ in 0..200_000 {
                let c = equilibrium_sample(density, target, 0.0, rest, &mut rng);
                for d in 0..NDIR {
                    if c & (1 << d) != 0 {
                        px += CXF[d] as f64;
                        mass += 1.0;
                    }
                }
                if c & REST_BIT != 0 {
                    mass += 1.0;
                }
            }
            let u = px / mass;
            assert!(
                (u - target as f64).abs() < 0.01,
                "rest = {rest}: asked for {target}, got {u}"
            );
        }
    }
}
