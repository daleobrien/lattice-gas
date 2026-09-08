//! Mass and momentum of a cell, looked up rather than taken apart bit by bit.
//!
//! `Lattice::total_particles` reads the cell array at 58 GB/s. Coarse-graining
//! reads the same array, byte for byte, five hundred times slower, because it
//! tests six bits and does three `f32` adds per cell. A cell is one byte with
//! 256 possible values, so its moments can simply be a table.
//!
//! Each entry packs four 16-bit fields into a `u64`. They cannot carry into
//! one another for any block small enough (see `MAX_BLOCK`), so a single `u64`
//! add per cell accumulates all four at once. Momentum is held in the exact
//! integer units of `CX2`/`CY2`, biased to stay non-negative; the bias comes
//! off once per block rather than once per cell. Summing in integers is also
//! more accurate than the `f32` running total it replaces, which was adding
//! several thousand values of alternating sign.

use crate::hex::{CX2, CY2, NDIR, REST_BIT, SOLID_BIT, SQRT3_2};

/// Field positions within a packed accumulator.
const MASS: u32 = 0;
const PX: u32 = 16;
const PY: u32 = 32;
const SOLID: u32 = 48;

/// Per-cell bias on each momentum field. `CX2` sums to -4 at worst and `CY2`
/// to -2, so these are what it takes to keep every field non-negative.
const PX_BIAS: i64 = 4;
const PY_BIAS: i64 = 2;

/// The largest run of cells a 16-bit field can hold. The biggest per-cell
/// contribution to any field is `PX_BIAS + 4 = 8`, so this is `65535 / 8`.
/// The default block is 400 cells.
pub const MAX_BLOCK: usize = u16::MAX as usize / 8;

/// `walls` decides whether a solid cell's particles count. A wall reflects
/// rather than absorbs, so a solid site does hold particles mid-bounce, and
/// the two callers disagree about whether they are part of the fluid.
const fn table(walls: bool) -> [u64; 256] {
    let mut t = [0u64; 256];
    let mut s = 0usize;
    while s < 256 {
        let c = s as u8;
        let solid = c & SOLID_BIT != 0;
        let (mut mass, mut px, mut py) = (0i64, 0i64, 0i64);
        if walls || !solid {
            let mut d = 0;
            while d < NDIR {
                if c & (1 << d) != 0 {
                    mass += 1;
                    px += CX2[d] as i64;
                    py += CY2[d] as i64;
                }
                d += 1;
            }
            if c & REST_BIT != 0 {
                mass += 1;
            }
        }
        t[s] = ((mass as u64) << MASS)
            | (((px + PX_BIAS) as u64) << PX)
            | (((py + PY_BIAS) as u64) << PY)
            | ((solid as u64) << SOLID);
        s += 1;
    }
    t
}

/// The fluid only: a solid cell contributes nothing but its own count. This is
/// what coarse-graining wants, since a block that is half obstacle should
/// report the velocity of the half that is fluid.
pub static FLUID: [u64; 256] = table(false);

/// Every particle present, including the ones a wall is in the middle of
/// turning round. `Lattice::mean_velocity` has always counted those.
pub static WITH_WALLS: [u64; 256] = table(true);

/// The moments of some set of cells, in exact integer units: momentum is in
/// halves of a lattice unit in x and in units of `sqrt(3)/2` in y, matching
/// `CX2` and `CY2`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Moments {
    pub mass: i64,
    pub px2: i64,
    pub py2: i64,
    pub solid: i64,
}

impl Moments {
    /// Momentum per particle, in lattice units. Zero for an empty set, as the
    /// bit-by-bit version returned.
    #[inline]
    pub fn velocity(&self) -> (f32, f32) {
        if self.mass == 0 {
            return (0.0, 0.0);
        }
        let m = self.mass as f32;
        (self.px2 as f32 * 0.5 / m, self.py2 as f32 * SQRT3_2 / m)
    }
}

/// Undo the packing. `n` is how many cells went into `acc`, which is what the
/// momentum bias has to be measured against.
#[inline]
pub fn unpack(acc: u64, n: usize) -> Moments {
    Moments {
        mass: ((acc >> MASS) & 0xFFFF) as i64,
        px2: (((acc >> PX) & 0xFFFF) as i64) - PX_BIAS * n as i64,
        py2: (((acc >> PY) & 0xFFFF) as i64) - PY_BIAS * n as i64,
        solid: ((acc >> SOLID) & 0xFFFF) as i64,
    }
}

/// Accumulate a run of cells into a packed sum. The caller is responsible for
/// keeping the total behind any one accumulator under `MAX_BLOCK`; the run
/// itself may be shorter, since a block is usually gathered a row at a time.
#[inline(always)]
pub fn accumulate(table: &[u64; 256], cells: &[u8], acc: &mut u64) {
    let mut a = 0u64;
    for &c in cells {
        a += table[c as usize];
    }
    *acc += a;
}

/// Moments of a whole slice of any length, chunked so the fields cannot
/// overflow.
pub fn sum(table: &[u64; 256], cells: &[u8]) -> Moments {
    let mut total = Moments::default();
    for chunk in cells.chunks(MAX_BLOCK) {
        let mut acc = 0u64;
        accumulate(table, chunk, &mut acc);
        let m = unpack(acc, chunk.len());
        total.mass += m.mass;
        total.px2 += m.px2;
        total.py2 += m.py2;
        total.solid += m.solid;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::{CXF, CYF, MOVING_MASK};

    /// The table has to agree with taking the cell apart by hand, for every
    /// one of the 256 bytes and both answers to the wall question.
    #[test]
    fn table_agrees_with_the_bit_by_bit_version() {
        for s in 0..256usize {
            let c = s as u8;
            let (mut mass, mut px, mut py) = (0.0f32, 0.0f32, 0.0f32);
            if c & SOLID_BIT == 0 {
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
            let m = unpack(FLUID[c as usize], 1);
            assert_eq!(m.mass as f32, mass, "state {c:#010b}");
            assert_eq!(m.px2 as f32 * 0.5, px, "state {c:#010b}");
            assert!((m.py2 as f32 * SQRT3_2 - py).abs() < 1e-6, "state {c:#010b}");
            assert_eq!(m.solid, (c & SOLID_BIT != 0) as i64);

            // With walls, the solid bit changes nothing but the counter.
            let w = unpack(WITH_WALLS[c as usize], 1);
            let bare = unpack(WITH_WALLS[(c & !SOLID_BIT) as usize], 1);
            assert_eq!((w.mass, w.px2, w.py2), (bare.mass, bare.px2, bare.py2));
        }
    }

    /// A full block of the densest possible cells must not carry between
    /// fields. This is the bound `MAX_BLOCK` is claiming.
    #[test]
    fn a_full_block_does_not_overflow_its_fields() {
        for &c in &[0u8, MOVING_MASK, MOVING_MASK | REST_BIT, 0b0100_0001, SOLID_BIT | 0x7F] {
            let cells = vec![c; MAX_BLOCK];
            let packed = sum(&WITH_WALLS, &cells);
            let one = unpack(WITH_WALLS[c as usize], 1);
            assert_eq!(packed.mass, one.mass * MAX_BLOCK as i64, "state {c:#010b}");
            assert_eq!(packed.px2, one.px2 * MAX_BLOCK as i64, "state {c:#010b}");
            assert_eq!(packed.py2, one.py2 * MAX_BLOCK as i64, "state {c:#010b}");
            assert_eq!(packed.solid, one.solid * MAX_BLOCK as i64, "state {c:#010b}");
        }
    }

    /// Chunking a long slice must give the same answer as one short one.
    #[test]
    fn chunking_is_invisible() {
        let mut cells = Vec::new();
        let mut x = 1u32;
        for _ in 0..MAX_BLOCK * 3 + 17 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            cells.push((x >> 24) as u8);
        }
        let whole = sum(&FLUID, &cells);
        let mut halves = Moments::default();
        for part in cells.chunks(1000) {
            let m = sum(&FLUID, part);
            halves.mass += m.mass;
            halves.px2 += m.px2;
            halves.py2 += m.py2;
            halves.solid += m.solid;
        }
        assert_eq!(whole, halves);
    }
}
