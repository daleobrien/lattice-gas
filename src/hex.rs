//! Geometry of the triangular (hexagonal-neighbourhood) lattice.
//!
//! Sites sit on rows spaced `sqrt(3)/2` apart; odd rows are offset by half a
//! column, so every site has six neighbours at unit distance. Direction 0 is
//! east, and the rest run counter-clockwise.

pub const NDIR: usize = 6;

/// Bit 6 of a cell state holds a stationary ("rest") particle.
pub const REST_BIT: u8 = 1 << 6;

/// Bit 7 marks the site as part of an obstacle.
///
/// This lives in the cell byte rather than in a parallel array because both
/// hot loops already have that byte in hand: the update loads it for the rest
/// bit, and the coarse-graining loads it for everything. A separate
/// `Vec<bool>` meant streaming a second copy of the lattice past both of them
/// for one bit of information.
pub const SOLID_BIT: u8 = 1 << 7;

/// The particle content of a cell: six moving bits and the rest bit.
pub const STATE_MASK: u8 = 0b0111_1111;
pub const MOVING_MASK: u8 = 0b0011_1111;

pub const SQRT3_2: f32 = 0.866_025_4;

/// x-components in half-units, y-components in units of sqrt(3)/2, so that
/// momentum can be accumulated in exact integers.
pub const CX2: [i32; NDIR] = [2, 1, -1, -2, -1, 1];
pub const CY2: [i32; NDIR] = [0, 1, 1, 0, -1, -1];

pub const CXF: [f32; NDIR] = [1.0, 0.5, -0.5, -1.0, -0.5, 0.5];
pub const CYF: [f32; NDIR] = [0.0, SQRT3_2, SQRT3_2, 0.0, -SQRT3_2, -SQRT3_2];

/// Opposite direction, i.e. `(d + 3) % 6`.
pub const OPP: [usize; NDIR] = [3, 4, 5, 0, 1, 2];

/// Column offsets to a neighbour, for sites on even and odd rows.
pub const DX_EVEN: [i32; NDIR] = [1, 0, -1, -1, -1, 0];
pub const DX_ODD: [i32; NDIR] = [1, 1, 0, -1, 0, 1];
pub const DY: [i32; NDIR] = [0, 1, 1, 0, -1, -1];

/// Reverse the six moving bits; a rest particle is left alone.
#[inline(always)]
pub fn reverse(state: u8) -> u8 {
    let m = state & MOVING_MASK;
    (state & REST_BIT) | (((m << 3) | (m >> 3)) & MOVING_MASK)
}

/// Physical position of lattice site `(x, y)`.
#[inline]
pub fn site_position(x: usize, y: usize) -> (f32, f32) {
    let px = x as f32 + if y & 1 == 1 { 0.5 } else { 0.0 };
    (px, y as f32 * SQRT3_2)
}
