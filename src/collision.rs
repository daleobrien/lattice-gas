//! Collision rules.
//!
//! Rather than hand-listing the FHP collision cases, we enumerate every cell
//! state and group states that share the same particle count and the same
//! total momentum. A collision replaces a state by one drawn uniformly from
//! its own group. That conserves mass and momentum exactly, is symmetric (so
//! semi-detailed balance holds and the equilibrium is the usual Fermi-Dirac
//! one), and it fires on every configuration that has an alternative at all --
//! which is what keeps the viscosity, and therefore the Reynolds number,
//! favourable.

use crate::hex::{CX2, CY2, NDIR, REST_BIT};
use std::collections::HashMap;

pub struct CollisionTable {
    /// Outcomes per state, padded to a common width so that a uniform draw
    /// over `stride` is a uniform draw over the group.
    stride: usize,
    data: Vec<u8>,
    pub n_states: usize,
    pub rest_particles: bool,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// Mass and (exact, integer) momentum of a state.
fn invariants(state: u8) -> (u32, i32, i32) {
    let mut mass = 0;
    let mut px = 0;
    let mut py = 0;
    for d in 0..NDIR {
        if state & (1 << d) != 0 {
            mass += 1;
            px += CX2[d];
            py += CY2[d];
        }
    }
    if state & REST_BIT != 0 {
        mass += 1;
    }
    (mass, px, py)
}

impl CollisionTable {
    pub fn build(rest_particles: bool) -> Self {
        let n_states = if rest_particles { 128 } else { 64 };

        let mut groups: HashMap<(u32, i32, i32), Vec<u8>> = HashMap::new();
        for s in 0..n_states {
            groups.entry(invariants(s as u8)).or_default().push(s as u8);
        }

        // A single width that every group size divides, so padding by
        // repetition stays unbiased.
        let stride = groups.values().fold(1usize, |acc, g| lcm(acc, g.len()));

        let mut data = vec![0u8; n_states * stride];
        for s in 0..n_states {
            let g = &groups[&invariants(s as u8)];
            for k in 0..stride {
                data[s * stride + k] = g[k % g.len()];
            }
        }

        CollisionTable {
            stride,
            data,
            n_states,
            rest_particles,
        }
    }

    /// Number of states that actually have somewhere else to go.
    pub fn active_states(&self) -> usize {
        (0..self.n_states)
            .filter(|&s| {
                let row = &self.data[s * self.stride..(s + 1) * self.stride];
                row.iter().any(|&o| o as usize != s)
            })
            .count()
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline(always)]
    pub fn apply(&self, state: u8, rand: u32) -> u8 {
        // Multiply-shift maps a u32 uniformly onto 0..stride.
        let k = ((rand as u64 * self.stride as u64) >> 32) as usize;
        unsafe { *self.data.get_unchecked(state as usize * self.stride + k) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collisions_conserve_mass_and_momentum() {
        for &rest in &[false, true] {
            let t = CollisionTable::build(rest);
            for s in 0..t.n_states {
                for k in 0..t.stride {
                    let out = t.data[s * t.stride + k];
                    assert_eq!(invariants(s as u8), invariants(out));
                }
            }
        }
    }

    #[test]
    fn groups_are_closed() {
        // If a maps to b, b must be able to map back to a.
        let t = CollisionTable::build(true);
        for s in 0..t.n_states {
            for k in 0..t.stride {
                let out = t.data[s * t.stride + k] as usize;
                assert!(t.data[out * t.stride..(out + 1) * t.stride]
                    .contains(&(s as u8)));
            }
        }
    }
}
