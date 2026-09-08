//! A cellular-automaton fluid, after the model in *A New Kind of Science*
//! (Wolfram, pp. 378-380): identical particles hop between sites of a
//! hexagonal lattice and collide by momentum-conserving rules. Averaged over
//! blocks of cells, the discrete gas behaves like a continuum fluid, and past
//! an obstacle it sheds a von Karman vortex street.
//!
//! The simulation lives in this library so that both the `lgca` binary and the
//! benchmarks in `benches/` can drive it.

pub mod collision;
pub mod hex;
pub mod lattice;
pub mod moments;
pub mod png;
pub mod render;
pub mod rng;
pub mod transport;
