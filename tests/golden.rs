//! Golden-output regression tests, for use when optimising.
//!
//! `cargo test` already checks that the physics is *right* -- conservation,
//! straight-line flight, wall reflection. These check that it is *unchanged*,
//! which is the question a performance change actually raises.
//!
//! There are two layers, and the difference between them is the whole point.
//!
//! **Exact.** A checksum of the cell array, of the coarse-grained field, and
//! the exact integer mass and momentum, after a fixed number of steps from a
//! fixed seed. These pin the simulation bit for bit. They will change if the
//! optimisation alters the number or the order of the random draws -- widening
//! the collision loop to SIMD, say, or re-banding the threads. That is a
//! legitimate thing to do, so a failure here is a question, not a verdict:
//! *did I mean to change the random stream?* If yes, regenerate the table (see
//! below) and check that the layer below still passes.
//!
//! **Behavioural.** Mean velocity and particle count, with tolerances wide
//! enough to survive a completely different random stream. These must not
//! change, whatever the optimisation does. Together with
//! `thread_count_does_not_change_conserved_quantities` and
//! `stepping_is_deterministic`, this is the layer that says the fluid is still
//! the same fluid.
//!
//! To regenerate the table after a deliberate change:
//!
//! ```text
//! cargo test --test golden -- --ignored --nocapture print_golden
//! ```
//!
//! and paste the printed block over `GOLDEN` below.

use lattice_gas::hex::{CX2, CY2, NDIR, REST_BIT, SQRT3_2};
use lattice_gas::lattice::Lattice;
use lattice_gas::render::Field;

/// FNV-1a. `std`'s hashers make no promise of stability across releases, and a
/// golden value that quietly changes when the toolchain does is worse than no
/// golden value at all.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

const DENSITY: f32 = 0.22;
const SPEED: f32 = 0.3;
const SEED: u64 = 0x5EED;
const BLOCK: usize = 16;

struct Case {
    name: &'static str,
    w: usize,
    h: usize,
    rest: bool,
    inlet: usize,
    plate: bool,
    threads: usize,
    steps: u64,
}

/// Between them these cover both collision tables, the solid-cell branch, the
/// serial inlet re-seed, and thread bands that do and do not divide the row
/// count evenly (128 rows over 3 threads gives bands of 43, 43 and 42).
const CASES: &[Case] = &[
    Case { name: "periodic/rest/t1",       w: 128, h: 128, rest: true,  inlet: 0, plate: false, threads: 1, steps: 200 },
    Case { name: "periodic/no-rest/t1",    w: 128, h: 128, rest: false, inlet: 0, plate: false, threads: 1, steps: 200 },
    Case { name: "periodic/rest/t3",       w: 128, h: 128, rest: true,  inlet: 0, plate: false, threads: 3, steps: 200 },
    Case { name: "channel/plate/inlet/t1", w: 128, h: 64,  rest: true,  inlet: 8, plate: true,  threads: 1, steps: 200 },
    Case { name: "channel/plate/inlet/t4", w: 128, h: 64,  rest: true,  inlet: 8, plate: true,  threads: 4, steps: 200 },
];

fn build(c: &Case, seed: u64) -> Lattice {
    let mut lat = Lattice::new(c.w, c.h, DENSITY, (SPEED, 0.0), c.inlet, c.rest, c.threads, seed);
    if c.plate {
        let yc = c.h as f32 * SQRT3_2 / 2.0;
        let (x0, x1) = (c.w as f32 / 5.0, c.w as f32 / 5.0 + 3.0);
        let (y0, y1) = (yc - 12.0, yc + 12.0);
        lat.add_solid(move |x, y| x >= x0 && x <= x1 && y >= y0 && y <= y1);
    }
    lat.init_equilibrium();
    lat
}

/// Exact mass and momentum. Momentum is in the half-unit integer coordinates
/// of `hex`, so it is exact rather than merely close.
fn invariants(lat: &Lattice) -> (i64, i64, i64) {
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

#[derive(PartialEq)]
struct Outcome {
    cells: u64,
    field: u64,
    mass: i64,
    px: i64,
    py: i64,
    ux: f32,
    uy: f32,
}

fn run(c: &Case, seed: u64) -> Outcome {
    let mut lat = build(c, seed);
    lat.advance(c.steps);

    let mut f = Field::new(&lat, BLOCK, BLOCK);
    f.sample(&lat, 1.0);
    let mut fb = Vec::with_capacity(f.ux.len() * 16);
    for v in f.ux.iter().chain(&f.uy).chain(&f.rho).chain(&f.solid) {
        fb.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    let (mass, px, py) = invariants(&lat);
    let (ux, uy) = lat.mean_velocity();
    Outcome { cells: fnv1a(&lat.cells), field: fnv1a(&fb), mass, px, py, ux, uy }
}

// ---------------------------------------------------------------------------
// The recorded output. Regenerate with the command in the module docs.
// ---------------------------------------------------------------------------

/// `(name, cell checksum, field checksum, mass, momentum x, momentum y, ux, uy)`
type Golden = (&'static str, u64, u64, i64, i64, i64, f32, f32);

const GOLDEN: &[Golden] = &[
    ("periodic/rest/t1", 0x858e03f21757be07, 0xd8e827046a8b9333, 25377, 15297, 217, 0.301395, 0.007405),
    ("periodic/no-rest/t1", 0xbf98e4154901aee5, 0x588e7d1af89f9407, 21845, 12822, -30, 0.293477, -0.001189),
    ("periodic/rest/t3", 0x9023e6212bf3db23, 0x908404c2b273513a, 25377, 15297, 217, 0.301395, 0.007405),
    ("channel/plate/inlet/t1", 0x29bae611004551e5, 0xd4740a4904d80cbd, 11516, 6377, -69, 0.276876, -0.005189),
    ("channel/plate/inlet/t4", 0x4a0fe07a15e75307, 0xc9fb2ebd12b3ad9e, 11539, 6455, -11, 0.279704, -0.000826),
];

fn golden(name: &str) -> Option<&'static Golden> {
    GOLDEN.iter().find(|g| g.0 == name)
}

// ---------------------------------------------------------------------------

/// Layer one: the simulation is unchanged bit for bit.
#[test]
fn golden_output_is_unchanged() {
    assert!(
        !GOLDEN.is_empty(),
        "no golden table recorded; run\n  \
         cargo test --test golden -- --ignored --nocapture print_golden\n\
         and paste the result into GOLDEN in tests/golden.rs"
    );

    let mut bad = Vec::new();
    for c in CASES {
        let got = run(c, SEED);
        let Some(g) = golden(c.name) else {
            bad.push(format!("{}: no golden entry recorded", c.name));
            continue;
        };
        if got.cells != g.1 {
            bad.push(format!("{}: cells 0x{:016x}, recorded 0x{:016x}", c.name, got.cells, g.1));
        }
        if got.field != g.2 {
            bad.push(format!("{}: field 0x{:016x}, recorded 0x{:016x}", c.name, got.field, g.2));
        }
        if (got.mass, got.px, got.py) != (g.3, g.4, g.5) {
            bad.push(format!(
                "{}: mass/momentum ({}, {}, {}), recorded ({}, {}, {})",
                c.name, got.mass, got.px, got.py, g.3, g.4, g.5
            ));
        }
    }

    assert!(
        bad.is_empty(),
        "the simulation's output has changed:\n  {}\n\n\
         If the random draws were deliberately reordered -- a wider collision \n\
         loop, a different thread banding -- this is expected. Check that \n\
         physics_is_unchanged still passes, then regenerate the table with\n  \
         cargo test --test golden -- --ignored --nocapture print_golden\n\
         and say in the commit message why the stream moved.\n\
         Otherwise the change altered the simulation, which was not the point.",
        bad.join("\n  ")
    );
}

/// Layer two: whatever the random stream does, the fluid is the same fluid.
///
/// The tolerances are deliberately loose. Measured over sixteen different
/// seeds, the spread of these cases is up to 0.0065 in velocity and 0.9% in
/// mass, so 0.05 and 5% sit five to eight standard deviations out: a different
/// random stream will not trip them. This is a coarse net for "the flow
/// stopped" or "the density collapsed", not a precision check; precision lives
/// in the unit tests and in layer one.
#[test]
fn physics_is_unchanged() {
    const U_TOL: f32 = 0.05;
    const MASS_TOL: f64 = 0.05;

    for c in CASES {
        let Some(g) = golden(c.name) else { continue };
        let got = run(c, SEED);

        assert!(
            (got.ux - g.6).abs() < U_TOL && (got.uy - g.7).abs() < U_TOL,
            "{}: mean velocity ({:+.4}, {:+.4}), recorded ({:+.4}, {:+.4}); \
             this is far outside the noise of a different random stream, so \
             the flow itself has changed",
            c.name, got.ux, got.uy, g.6, g.7
        );

        let rel = (got.mass - g.3) as f64 / g.3 as f64;
        assert!(
            rel.abs() < MASS_TOL,
            "{}: {} particles against a recorded {} ({:+.1}%); mass is not \
             being carried correctly",
            c.name, got.mass, g.3, rel * 100.0
        );
    }
}

/// The same configuration, run twice, must give exactly the same answer.
///
/// This is the test that catches a data race: `step` writes its output rows
/// from several threads at once, and an optimisation that widens what a thread
/// touches -- or reaches for `unsafe` to skip a bounds check -- can make the
/// result depend on how the threads happen to interleave. That shows up here
/// long before it shows up as a wrong picture.
#[test]
fn stepping_is_deterministic() {
    for c in CASES {
        let a = run(c, SEED);
        let b = run(c, SEED);
        assert!(
            a == b,
            "{}: two identical runs disagree (cells 0x{:016x} vs 0x{:016x}), \
             so the update is not deterministic -- suspect a race between the \
             row threads",
            c.name, a.cells, b.cells
        );
    }
}

/// Mass and momentum are exactly conserved in a closed periodic box, so they
/// cannot depend on how the rows were split between threads -- even though the
/// microstate does, because each thread draws from its own generator.
///
/// This is the sharpest invariant in the file: it is exact, it holds for any
/// thread count, and no reordering of the random draws can excuse breaking it.
#[test]
fn thread_count_does_not_change_conserved_quantities() {
    for &rest in &[false, true] {
        let mut expected: Option<(i64, i64, i64)> = None;
        for threads in [1usize, 2, 3, 5, 8] {
            let c = Case {
                name: "closed box",
                w: 64,
                h: 64,
                rest,
                inlet: 0,
                plate: false,
                threads,
                steps: 100,
            };
            let mut lat = build(&c, SEED);
            let before = invariants(&lat);
            lat.advance(c.steps);
            let after = invariants(&lat);

            assert_eq!(
                before, after,
                "rest = {rest}, {threads} threads: the closed box did not \
                 conserve mass and momentum"
            );
            match expected {
                None => expected = Some(after),
                Some(e) => assert_eq!(
                    e, after,
                    "rest = {rest}: {threads} threads gave different conserved \
                     quantities from one thread"
                ),
            }
        }
    }
}

/// Not a test: prints a `GOLDEN` table for pasting into this file.
#[test]
#[ignore = "regenerates the golden table; run explicitly"]
fn print_golden() {
    println!("\nconst GOLDEN: &[Golden] = &[");
    for c in CASES {
        let o = run(c, SEED);
        println!(
            "    (\"{}\", 0x{:016x}, 0x{:016x}, {}, {}, {}, {:.6}, {:.6}),",
            c.name, o.cells, o.field, o.mass, o.px, o.py, o.ux, o.uy
        );
    }
    println!("];\n");
}
