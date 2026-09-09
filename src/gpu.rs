//! The lattice on the GPU, stored as bitplanes.
//!
//! The byte-per-cell update in `lattice.rs` spends its time on a table lookup
//! and a random draw *per cell*. Packed as bitplanes --- one bit per cell, 32
//! cells to a `uint`, a separate plane per direction --- propagation becomes a
//! shift of whole words and collision becomes Boolean algebra evaluated on 32
//! cells at once. One random word then feeds 32 cells instead of one.
//!
//! The collision rule is not hand-written. It is emitted, at startup, from the
//! same `collision::classes` enumeration the CPU lookup table is built from:
//! for each momentum class, a term that recognises the class and a selector
//! that picks uniformly among its members. That way the two implementations
//! cannot drift into describing different physics, and
//! `the_shader_rule_matches_the_enumeration` checks the emitted circuit against
//! the enumeration for every state and every draw.

use crate::collision;
use crate::hex::{NDIR, REST_BIT, SOLID_BIT, SQRT3_2};
use crate::lattice::Equilibrium;
use crate::metal::{Buffer, Device, Pipeline};
use crate::render::Field;

/// Cells per word. `uint` is what an Apple GPU wants to move and operate on.
const BITS: usize = 32;
/// Direction planes, plus the rest plane. The solid plane lives on its own,
/// outside the ping-pong, because it never changes.
const PLANES: usize = NDIR + 1;

/// Plane `d` is bit `d` of the cell byte, all the way up: the six directions,
/// then the rest bit, then the solid bit. `store` unpacks in that order
/// without naming the constants, so say here that they line up.
const _: () = assert!(REST_BIT == 1 << NDIR && SOLID_BIT == 1 << PLANES);

// ---------------------------------------------------------------------------
// Emitting the collision rule
// ---------------------------------------------------------------------------

/// The Boolean circuit for one collision, as Metal source.
///
/// Classes of size 2, 3 and 5 need a fair bit, a uniform choice of three and a
/// uniform choice of five. Those arrive as one-hot selectors; when a selector
/// is all zero the cell is simply left alone, which is what the rejection
/// sampling in the shader does when it runs out of rounds. Leaving a state put
/// with some fixed probability keeps the transition matrix doubly stochastic,
/// so semi-detailed balance --- and with it the equilibrium --- is untouched.
pub fn collide_source(rest_particles: bool) -> String {
    let nbits = if rest_particles { 7 } else { 6 };
    let classes: Vec<Vec<u8>> = collision::classes(rest_particles)
        .into_iter()
        .filter(|g| g.len() > 1)
        .collect();
    for g in &classes {
        assert!(
            matches!(g.len(), 2 | 3 | 5),
            "no selector for a class of {} states; the shader knows about 2, 3 and 5",
            g.len()
        );
    }

    let mut needed: Vec<u8> = classes.iter().flatten().copied().collect();
    needed.sort_unstable();

    let mut s = String::new();
    s.push_str(
        "inline void collide(thread uint n[7], uint s2, thread uint s3[3],\n\
         \x20                   thread uint s5[5], thread uint o[7]) {\n",
    );
    for d in 0..7 {
        s.push_str(&format!("  uint n{d} = n[{d}];\n"));
    }
    for d in 0..nbits {
        s.push_str(&format!("  uint c{d} = ~n{d};\n"));
    }
    s.push_str("  uint t2[2]; t2[0] = s2; t2[1] = ~s2;\n");

    // A product tree over the state bits, so states that share a prefix share
    // the work. Branches that no class needs are never built.
    let mut frontier: Vec<(u8, String)> = vec![(0, String::new())];
    for level in 0..nbits {
        let mask: u8 = ((1u16 << (level + 1)) - 1) as u8;
        let mut next = Vec::new();
        for (val, expr) in &frontier {
            for bit in 0..2u8 {
                let nv = val | (bit << level);
                if !needed.iter().any(|&st| st & mask == nv) {
                    continue;
                }
                let lit = if bit == 1 { format!("n{level}") } else { format!("c{level}") };
                if level == 0 {
                    next.push((nv, lit));
                } else {
                    let name = format!("t{}_{}", level + 1, nv);
                    s.push_str(&format!("  uint {name} = {expr} & {lit};\n"));
                    next.push((nv, name));
                }
            }
        }
        frontier = next;
    }
    let term = |st: u8| -> &str {
        &frontier.iter().find(|(v, _)| *v == st).expect("state in the tree").1
    };

    let mut out: Vec<Vec<String>> = vec![Vec::new(); 7];
    let mut acted: Vec<String> = Vec::new();
    for (ci, g) in classes.iter().enumerate() {
        let sel = match g.len() {
            2 => "t2",
            3 => "s3",
            _ => "s5",
        };
        let members: Vec<String> = g.iter().map(|&st| term(st).to_string()).collect();
        s.push_str(&format!("  uint i{ci} = {};\n", members.join(" | ")));
        let any: Vec<String> = (0..g.len()).map(|k| format!("{sel}[{k}]")).collect();
        s.push_str(&format!("  uint r{ci} = i{ci} & ({});\n", any.join(" | ")));
        acted.push(format!("r{ci}"));
        for (k, &member) in g.iter().enumerate() {
            s.push_str(&format!("  uint p{ci}_{k} = i{ci} & {sel}[{k}];\n"));
            for d in 0..7 {
                if member & (1 << d) != 0 {
                    out[d].push(format!("p{ci}_{k}"));
                }
            }
        }
    }

    s.push_str(&format!("  uint keep = ~({});\n", acted.join(" | ")));
    for d in 0..7 {
        if d >= nbits {
            s.push_str(&format!("  o[{d}] = n{d};\n"));
        } else if out[d].is_empty() {
            s.push_str(&format!("  o[{d}] = n{d} & keep;\n"));
        } else {
            s.push_str(&format!("  o[{d}] = (n{d} & keep) | {};\n", out[d].join(" | ")));
        }
    }
    s.push_str("}\n");
    s
}

/// Everything but the collision: propagation, walls, the inflow boundary, the
/// draws, coarse-graining, a particle count, and a kernel that exists only so
/// the tests can check the emitted rule.
const KERNELS: &str = r#"
// A hash with good avalanche, so that neighbouring cells and consecutive steps
// get uncorrelated draws from a cheap stateless source.
inline uint mix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}

// One-hot selectors. Rejection sampling: each round resolves the lanes it can
// and leaves the rest for the next one. A lane still unresolved at the end
// keeps its state, which is a fixed probability and so still doubly
// stochastic.
inline void draws(uint seed, thread uint& s2, thread uint s3[3], thread uint s5[5]) {
    uint r = mix(seed);
    s2 = r;
    s3[0] = 0u; s3[1] = 0u; s3[2] = 0u;
    uint open = 0xffffffffu;
    for (int k = 0; k < 3; ++k) {
        r = mix(r); uint a = r; r = mix(r); uint b = r;
        s3[0] |= open & ~a & ~b;
        s3[1] |= open &  a & ~b;
        s3[2] |= open & ~a &  b;
        open  &= a & b;
    }
    s5[0] = 0u; s5[1] = 0u; s5[2] = 0u; s5[3] = 0u; s5[4] = 0u;
    uint open5 = 0xffffffffu;
    for (int k = 0; k < 4; ++k) {
        r = mix(r); uint a = r; r = mix(r); uint b = r; r = mix(r); uint c = r;
        s5[0] |= open5 & ~a & ~b & ~c;
        s5[1] |= open5 &  a & ~b & ~c;
        s5[2] |= open5 & ~a &  b & ~c;
        s5[3] |= open5 &  a &  b & ~c;
        s5[4] |= open5 & ~a & ~b &  c;
        open5 &= c & (a | b);
    }
}

// A word of independent Bernoulli bits, each true with probability t / 2^24.
//
// Reading the threshold from its least significant bit upwards, `x` after
// step j is true with probability equal to the bits consumed so far: an OR
// with a fresh uniform word adds a half, an AND with one halves what is
// there. Twenty-four rounds therefore reproduce the CPU's `k < t` on a 24-bit
// draw exactly, one probability at a time rather than one cell at a time.
inline uint bernoulli(uint t, thread uint& r) {
    if (t == 0u) return 0u;
    if (t >= (1u << 24)) return 0xffffffffu;
    uint x = 0u;
    for (uint j = 0; j < 24u; ++j) {
        r = mix(r);
        x = ((t >> j) & 1u) ? (r | x) : (r & x);
    }
    return x;
}

// P: 0 wpr, 1 h, 2 seed, 3 lastword, 4 lastbit, 5 tailmask, 6 total,
//    7 inlet words per row, 8 inlet columns, 9 inlet threads (h * 7),
//    10 words per row outside the inlet, 11..17 the seven thresholds
kernel void lgca_step(device const uint* a       [[buffer(0)]],
                      device uint* b             [[buffer(1)]],
                      device const uint* solid   [[buffer(2)]],
                      device const uint* anysolid[[buffer(3)]],
                      constant uint* P           [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    uint wpr = P[0], hh = P[1], total = P[6];
    if (gid >= total) return;

    // Thread index to lattice word, with the words the inflow boundary touches
    // brought to the front. The inlet is a couple of words at the start of each
    // row, so under the obvious row-major mapping they are `wpr` apart and half
    // of all 32-wide SIMD groups contain one -- which means half the machine
    // runs the re-seed while thirty-one lanes in thirty-two wait for it.
    // Gathered at the front they occupy 1.5% of the groups instead of 50%.
    // Threads still walk a row in order, so nothing about coalescing changes;
    // with no inlet `iw` is zero and this is exactly `gid / wpr`.
    // Written as one division with selected operands rather than two branches
    // with one each: integer division by a runtime value is not cheap, and
    // this kernel has no spare cycles to hide a second one behind.
    uint iw = P[7], nthr = P[9];
    bool in_inlet = gid < nthr;
    uint per = in_inlet ? iw : P[10];        // words per row in this range
    uint g = in_inlet ? gid : gid - nthr;
    uint y = g / per;
    uint j = (in_inlet ? 0u : iw) + (g - y * per);
    uint lastword = P[3], lastbit = P[4];

    const int DXE[6] = {1, 0, -1, -1, -1, 0};
    const int DXO[6] = {1, 1,  0, -1,  0, 1};
    const int DY[6]  = {0, 1,  1,  0, -1, -1};
    const int OPP[6] = {3, 4, 5, 0, 1, 2};

    uint n[7];
    for (uint d = 0; d < 6; ++d) {
        int bk = OPP[d];
        int dx = (y & 1u) ? DXO[bk] : DXE[bk];
        int sy = int(y) + DY[bk];
        if (sy < 0) sy += int(hh);
        if (sy >= int(hh)) sy -= int(hh);
        uint base = d * total + uint(sy) * wpr;
        uint w = a[base + j];
        uint v;
        if (dx == 0) {
            v = w;
        } else if (dx > 0) {
            // bit x takes bit x+1
            uint hi = (j + 1u < wpr) ? a[base + j + 1u] : 0u;
            v = (w >> 1) | (hi << 31);
            if (j == lastword) {                       // the one bit that wraps
                v = (v & ~(1u << lastbit)) | ((a[base] & 1u) << lastbit);
            }
        } else {
            // bit x takes bit x-1
            uint lo = (j > 0u) ? a[base + j - 1u] : 0u;
            v = (w << 1) | (lo >> 31);
            if (j == 0u) {
                v = (v & ~1u) | ((a[base + lastword] >> lastbit) & 1u);
            }
        }
        n[d] = v;
    }
    n[6] = a[6u * total + y * wpr + j];               // rest particles do not move

    // Keyed on the lattice word, not on the thread index. Those were the same
    // thing until the inlet remapping above, and keeping them apart is what
    // lets a kernel that visits the words in a different order --- one fusing
    // two steps, say --- be checked against this one bit for bit.
    uint word = y * wpr + j;
    uint s2; uint s3[3]; uint s5[5];
    draws(word ^ P[2], s2, s3, s5);
    uint o[7];
    collide(n, s2, s3, s5, o);

    // A wall sends every particle back the way it came, and never collides.
    //
    // The obstacle is a few hundred words out of a million, but the solid
    // plane is one of the fifteen words a step moves per lattice word --- so
    // reading it unconditionally spends 7% of a memory-bound step on a value
    // that is almost always zero. `anysolid` holds one bit per word, so
    // thirty-two threads share one load of it, and it stays in cache.
    uint sm = 0u;
    if ((anysolid[word >> 5] >> (word & 31u)) & 1u) {
        sm = solid[word];
    }
    uint fluid = ~sm;
    uint res[7];
    for (uint d = 0; d < 6; ++d) res[d] = (sm & n[(d + 3u) % 6u]) | (fluid & o[d]);
    res[6] = (sm & n[6]) | (fluid & o[6]);

    // The inflow boundary: the leftmost columns are redrawn from equilibrium
    // every step, which is what maintains the mean flow. A wall is not
    // re-seeded, so `fluid` masks it out.
    if (j < iw) {
        uint span = min(P[8] - j * 32u, 32u);
        uint imask = (span >= 32u) ? 0xffffffffu : ((1u << span) - 1u);
        uint apply = imask & fluid;
        if (apply != 0u) {
            uint r = mix(word ^ P[2] ^ 0x9e3779b9u);
            for (uint d = 0; d < 7u; ++d) {
                res[d] = (res[d] & ~apply) | (bernoulli(P[11u + d], r) & apply);
            }
        }
    }

    uint mask = (j == lastword) ? P[5] : 0xffffffffu;
    for (uint d = 0; d < 7; ++d) b[d * total + y * wpr + j] = res[d] & mask;
}

// Coarse-graining, one thread per block of the display grid.
//
// The whole point of running the update on the GPU is that the program stops
// synchronising every step, and a frame is 500 steps but a field sample is
// 5. So the running time average lives in device memory and is blended here;
// nothing crosses back until a frame is actually written.
//
// Obstacle sites are left out of the mass and momentum -- a block that is half
// wall reports the velocity of the half that is fluid -- but counted in
// `sol`, matching `Field::sample`.
//
// Q: 0 wpr, 1 total, 2 bx, 3 by, 4 bw, 5 bh, 6 alpha, 7 sqrt(3)/2
kernel void lgca_sample(device const uint* a     [[buffer(0)]],
                        device const uint* solid [[buffer(1)]],
                        device float* ux         [[buffer(2)]],
                        device float* uy         [[buffer(3)]],
                        device float* rho        [[buffer(4)]],
                        device float* sol        [[buffer(5)]],
                        constant uint* Q         [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    uint bw = Q[4], bh = Q[5];
    if (gid >= bw * bh) return;
    uint jb = gid / bw, ib = gid - jb * bw;
    uint wpr = Q[0], total = Q[1], bx = Q[2], by = Q[3];

    const int CX2[6] = {2, 1, -1, -2, -1, 1};
    const int CY2[6] = {0, 1, 1, 0, -1, -1};

    // A block is bx columns wide wherever it happens to fall, so it straddles
    // word boundaries; each row contributes one or two masked words.
    uint x0 = ib * bx, x1 = x0 + bx;
    uint w0 = x0 >> 5, w1 = (x1 - 1u) >> 5;
    int mass = 0, px = 0, py = 0;
    uint nsolid = 0u;
    for (uint y = jb * by; y < (jb + 1u) * by; ++y) {
        uint base = y * wpr;
        for (uint wi = w0; wi <= w1; ++wi) {
            uint lo = max(x0, wi << 5) - (wi << 5);
            uint hi = min(x1, (wi + 1u) << 5) - (wi << 5);
            uint m = (hi - lo >= 32u) ? 0xffffffffu : (((1u << (hi - lo)) - 1u) << lo);
            uint sm = solid[base + wi] & m;
            nsolid += popcount(sm);
            uint fm = m & ~sm;
            for (uint d = 0; d < 6u; ++d) {
                int c = int(popcount(a[d * total + base + wi] & fm));
                mass += c;
                px += CX2[d] * c;
                py += CY2[d] * c;
            }
            mass += int(popcount(a[6u * total + base + wi] & fm));
        }
    }

    float n = float(bx * by);
    float m = float(mass);
    float vx = 0.0f, vy = 0.0f;
    if (mass > 0) {
        vx = float(px) * 0.5f / m;
        vy = float(py) * as_type<float>(Q[7]) / m;
    }
    float alpha = as_type<float>(Q[6]);
    ux[gid] += alpha * (vx - ux[gid]);
    uy[gid] += alpha * (vy - uy[gid]);
    rho[gid] += alpha * (m / n - rho[gid]);
    sol[gid] = float(nsolid) / n;
}

// Particles on the lattice, obstacle sites included, one thread per row.
// R: 0 wpr, 1 h, 2 total, 3 lastword, 4 tailmask
kernel void lgca_count(device const uint* a       [[buffer(0)]],
                       device atomic_uint* out    [[buffer(1)]],
                       constant uint* R           [[buffer(2)]],
                       uint y [[thread_position_in_grid]]) {
    if (y >= R[1]) return;
    uint wpr = R[0], total = R[2], lastword = R[3], tail = R[4];
    uint n = 0u;
    for (uint d = 0; d < 7u; ++d) {
        uint base = d * total + y * wpr;
        for (uint j = 0; j < wpr; ++j) {
            n += popcount(a[base + j] & ((j == lastword) ? tail : 0xffffffffu));
        }
    }
    atomic_fetch_add_explicit(&out[0], n, memory_order_relaxed);
}

// Exists so a test can check the emitted circuit itself, rather than a
// transliteration of it. Each thread is one (state, draw) combination.
kernel void lgca_verify(device uint* out [[buffer(0)]],
                        constant uint* P [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= P[0]) return;
    uint five = i % 6u, three = (i / 6u) % 4u, two = (i / 24u) % 2u, state = i / 48u;
    uint n[7];
    for (uint d = 0; d < 7; ++d) n[d] = ((state >> d) & 1u) ? 0xffffffffu : 0u;
    uint s3[3] = {0u, 0u, 0u}, s5[5] = {0u, 0u, 0u, 0u, 0u};
    if (three < 3u) s3[three] = 0xffffffffu;
    if (five < 5u) s5[five] = 0xffffffffu;
    uint o[7];
    collide(n, two ? 0xffffffffu : 0u, s3, s5, o);
    uint r = 0u, bad = 0u;
    for (uint d = 0; d < 7; ++d) {
        if (o[d] == 0xffffffffu) r |= 1u << d;
        else if (o[d] != 0u) bad = 1u;               // lanes must never disagree
    }
    out[i] = r | (bad << 16);
}
"#;

// ---------------------------------------------------------------------------
// The lattice
// ---------------------------------------------------------------------------

/// The coarse-grained field, kept on the device between frames so that
/// sampling every fifth step does not mean synchronising every fifth step.
struct GpuField {
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    ux: Buffer,
    uy: Buffer,
    rho: Buffer,
    sol: Buffer,
}

/// Arguments to `lgca_sample`. Floats go through as bit patterns, which is
/// exact --- `SQRT3_2` in particular has to be the same number on both sides
/// or the two paths would disagree about `uy` in the last decimal place.
fn sample_params(g: &GpuField, wpr: usize, total: usize, alpha: f32) -> [u32; 8] {
    [
        wpr as u32,
        total as u32,
        g.bx as u32,
        g.by as u32,
        g.bw as u32,
        g.bh as u32,
        alpha.to_bits(),
        SQRT3_2.to_bits(),
    ]
}

pub struct GpuLattice {
    dev: Device,
    step: Pipeline,
    sample: Pipeline,
    count: Pipeline,
    a: Buffer,
    b: Buffer,
    solid: Buffer,
    /// One bit per lattice word: is there any obstacle in it? Lets the step
    /// skip the solid load for the overwhelming majority of words.
    anysolid: Buffer,
    /// One `uint` for `lgca_count` to accumulate into.
    counter: Buffer,
    field: Option<GpuField>,
    pub w: usize,
    pub h: usize,
    wpr: usize,
    total: usize,
    pub steps: u64,
    seed: u64,
    inlet_cols: usize,
    /// The inflow equilibrium as 24-bit thresholds, rest particle last.
    thresh: [u32; PLANES],
}

impl GpuLattice {
    pub fn new(w: usize, h: usize, rest_particles: bool, seed: u64) -> Result<Self, String> {
        assert!(h % 2 == 0, "row count must be even for the lattice to wrap cleanly in y");
        assert!(w > 4 && h > 4, "lattice is too small");
        let dev = Device::new().ok_or("no Metal device on this machine")?;
        let source = format!("#include <metal_stdlib>\nusing namespace metal;\n{}{}",
                             collide_source(rest_particles), KERNELS);
        let step = dev.pipeline(&source, "lgca_step")?;
        let sample = dev.pipeline(&source, "lgca_sample")?;
        let count = dev.pipeline(&source, "lgca_count")?;
        let wpr = w.div_ceil(BITS);
        let total = wpr * h;
        let a = dev.buffer(total * PLANES * 4);
        let b = dev.buffer(total * PLANES * 4);
        let solid = dev.buffer(total * 4);
        let anysolid = dev.buffer(total.div_ceil(32) * 4);
        let counter = dev.buffer(4);
        Ok(GpuLattice {
            dev,
            step,
            sample,
            count,
            a,
            b,
            solid,
            anysolid,
            counter,
            field: None,
            w,
            h,
            wpr,
            total,
            steps: 0,
            seed,
            inlet_cols: 0,
            thresh: [0; PLANES],
        })
    }

    pub fn device_name(&self) -> String {
        self.dev.name()
    }

    /// Re-seed the leftmost `cols` columns from the inflow equilibrium on
    /// every step. The thresholds come from `Equilibrium` itself, so the two
    /// paths draw the inflow from the same distribution rather than from two
    /// descriptions of it.
    pub fn set_inlet(&mut self, cols: usize, density: f32, inflow: (f32, f32), rest: bool) {
        self.inlet_cols = cols.min(self.w);
        self.thresh = Equilibrium::new(density, inflow.0, inflow.1, rest).thresholds();
    }

    /// Give the lattice somewhere on the device to accumulate a coarse-grained
    /// field, laid out to match a `Field` of the same block size.
    pub fn attach_field(&mut self, bx: usize, by: usize) {
        let (bw, bh) = (self.w / bx, self.h / by);
        let n = bw * bh;
        self.field = Some(GpuField {
            bx,
            by,
            bw,
            bh,
            ux: self.dev.buffer(n * 4),
            uy: self.dev.buffer(n * 4),
            rho: self.dev.buffer(n * 4),
            sol: self.dev.buffer(n * 4),
        });
    }

    fn params(&self) -> [u32; 11 + PLANES] {
        let lastword = ((self.w - 1) / BITS) as u32;
        let lastbit = ((self.w - 1) % BITS) as u32;
        let tail = if lastbit == 31 { u32::MAX } else { (1u32 << (lastbit + 1)) - 1 };
        let iw = self.inlet_cols.div_ceil(BITS);
        let mut p = [0u32; 11 + PLANES];
        p[0] = self.wpr as u32;
        p[1] = self.h as u32;
        p[2] = (self.seed ^ self.steps.wrapping_mul(0x9E37_79B9_7F4A_7C15)) as u32;
        p[3] = lastword;
        p[4] = lastbit;
        p[5] = tail;
        p[6] = self.total as u32;
        p[7] = iw as u32;
        p[8] = self.inlet_cols as u32;
        p[9] = (iw * self.h) as u32;
        // Never zero: an inlet spanning the whole width leaves this range
        // empty, but the divisor is still evaluated.
        p[10] = (self.wpr - iw).max(1) as u32;
        p[11..].copy_from_slice(&self.thresh);
        p
    }

    /// Spread a byte-per-cell lattice across the planes. `SOLID_BIT` goes to
    /// its own buffer, since it does not take part in the ping-pong.
    pub fn load(&mut self, cells: &[u8]) {
        assert_eq!(cells.len(), self.w * self.h, "wrong number of cells");
        let (w, wpr, total) = (self.w, self.wpr, self.total);
        for v in self.a.as_mut_slice::<u32>().iter_mut() {
            *v = 0;
        }
        for v in self.solid.as_mut_slice::<u32>().iter_mut() {
            *v = 0;
        }
        for v in self.anysolid.as_mut_slice::<u32>().iter_mut() {
            *v = 0;
        }
        let planes = self.a.as_mut_slice::<u32>();
        let solid = self.solid.as_mut_slice::<u32>();
        for y in 0..self.h {
            for x in 0..w {
                let c = cells[y * w + x];
                let word = y * wpr + x / BITS;
                let bit = 1u32 << (x % BITS);
                for d in 0..NDIR {
                    if c & (1 << d) != 0 {
                        planes[d * total + word] |= bit;
                    }
                }
                if c & REST_BIT != 0 {
                    planes[NDIR * total + word] |= bit;
                }
                if c & SOLID_BIT != 0 {
                    solid[word] |= bit;
                }
            }
        }
        let (solid, any) = (self.solid.as_slice::<u32>(), self.anysolid.as_mut_slice::<u32>());
        for (i, &s) in solid.iter().enumerate() {
            if s != 0 {
                any[i >> 5] |= 1u32 << (i & 31);
            }
        }
    }

    /// Put the lattice back to a given state and random seed, keeping the
    /// compiled shader. On this path that is the point: compiling costs more
    /// than a short run does, so a measurement that wants twenty realisations
    /// should pay for it once.
    pub fn restart(&mut self, cells: &[u8], seed: u64) {
        self.load(cells);
        self.seed = seed;
        self.steps = 0;
    }

    /// Gather the planes back into one byte per cell.
    ///
    /// A word at a time rather than a cell at a time: the eight planes that
    /// describe a cell are eight words that also describe the thirty-one cells
    /// beside it, so they are worth loading once and taking apart in
    /// registers. Done per cell this was the most expensive thing in
    /// `transport::measure`, which projects the lattice onto a Fourier mode
    /// every few steps and comes through here to do it.
    pub fn store(&self, cells: &mut [u8]) {
        assert_eq!(cells.len(), self.w * self.h, "wrong number of cells");
        let (w, wpr, total) = (self.w, self.wpr, self.total);
        let planes = self.a.as_slice::<u32>();
        let solid = self.solid.as_slice::<u32>();
        for (y, row) in cells.chunks_mut(w).enumerate() {
            for (jw, out) in row.chunks_mut(BITS).enumerate() {
                let word = y * wpr + jw;
                let mut p = [0u32; PLANES + 1];
                for (d, v) in p.iter_mut().enumerate().take(PLANES) {
                    *v = planes[d * total + word];
                }
                p[PLANES] = solid[word];
                for (k, c) in out.iter_mut().enumerate() {
                    let mut b = 0u8;
                    for (d, &v) in p.iter().enumerate() {
                        b |= (((v >> k) & 1) as u8) << d;
                    }
                    *c = b;
                }
            }
        }
    }

    /// Run `n` steps. They go into as few command buffers as the budget below
    /// allows, so the GPU is asked once rather than `n` times --- a round trip
    /// costs more than seven steps do.
    pub fn advance(&mut self, n: u64) {
        self.advance_sampling(n, 0, 0.0);
    }

    /// Advance `n` steps, folding the lattice into the attached field every
    /// `sample_every` of them --- and at the end of the run, so a ragged last
    /// chunk is sampled too, as the CPU loop does. `sample_every` of 0 skips
    /// sampling altogether.
    ///
    /// Nothing is read back here. The whole point of batching is that the
    /// caller synchronises when it wants a frame, not when it wants a sample.
    pub fn advance_sampling(&mut self, n: u64, sample_every: u64, alpha: f32) {
        // Enough to keep submission out of the measurement, short enough that
        // a command buffer never runs long enough to look like a hang.
        const MAX_DISPATCH: usize = 512;
        let (total, wpr) = (self.total, self.wpr);
        let mut batch = self.dev.batch();
        let (mut left, mut since) = (n, 0u64);
        while left > 0 {
            let p = self.params();
            batch.dispatch(
                &self.step,
                &[&self.a, &self.b, &self.solid, &self.anysolid],
                &p,
                total as u64,
            );
            std::mem::swap(&mut self.a, &mut self.b);
            self.steps += 1;
            left -= 1;
            since += 1;

            if sample_every > 0 && (since >= sample_every || left == 0) {
                since = 0;
                if let Some(g) = self.field.as_ref() {
                    let q = sample_params(g, wpr, total, alpha);
                    batch.dispatch(
                        &self.sample,
                        &[&self.a, &self.solid, &g.ux, &g.uy, &g.rho, &g.sol],
                        &q,
                        (g.bw * g.bh) as u64,
                    );
                }
            }
            if batch.dispatches() >= MAX_DISPATCH {
                batch.wait();
                batch = self.dev.batch();
            }
        }
        batch.wait();
    }

    /// Fold the current state into the attached field once, and wait. For
    /// the initial sample, and for tests; a run uses `advance_sampling`.
    pub fn sample_field(&mut self, alpha: f32) {
        let Some(g) = self.field.as_ref() else { return };
        let q = sample_params(g, self.wpr, self.total, alpha);
        let mut batch = self.dev.batch();
        batch.dispatch(
            &self.sample,
            &[&self.a, &self.solid, &g.ux, &g.uy, &g.rho, &g.sol],
            &q,
            (g.bw * g.bh) as u64,
        );
        batch.wait();
    }

    /// Copy the device-side field into a `Field`. Shared storage, so this is a
    /// memcpy of four small arrays rather than a transfer.
    pub fn read_field(&self, f: &mut Field) {
        let g = self.field.as_ref().expect("no field attached to this lattice");
        assert_eq!(
            (g.bw, g.bh, g.bx, g.by),
            (f.bw, f.bh, f.bx, f.by),
            "the field on the device has a different block grid from the one being filled"
        );
        f.ux.copy_from_slice(g.ux.as_slice::<f32>());
        f.uy.copy_from_slice(g.uy.as_slice::<f32>());
        f.rho.copy_from_slice(g.rho.as_slice::<f32>());
        f.solid.copy_from_slice(g.sol.as_slice::<f32>());
    }

    /// Particles on the lattice, obstacle sites included: the same quantity
    /// `Lattice::total_particles` counts, without unpacking the planes.
    pub fn total_particles(&mut self) -> u64 {
        let lastbit = (self.w - 1) % BITS;
        let r = [
            self.wpr as u32,
            self.h as u32,
            self.total as u32,
            ((self.w - 1) / BITS) as u32,
            if lastbit == 31 { u32::MAX } else { (1u32 << (lastbit + 1)) - 1 },
        ];
        self.counter.as_mut_slice::<u32>()[0] = 0;
        let mut batch = self.dev.batch();
        batch.dispatch(&self.count, &[&self.a, &self.counter], &r, self.h as u64);
        batch.wait();
        u64::from(self.counter.as_slice::<u32>()[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::classes;
    use crate::hex::{CX2, CY2, DX_EVEN, DX_ODD, DY, MOVING_MASK, STATE_MASK};
    use crate::lattice::Lattice;
    use crate::rng::Rng;

    fn device_or_skip() -> bool {
        if Device::new().is_none() {
            eprintln!("no Metal device; skipping");
            return false;
        }
        true
    }

    /// The circuit the GPU actually runs, checked against the enumeration it
    /// was emitted from, for every state and every draw.
    #[test]
    fn the_shader_rule_matches_the_enumeration() {
        if !device_or_skip() {
            return;
        }
        for &rest in &[true, false] {
            let dev = Device::new().unwrap();
            let src = format!("#include <metal_stdlib>\nusing namespace metal;\n{}{}",
                              collide_source(rest), KERNELS);
            let pso = dev.pipeline(&src, "lgca_verify").expect("verify pipeline");
            let n_states: usize = if rest { 128 } else { 64 };
            let n = n_states * 48;
            let out = dev.buffer(n * 4);
            let mut batch = dev.batch();
            batch.dispatch(&pso, &[&out], &[n as u32], n as u64);
            batch.wait();

            let mut class_of = vec![Vec::new(); 128];
            for g in classes(rest) {
                for &s in &g {
                    class_of[s as usize] = g.clone();
                }
            }
            let got = out.as_slice::<u32>();
            for i in 0..n {
                let (five, three, two, state) =
                    (i % 6, (i / 6) % 4, (i / 24) % 2, (i / 48) as u8);
                assert_eq!(got[i] >> 16, 0, "lanes disagreed on state {state:07b}");
                let g = &class_of[state as usize];
                let want = if g.len() == 1 {
                    state
                } else {
                    match g.len() {
                        2 => g[if two == 1 { 0 } else { 1 }],
                        3 => if three < 3 { g[three] } else { state },
                        5 => if five < 5 { g[five] } else { state },
                        _ => unreachable!(),
                    }
                };
                assert_eq!(
                    got[i] as u8, want,
                    "rest={rest} state {state:07b} two={two} three={three} five={five}"
                );
            }
        }
    }

    /// A lone particle has no collision partner, so it must fly straight. This
    /// is the propagation test from `lattice.rs`, run on the GPU, and it pins
    /// down the shifts and the wrap for both row parities.
    #[test]
    fn a_single_particle_travels_in_a_straight_line() {
        if !device_or_skip() {
            return;
        }
        // A width that is not a multiple of the word size, to exercise the tail.
        for w in [64usize, 80] {
            for d in 0..NDIR {
                for start_row in [10usize, 11] {
                    let (h, steps) = (48usize, 8usize);
                    let mut g = GpuLattice::new(w, h, true, 1).expect("gpu lattice");
                    let mut cells = vec![0u8; w * h];
                    let (x0, y0) = (24usize, start_row);
                    cells[y0 * w + x0] = 1 << d;
                    g.load(&cells);
                    g.advance(steps as u64);
                    g.store(&mut cells);

                    let live: Vec<usize> =
                        (0..cells.len()).filter(|&i| cells[i] & STATE_MASK != 0).collect();
                    assert_eq!(live.len(), 1, "w={w} direction {d}: particle was lost");
                    assert_eq!(cells[live[0]], 1 << d, "w={w} direction {d}: it turned");

                    // Walk the expected path with the same offsets the CPU uses.
                    let (mut x, mut y) = (x0 as i32, y0 as i32);
                    for _ in 0..steps {
                        let dx = if y & 1 == 1 { DX_ODD[d] } else { DX_EVEN[d] };
                        x = (x + dx).rem_euclid(w as i32);
                        y = (y + DY[d]).rem_euclid(h as i32);
                    }
                    assert_eq!(
                        live[0],
                        (y as usize) * w + x as usize,
                        "w={w} direction {d} from row {start_row}: wrong destination"
                    );
                }
            }
        }
    }

    /// A particle that runs into a wall comes back the way it came.
    #[test]
    fn walls_reverse_particles() {
        if !device_or_skip() {
            return;
        }
        let (w, h) = (64usize, 48usize);
        let mut g = GpuLattice::new(w, h, true, 1).unwrap();
        let mut cells = vec![0u8; w * h];
        let (x0, y0) = (20usize, 24usize);
        cells[y0 * w + x0 + 3] = SOLID_BIT;
        cells[y0 * w + x0] = 1; // heading east
        g.load(&cells);
        g.advance(6);
        g.store(&mut cells);
        let live: Vec<usize> =
            (0..cells.len()).filter(|&i| cells[i] & STATE_MASK != 0).collect();
        assert_eq!(live.len(), 1);
        assert_eq!(cells[live[0]] & STATE_MASK, 1 << 3, "should now head west");
        assert_eq!(live[0], y0 * w + x0, "should be back where it started");
    }

    /// Collisions conserve mass and momentum exactly, so a periodic box must
    /// hold both fixed however many steps it runs.
    #[test]
    fn a_periodic_box_conserves_mass_and_momentum() {
        if !device_or_skip() {
            return;
        }
        for &rest in &[true, false] {
            let (w, h) = (128usize, 64usize);
            let mut g = GpuLattice::new(w, h, rest, 0xC0FFEE).unwrap();
            let mut cells = vec![0u8; w * h];
            let mut rng = Rng::new(7);
            for c in cells.iter_mut() {
                *c = (rng.next_u32() as u8) & if rest { STATE_MASK } else { MOVING_MASK };
            }
            let totals = |cs: &[u8]| {
                let (mut m, mut px, mut py) = (0i64, 0i64, 0i64);
                for &c in cs {
                    for d in 0..NDIR {
                        if c & (1 << d) != 0 {
                            m += 1;
                            px += CX2[d] as i64;
                            py += CY2[d] as i64;
                        }
                    }
                    if c & REST_BIT != 0 {
                        m += 1;
                    }
                }
                (m, px, py)
            };
            let before = totals(&cells);
            g.load(&cells);
            g.advance(250);
            g.store(&mut cells);
            assert_eq!(before, totals(&cells), "rest = {rest}");
        }
    }

    /// Packing and unpacking has to be the identity, including on a width that
    /// leaves a partly used word at the end of every row.
    #[test]
    fn load_and_store_round_trip() {
        if !device_or_skip() {
            return;
        }
        for w in [64usize, 100] {
            let h = 16usize;
            let mut g = GpuLattice::new(w, h, true, 1).unwrap();
            let mut rng = Rng::new(11);
            let want: Vec<u8> = (0..w * h)
                .map(|_| {
                    let b = rng.next_u32() as u8;
                    if b & 0x80 != 0 { SOLID_BIT } else { b & STATE_MASK }
                })
                .collect();
            g.load(&want);
            let mut got = vec![0u8; w * h];
            g.store(&mut got);
            assert_eq!(want, got, "w={w}");
        }
    }

    /// Coarse-graining on the device has to produce the field the CPU pass
    /// produces, block for block --- including the awkward part, which is that
    /// a block is 20 cells wide and words are 32, so every block straddles a
    /// word boundary differently.
    #[test]
    fn the_gpu_coarse_grains_the_same_field() {
        if !device_or_skip() {
            return;
        }
        use crate::hex::SQRT3_2;
        for (w, h, block) in [(256usize, 128usize, 20usize), (200, 96, 16)] {
            let mut cpu = Lattice::new(w, h, 0.22, (0.3, 0.0), 0, true, 4, 0x5EED);
            let yc = h as f32 * SQRT3_2 / 2.0;
            cpu.add_solid(move |x, y| {
                (x - 60.0) * (x - 60.0) + (y - yc) * (y - yc) <= 18.0 * 18.0
            });
            cpu.init_equilibrium();
            cpu.advance(30);

            let mut want = Field::new(&cpu, block, block);
            want.sample(&cpu, 1.0);

            let mut g = GpuLattice::new(w, h, true, 1).unwrap();
            g.attach_field(block, block);
            g.load(&cpu.cells);
            g.sample_field(1.0);
            let mut got = Field::new(&cpu, block, block);
            g.read_field(&mut got);

            for k in 0..want.ux.len() {
                for (name, a, b) in [
                    ("ux", want.ux[k], got.ux[k]),
                    ("uy", want.uy[k], got.uy[k]),
                    ("rho", want.rho[k], got.rho[k]),
                    ("solid", want.solid[k], got.solid[k]),
                ] {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "{w}x{h} block {block}: {name} at block {k} is {b}, \
                         the CPU makes it {a}"
                    );
                }
            }
        }
    }

    /// The running time average is the reason the field lives on the device at
    /// all, so it has to blend the way `Field::sample` blends.
    #[test]
    fn the_gpu_field_averages_over_time() {
        if !device_or_skip() {
            return;
        }
        const ALPHA: f32 = 0.2;
        let (w, h, block) = (192usize, 96usize, 16usize);
        let mut cpu = Lattice::new(w, h, 0.22, (0.3, 0.0), 4, true, 1, 0x5EED);
        cpu.init_equilibrium();

        let mut want = Field::new(&cpu, block, block);
        let mut g = GpuLattice::new(w, h, true, 1).unwrap();
        g.attach_field(block, block);
        g.load(&cpu.cells);

        // Two independent random streams, so only the averaging itself is
        // being compared, not the microstates: sample the *same* cells on
        // both sides each time, and step the CPU between samples.
        for _ in 0..6 {
            want.sample(&cpu, ALPHA);
            g.sample_field(ALPHA);
            cpu.advance(5);
            g.load(&cpu.cells);
        }
        let mut got = Field::new(&cpu, block, block);
        g.read_field(&mut got);
        for k in 0..want.ux.len() {
            assert!(
                (want.ux[k] - got.ux[k]).abs() < 1e-5
                    && (want.rho[k] - got.rho[k]).abs() < 1e-5,
                "block {k}: ux {} vs {}, rho {} vs {}",
                want.ux[k], got.ux[k], want.rho[k], got.rho[k]
            );
        }
    }

    /// Counting particles without unpacking the planes.
    #[test]
    fn the_gpu_counts_the_same_particles() {
        if !device_or_skip() {
            return;
        }
        // A width that is not a multiple of 32, so the tail mask matters: if
        // the count ignored it, the padding bits would have to be clean by
        // luck rather than by construction.
        for (w, h) in [(256usize, 64usize), (100, 32), (129, 18)] {
            let mut cpu = Lattice::new(w, h, 0.3, (0.2, 0.0), 0, true, 1, 0x5EED);
            cpu.add_solid(|x, y| x > 20.0 && x < 30.0 && y > 5.0 && y < 15.0);
            cpu.init_equilibrium();
            cpu.advance(20);

            let mut g = GpuLattice::new(w, h, true, 1).unwrap();
            g.load(&cpu.cells);
            assert_eq!(g.total_particles(), cpu.total_particles(), "{w}x{h}");
        }
    }

    /// The inflow boundary, which is the one part of the update that creates
    /// particles rather than moving them.
    #[test]
    fn the_inlet_reseeds_at_the_requested_equilibrium() {
        if !device_or_skip() {
            return;
        }
        const DENSITY: f32 = 0.22;
        const SPEED: f32 = 0.4;
        // 40 columns, so the inlet spans two words and the second one is only
        // partly covered.
        for cols in [8usize, 40] {
            let (w, h, steps) = (128usize, 64usize, 60usize);
            let mut g = GpuLattice::new(w, h, true, 0xC0FFEE).unwrap();
            g.set_inlet(cols, DENSITY, (SPEED, 0.0), true);
            let mut cells = vec![0u8; w * h];
            g.load(&cells);

            let mut hits = [0u64; NDIR + 1];
            let mut n = 0u64;
            for _ in 0..steps {
                g.advance(1);
                g.store(&mut cells);
                for y in 0..h {
                    for x in 0..cols {
                        let c = cells[y * w + x];
                        for d in 0..NDIR {
                            hits[d] += u64::from(c >> d & 1);
                        }
                        hits[NDIR] += u64::from(c >> 6 & 1);
                        n += 1;
                    }
                }
            }

            let want = Equilibrium::new(DENSITY, SPEED, 0.0, true).thresholds();
            for d in 0..=NDIR {
                let p = f64::from(want[d]) / f64::from(1u32 << 24);
                let got = hits[d] as f64 / n as f64;
                let sigma = (p * (1.0 - p) / n as f64).sqrt();
                assert!(
                    (got - p).abs() < 5.0 * sigma,
                    "inlet of {cols} columns, plane {d}: occupancy {got:.5} against a \
                     requested {p:.5}, {:.1} sigma out",
                    (got - p).abs() / sigma
                );
            }
        }
    }

    /// A wall in the inflow zone stays a wall. If the re-seed ignored the
    /// solid mask it would manufacture particles inside the obstacle, so an
    /// all-solid lattice must stay empty however long the inlet runs.
    #[test]
    fn the_inlet_does_not_seed_obstacles() {
        if !device_or_skip() {
            return;
        }
        let (w, h) = (96usize, 32usize);
        let mut g = GpuLattice::new(w, h, true, 5).unwrap();
        g.set_inlet(w, 0.4, (0.3, 0.0), true);
        let mut cells = vec![SOLID_BIT; w * h];
        g.load(&cells);
        g.advance(10);
        g.store(&mut cells);
        assert!(
            cells.iter().all(|&c| c == SOLID_BIT),
            "the inlet wrote particles into {} obstacle cells",
            cells.iter().filter(|&&c| c != SOLID_BIT).count()
        );
    }

    /// The re-seed must stop at the column it is told to stop at.
    #[test]
    fn the_inlet_leaves_the_rest_of_the_lattice_alone() {
        if !device_or_skip() {
            return;
        }
        let (w, h, cols) = (128usize, 32usize, 8usize);
        let mut g = GpuLattice::new(w, h, true, 9).unwrap();
        g.set_inlet(cols, 0.4, (0.3, 0.0), true);
        let mut cells = vec![0u8; w * h];
        g.load(&cells);
        // One step: the inlet has been written, and nothing has had time to
        // travel out of it yet.
        g.advance(1);
        g.store(&mut cells);
        for y in 0..h {
            for x in cols..w {
                assert_eq!(cells[y * w + x], 0, "cell ({x}, {y}) was written");
            }
        }
        let seeded = (0..h)
            .flat_map(|y| (0..cols).map(move |x| y * w + x))
            .filter(|&i| cells[i] != 0)
            .count();
        assert!(seeded > h * cols / 2, "the inlet barely seeded anything: {seeded}");
    }

    /// The acceptance test for the whole phase: a second implementation of the
    /// rule is only worth having if it is the same fluid.
    ///
    /// A transverse shear wave in a quiescent periodic box decays as
    /// `exp(-nu k^2 t)`, so fitting the log amplitude measures the viscosity
    /// the code actually has. Both paths are driven through the identical
    /// protocol from the identical initial cells, and averaged over four
    /// realisations first, because the lattice is noisy and the signal adds
    /// while the fluctuations cancel.
    /// Deliberately `#[ignore]`d: it runs the CPU path for twenty thousand
    /// steps and takes several seconds, which does not belong in the ordinary
    /// test loop. Run it after touching the rule or the shader:
    ///
    ///     cargo test --release --lib -- --ignored --nocapture viscosity
    ///
    /// Measured over four independent estimates, this estimator has a spread of
    /// about 4% on the CPU and 8% on the GPU at this size, so the tolerance is
    /// set to catch a real difference in the fluid rather than a noisy draw.
    /// Both paths sit a little above the 0.2989 that `transport::measure`
    /// reports, because that picks its fitting window adaptively and this uses
    /// a fixed one; the point here is the difference between the two, which is
    /// measured the same way on both sides.
    #[test]
    #[ignore = "several seconds; run deliberately after changing the rule"]
    fn the_gpu_fluid_has_the_same_viscosity() {
        if !device_or_skip() {
            return;
        }
        use crate::hex::SQRT3_2;
        use crate::lattice::Equilibrium;
        use crate::moments;

        const SIZE: usize = 256;
        const DENSITY: f32 = 0.22;
        const TRANSIENT: u64 = 200;
        const SAMPLES: usize = 20;
        // The mode decays as exp(-nu k^2 t) with k^2 about 8e-4 here, so the
        // window has to run into the thousands of steps for the amplitude to
        // fall far enough to fit against. A short one measures mostly noise.
        const INTERVAL: u64 = 250;
        const REPS: u64 = 16;
        let (w, h) = (SIZE, SIZE);
        let k = 2.0 * std::f32::consts::PI / (h as f32 * SQRT3_2);
        let amp = 0.05f32;

        // The mode amplitude: row-mean velocity projected onto cos and sin.
        let project = |cells: &[u8]| -> (f64, f64) {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for y in 0..h {
                let m = moments::sum(&moments::WITH_WALLS, &cells[y * w..(y + 1) * w]);
                let u = m.velocity().0;
                let p = k * y as f32 * SQRT3_2;
                a += (u * p.cos()) as f64;
                b += (u * p.sin()) as f64;
            }
            (2.0 * a / h as f64, 2.0 * b / h as f64)
        };
        let seeded = |rep: u64| -> Vec<u8> {
            let s = 0x5EED ^ (rep.wrapping_mul(0x9E37_79B9) << 20);
            let mut rng = Rng::new(s ^ 0xBEEF);
            let mut cells = vec![0u8; w * h];
            for y in 0..h {
                let ux = amp * (k * y as f32 * SQRT3_2).sin();
                let eq = Equilibrium::new(DENSITY, ux, 0.0, true);
                for x in 0..w {
                    cells[y * w + x] = eq.sample(&mut rng);
                }
            }
            cells
        };
        let fit = |trace: &[(f64, f64)]| -> f32 {
            let pts: Vec<(f64, f64)> = trace
                .iter()
                .enumerate()
                .map(|(i, (a, b))| {
                    ((i as u64 * INTERVAL) as f64, (a * a + b * b).sqrt().max(1e-12).ln())
                })
                .collect();
            let n = pts.len() as f64;
            let sx: f64 = pts.iter().map(|p| p.0).sum();
            let sy: f64 = pts.iter().map(|p| p.1).sum();
            let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
            let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
            let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            (-slope / (k * k) as f64) as f32
        };

        let mut cpu_trace = vec![(0.0f64, 0.0f64); SAMPLES];
        let mut gpu_trace = vec![(0.0f64, 0.0f64); SAMPLES];
        for rep in 0..REPS {
            let start = seeded(rep);
            let seed = 0x5EED ^ (rep << 20);

            let mut cpu = Lattice::new(w, h, DENSITY, (0.0, 0.0), 0, true, 4, seed);
            cpu.cells.copy_from_slice(&start);
            cpu.advance(TRANSIENT);
            for slot in cpu_trace.iter_mut() {
                let (a, b) = project(&cpu.cells);
                slot.0 += a / REPS as f64;
                slot.1 += b / REPS as f64;
                cpu.advance(INTERVAL);
            }

            let mut gpu = GpuLattice::new(w, h, true, seed).unwrap();
            gpu.load(&start);
            gpu.advance(TRANSIENT);
            let mut cells = vec![0u8; w * h];
            for slot in gpu_trace.iter_mut() {
                gpu.store(&mut cells);
                let (a, b) = project(&cells);
                slot.0 += a / REPS as f64;
                slot.1 += b / REPS as f64;
                gpu.advance(INTERVAL);
            }
        }

        let (nu_cpu, nu_gpu) = (fit(&cpu_trace), fit(&gpu_trace));
        let rel = (nu_gpu - nu_cpu).abs() / nu_cpu;
        println!(
            "viscosity: cpu nu = {nu_cpu:.4}, gpu nu = {nu_gpu:.4} ({:.1}% apart); \
             transport::measure reports 0.2989 for this density",
            100.0 * rel
        );
        assert!(
            rel < 0.12,
            "the GPU is a different fluid: nu = {nu_gpu:.4} against the CPU's {nu_cpu:.4}"
        );
    }

    /// The other half of the acceptance test.
    ///
    /// Viscosity says the fluid dissipates the same; this says it *advects*
    /// the same, which is the coefficient the Reynolds number is proportional
    /// to. A transverse wave riding on a uniform stream is carried along at
    /// `g * U` rather than at `U` --- a lattice gas is not Galilean invariant
    /// --- so the drift of the mode's phase measures `g` directly.
    ///
    /// Same shape as the viscosity test above: an identical protocol from
    /// identical initial cells, averaged over realisations before the fit.
    ///
    ///     cargo test --release --lib -- --ignored --nocapture advection
    ///
    /// This is by far the sharper of the two acceptance tests, and worth
    /// having for that reason. Measured over four independent estimates the
    /// spread is 0.24% on the CPU and 0.11% on the GPU, against 4% and 8% for
    /// the viscosity fit --- a phase that turns steadily is a much cleaner
    /// thing to measure than an amplitude decaying into the lattice's own
    /// noise. So the tolerance here is 3%, some ten standard deviations of the
    /// difference between two estimates, and it would notice a change in `g`
    /// that the viscosity test would never see.
    #[test]
    #[ignore = "several seconds; run deliberately after changing the rule"]
    fn the_gpu_fluid_has_the_same_advection_factor() {
        if !device_or_skip() {
            return;
        }
        use crate::lattice::Equilibrium;
        use crate::moments;
        use crate::transport::column_uy;
        use std::f64::consts::PI;

        const SIZE: usize = 256;
        const DENSITY: f32 = 0.22;
        const U0: f32 = 0.25;
        const TRANSIENT: u64 = 200;
        const SAMPLES: usize = 20;
        // The phase turns at g * U * k, about 0.0027 rad a step, and the
        // amplitude decays with an e-folding time of some 5,000 steps. A
        // 2,000-step window turns far enough to fit and not so far that the
        // mode has gone.
        const INTERVAL: u64 = 100;
        const REPS: u64 = 16;
        let (w, h) = (SIZE, SIZE);
        let k = 2.0 * std::f32::consts::PI / w as f32;
        let amp = 0.05f32;

        let project = |cells: &[u8]| -> (f64, f64) {
            let u = column_uy(cells, w, h);
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for x in 0..w {
                let p = k * x as f32;
                a += (u[x] * p.cos()) as f64;
                b += (u[x] * p.sin()) as f64;
            }
            (2.0 * a / w as f64, 2.0 * b / w as f64)
        };
        let seeded = |rep: u64| -> Vec<u8> {
            let s = 0x5EED ^ (rep.wrapping_mul(0x9E37_79B9) << 20) ^ 0x1357;
            let mut rng = Rng::new(s ^ 0xF00D);
            let cols: Vec<Equilibrium> = (0..w)
                .map(|x| Equilibrium::new(DENSITY, U0, amp * (k * x as f32).sin(), true))
                .collect();
            let mut cells = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    cells[y * w + x] = cols[x].sample(&mut rng);
                }
            }
            cells
        };
        // The realised mean speed, not the requested one, is what advects it.
        let u_actual = moments::sum(&moments::WITH_WALLS, &seeded(0)).velocity().0;

        // Unwrap the phase and fit its drift.
        let fit = |trace: &[(f64, f64)]| -> f32 {
            let mut unwrapped = 0.0f64;
            let mut prev = trace[0].0.atan2(trace[0].1);
            let mut pts = Vec::with_capacity(trace.len());
            for (i, (a, b)) in trace.iter().enumerate() {
                let mut delta = a.atan2(*b) - prev;
                while delta > PI {
                    delta -= 2.0 * PI;
                }
                while delta < -PI {
                    delta += 2.0 * PI;
                }
                unwrapped += delta;
                prev = a.atan2(*b);
                pts.push(((i as u64 * INTERVAL) as f64, unwrapped));
            }
            let n = pts.len() as f64;
            let sx: f64 = pts.iter().map(|p| p.0).sum();
            let sy: f64 = pts.iter().map(|p| p.1).sum();
            let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
            let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
            let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            (-slope / (k * u_actual) as f64) as f32
        };

        let mut cpu_trace = vec![(0.0f64, 0.0f64); SAMPLES];
        let mut gpu_trace = vec![(0.0f64, 0.0f64); SAMPLES];
        for rep in 0..REPS {
            let start = seeded(rep);
            let seed = 0x5EED ^ (rep << 20);

            let mut cpu = Lattice::new(w, h, DENSITY, (U0, 0.0), 0, true, 4, seed);
            cpu.restart(&start, seed);
            cpu.advance(TRANSIENT);
            for slot in cpu_trace.iter_mut() {
                let (a, b) = project(&cpu.cells);
                slot.0 += a / REPS as f64;
                slot.1 += b / REPS as f64;
                cpu.advance(INTERVAL);
            }

            let mut gpu = GpuLattice::new(w, h, true, seed).unwrap();
            gpu.load(&start);
            gpu.advance(TRANSIENT);
            let mut cells = vec![0u8; w * h];
            for slot in gpu_trace.iter_mut() {
                gpu.store(&mut cells);
                let (a, b) = project(&cells);
                slot.0 += a / REPS as f64;
                slot.1 += b / REPS as f64;
                gpu.advance(INTERVAL);
            }
        }

        let (g_cpu, g_gpu) = (fit(&cpu_trace), fit(&gpu_trace));
        let rel = (g_gpu - g_cpu).abs() / g_cpu;
        println!("advection: cpu g = {g_cpu:.4}, gpu g = {g_gpu:.4} ({:.1}% apart)", 100.0 * rel);
        assert!(
            rel < 0.03,
            "the GPU advects differently: g = {g_gpu:.4} against the CPU's {g_cpu:.4}"
        );
    }

    /// The GPU is a second implementation of the same model, so it should
    /// reproduce the CPU's statistics even though the random streams differ.
    #[test]
    fn the_gpu_agrees_with_the_cpu_on_the_bulk_numbers() {
        if !device_or_skip() {
            return;
        }
        let (w, h) = (256usize, 128usize);
        let mut cpu = Lattice::new(w, h, 0.22, (0.3, 0.0), 0, true, 4, 0x5EED);
        cpu.init_equilibrium();
        let start = cpu.cells.clone();

        let mut g = GpuLattice::new(w, h, true, 0x5EED).unwrap();
        g.load(&start);
        cpu.advance(120);
        g.advance(120);
        let mut gcells = vec![0u8; w * h];
        g.store(&mut gcells);

        let gpu_lat = {
            let mut l = Lattice::new(w, h, 0.22, (0.3, 0.0), 0, true, 1, 1);
            l.cells.copy_from_slice(&gcells);
            l
        };
        let (ax, ay) = cpu.mean_velocity();
        let (bx, by) = gpu_lat.mean_velocity();
        let (ma, mb) = (cpu.total_particles() as f64, gpu_lat.total_particles() as f64);
        assert!((ax - bx).abs() < 0.02 && (ay - by).abs() < 0.02,
                "mean velocity: cpu ({ax:.4}, {ay:.4}) gpu ({bx:.4}, {by:.4})");
        assert!((ma - mb).abs() / ma < 0.01, "particle count: cpu {ma}, gpu {mb}");
    }
}
