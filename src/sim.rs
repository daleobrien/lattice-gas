//! The simulation, on whichever processor is running it.
//!
//! The two paths do the same physics but keep the lattice in different
//! shapes: one byte per cell on the CPU, one bit per cell per direction on the
//! GPU. What they have in common is a run loop that advances some steps,
//! folds the lattice into a coarse-grained field every few of them, and wants
//! that field on the CPU only when a frame is written.
//!
//! That last part is the whole reason this type exists rather than a bare
//! branch at each call site. A GPU round trip costs about seven steps, so a
//! program that synchronised every fifth step would spend more time waiting
//! than simulating. `advance_sampling` therefore takes the *whole* interval
//! between frames and does the sampling inside it, so the caller synchronises
//! once per frame instead of once per sample.

use crate::lattice::Lattice;
use crate::render::Field;

pub struct Sim {
    backend: Backend,
}

enum Backend {
    Cpu(Lattice),
    /// The lattice, and a byte-per-cell cache for the analysis passes that
    /// still want to look at it that way. Empty until something asks.
    #[cfg(target_os = "macos")]
    Gpu(Box<crate::gpu::GpuLattice>, Vec<u8>),
}

impl Sim {
    /// Take a prepared lattice --- obstacles marked, cells initialised --- and
    /// run it on the GPU if one is wanted and available. The second return
    /// value says where it ended up and why, for the caller to print.
    pub fn new(lat: Lattice, want_gpu: bool) -> (Sim, String) {
        let threads = lat.threads();
        let cpu = |note: &str| format!("CPU, {threads} threads{note}");
        if !want_gpu {
            return (Sim { backend: Backend::Cpu(lat) }, cpu(""));
        }

        #[cfg(target_os = "macos")]
        match Self::on_gpu(&lat) {
            Ok((gpu, name)) => (
                Sim { backend: Backend::Gpu(Box::new(gpu), Vec::new()) },
                format!("GPU ({name}), 32 cells to a word"),
            ),
            // Not an error: no Metal device, or a driver that will not compile
            // the shader, just means the CPU path runs instead.
            Err(e) => (
                Sim { backend: Backend::Cpu(lat) },
                cpu(&format!(" (--gpu asked for, but {e})")),
            ),
        }
        #[cfg(not(target_os = "macos"))]
        (Sim { backend: Backend::Cpu(lat) }, cpu(" (--gpu needs Metal, so macOS)"))
    }

    #[cfg(target_os = "macos")]
    fn on_gpu(lat: &Lattice) -> Result<(crate::gpu::GpuLattice, String), String> {
        let rest = lat.table.rest_particles;
        let mut g = crate::gpu::GpuLattice::new(lat.w, lat.h, rest, lat.seed())?;
        g.set_inlet(lat.inlet_cols, lat.density, lat.inflow, rest);
        g.load(&lat.cells);
        let name = g.device_name();
        Ok((g, name))
    }

    pub fn on_gpu_now(&self) -> bool {
        match &self.backend {
            Backend::Cpu(_) => false,
            #[cfg(target_os = "macos")]
            Backend::Gpu(..) => true,
        }
    }

    /// Give the simulation somewhere to accumulate a coarse-grained field.
    /// Only the GPU path needs telling; the CPU one writes straight into the
    /// `Field` it is handed.
    pub fn attach_field(&mut self, field: &Field) {
        match &mut self.backend {
            Backend::Cpu(_) => {}
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => g.attach_field(field.bx, field.by),
        }
    }

    /// Put the lattice back to a given state and random seed, for the next
    /// realisation of a measurement. Reusing one simulation matters on the
    /// GPU, where compiling the shader costs more than a short run does.
    pub fn restart(&mut self, cells: &[u8], seed: u64) {
        match &mut self.backend {
            Backend::Cpu(lat) => lat.restart(cells, seed),
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => g.restart(cells, seed),
        }
    }

    /// The lattice as one byte per cell. On the GPU this gathers the planes
    /// into a cache, so it is a real cost --- fine for the once-per-sample
    /// analysis passes, not for anything per step.
    pub fn cells(&mut self) -> &[u8] {
        match &mut self.backend {
            Backend::Cpu(lat) => &lat.cells,
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, cache) => {
                if cache.len() != g.w * g.h {
                    cache.resize(g.w * g.h, 0);
                }
                g.store(cache);
                cache
            }
        }
    }

    /// Advance `steps` steps, folding the lattice into `field` every
    /// `sample_every` of them and at the end of the run, exactly as the CPU
    /// loop has always done. `field` is up to date when this returns.
    pub fn advance_sampling(
        &mut self,
        steps: u64,
        sample_every: u64,
        alpha: f32,
        field: &mut Field,
    ) {
        let every = sample_every.max(1);
        match &mut self.backend {
            Backend::Cpu(lat) => {
                let mut left = steps;
                while left > 0 {
                    let take = every.min(left);
                    lat.advance(take);
                    field.sample(lat, alpha);
                    left -= take;
                }
            }
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => {
                g.advance_sampling(steps, every, alpha);
                g.read_field(field);
            }
        }
    }

    /// One sample of the state as it stands, with no stepping: the first one,
    /// taken after the warmup to give the running average something to start
    /// from.
    pub fn sample(&mut self, field: &mut Field, alpha: f32) {
        match &mut self.backend {
            Backend::Cpu(lat) => field.sample(lat, alpha),
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => {
                g.sample_field(alpha);
                g.read_field(field);
            }
        }
    }

    pub fn advance(&mut self, steps: u64) {
        match &mut self.backend {
            Backend::Cpu(lat) => lat.advance(steps),
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => g.advance(steps),
        }
    }

    /// Particles on the lattice, obstacle sites included.
    pub fn total_particles(&mut self) -> u64 {
        match &mut self.backend {
            Backend::Cpu(lat) => lat.total_particles(),
            #[cfg(target_os = "macos")]
            Backend::Gpu(g, _) => g.total_particles(),
        }
    }
}
