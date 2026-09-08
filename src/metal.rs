//! A hand-rolled binding to Metal, so that the GPU path costs no dependencies.
//!
//! Metal has no C API; everything goes through Objective-C message sends. That
//! is less forbidding than it sounds --- `objc_msgSend` is an ordinary C symbol,
//! and a message send is a call to it with the receiver and a selector in front
//! of the arguments. The only real care needed is that `objc_msgSend` has no
//! single type, so every call site casts it to the signature it wants. The
//! `msg*` helpers below keep all of those casts in one place.
//!
//! Buffers are allocated shared rather than private. On Apple silicon the CPU
//! and GPU are looking at the same memory, so a shared buffer is not a copy of
//! the lattice --- it *is* the lattice, and the analysis passes can read it
//! without any transfer at all.

use std::ffi::{c_char, c_void, CStr, CString};

pub type Id = *mut c_void;
type Sel = *const c_void;

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "objc")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

unsafe fn msg0<R>(o: Id, s: Sel) -> R {
    let f: extern "C" fn(Id, Sel) -> R = std::mem::transmute(objc_msgSend as *const ());
    f(o, s)
}
unsafe fn msg1<A, R>(o: Id, s: Sel, a: A) -> R {
    let f: extern "C" fn(Id, Sel, A) -> R = std::mem::transmute(objc_msgSend as *const ());
    f(o, s, a)
}
unsafe fn msg2<A, B, R>(o: Id, s: Sel, a: A, b: B) -> R {
    let f: extern "C" fn(Id, Sel, A, B) -> R = std::mem::transmute(objc_msgSend as *const ());
    f(o, s, a, b)
}
unsafe fn msg3<A, B, C, R>(o: Id, s: Sel, a: A, b: B, c: C) -> R {
    let f: extern "C" fn(Id, Sel, A, B, C) -> R = std::mem::transmute(objc_msgSend as *const ());
    f(o, s, a, b, c)
}

fn sel(name: &str) -> Sel {
    let c = CString::new(name).expect("selector name");
    unsafe { sel_registerName(c.as_ptr()) }
}

fn class(name: &str) -> Id {
    let c = CString::new(name).expect("class name");
    unsafe { objc_getClass(c.as_ptr()) }
}

/// `MTLSize`. Larger than sixteen bytes, so the calling convention passes it
/// indirectly; declaring it by value here lets Rust's `extern "C"` do that.
#[repr(C)]
#[derive(Clone, Copy)]
struct MtlSize {
    w: u64,
    h: u64,
    d: u64,
}

/// An owned Objective-C object, released when it goes out of scope.
struct Obj(Id);

impl Drop for Obj {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { msg0::<()>(self.0, sel("release")) }
        }
    }
}

/// Scope for the objects Metal hands back autoreleased, which is most of the
/// per-step ones. Without a pool in scope they would simply accumulate.
struct Pool(*mut c_void);

impl Pool {
    fn new() -> Self {
        unsafe { Pool(objc_autoreleasePoolPush()) }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}

fn nsstring(s: &str) -> Obj {
    let c = CString::new(s).expect("string with no interior nul");
    unsafe {
        let a: Id = msg0(class("NSString"), sel("alloc"));
        Obj(msg1(a, sel("initWithUTF8String:"), c.as_ptr()))
    }
}

fn error_text(e: Id) -> String {
    if e.is_null() {
        return "(no error object)".into();
    }
    unsafe {
        let d: Id = msg0(e, sel("localizedDescription"));
        let p: *const c_char = msg0(d, sel("UTF8String"));
        if p.is_null() {
            "(no description)".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

pub struct Device {
    dev: Id,
    queue: Obj,
}

pub struct Pipeline(Obj);

pub struct Buffer {
    buf: Obj,
    bytes: usize,
}

impl Device {
    /// `None` when there is no Metal device, which is the signal to fall back
    /// to the CPU implementation rather than an error.
    pub fn new() -> Option<Device> {
        unsafe {
            let dev = MTLCreateSystemDefaultDevice();
            if dev.is_null() {
                return None;
            }
            let queue: Id = msg0(dev, sel("newCommandQueue"));
            if queue.is_null() {
                return None;
            }
            Some(Device { dev, queue: Obj(queue) })
        }
    }

    pub fn name(&self) -> String {
        unsafe {
            let n: Id = msg0(self.dev, sel("name"));
            let p: *const c_char = msg0(n, sel("UTF8String"));
            if p.is_null() {
                "(unnamed device)".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    /// Compile one function out of a source string. The error carries the
    /// compiler's own diagnostics, which is the whole reason for the `Result`.
    pub fn pipeline(&self, source: &str, function: &str) -> Result<Pipeline, String> {
        unsafe {
            let _pool = Pool::new();
            let src = nsstring(source);
            let mut err: Id = std::ptr::null_mut();
            let lib: Id = msg3(
                self.dev,
                sel("newLibraryWithSource:options:error:"),
                src.0,
                std::ptr::null_mut::<c_void>(),
                &mut err as *mut Id,
            );
            if lib.is_null() {
                return Err(format!("shader compilation failed:\n{}", error_text(err)));
            }
            let lib = Obj(lib);
            let name = nsstring(function);
            let func: Id = msg1(lib.0, sel("newFunctionWithName:"), name.0);
            if func.is_null() {
                return Err(format!("the shader defines no function called {function}"));
            }
            let func = Obj(func);
            let mut err2: Id = std::ptr::null_mut();
            let pso: Id = msg2(
                self.dev,
                sel("newComputePipelineStateWithFunction:error:"),
                func.0,
                &mut err2 as *mut Id,
            );
            if pso.is_null() {
                return Err(format!("pipeline state failed:\n{}", error_text(err2)));
            }
            Ok(Pipeline(Obj(pso)))
        }
    }

    /// Zero-filled shared storage: one allocation the CPU and GPU both address.
    pub fn buffer(&self, bytes: usize) -> Buffer {
        unsafe {
            let b: Id = msg2(
                self.dev,
                sel("newBufferWithLength:options:"),
                bytes as u64,
                0u64, // MTLResourceStorageModeShared
            );
            assert!(!b.is_null(), "could not allocate a {bytes}-byte Metal buffer");
            Buffer { buf: Obj(b), bytes }
        }
    }

    /// Start a command buffer. Dispatches encoded into one of these run back to
    /// back on the GPU with no round trip between them, which is what makes a
    /// step cost the step rather than the submission.
    pub fn batch(&self) -> Batch<'_> {
        unsafe {
            let pool = Pool::new();
            let cb: Id = msg0(self.queue.0, sel("commandBuffer"));
            let enc: Id = msg0(cb, sel("computeCommandEncoder"));
            Batch { cb, enc, _pool: pool, _dev: self, n: 0 }
        }
    }
}

pub struct Batch<'a> {
    cb: Id,
    enc: Id,
    _pool: Pool,
    _dev: &'a Device,
    n: usize,
}

impl Batch<'_> {
    /// One dispatch. `params` is passed inline as the buffer just past `bufs`,
    /// so small per-step arguments need no allocation.
    pub fn dispatch(&mut self, p: &Pipeline, bufs: &[&Buffer], params: &[u32], threads: u64) {
        if threads == 0 {
            return;
        }
        unsafe {
            msg1::<Id, ()>(self.enc, sel("setComputePipelineState:"), p.0 .0);
            for (i, b) in bufs.iter().enumerate() {
                msg3::<Id, u64, u64, ()>(
                    self.enc,
                    sel("setBuffer:offset:atIndex:"),
                    b.buf.0,
                    0,
                    i as u64,
                );
            }
            if !params.is_empty() {
                msg3::<*const c_void, u64, u64, ()>(
                    self.enc,
                    sel("setBytes:length:atIndex:"),
                    params.as_ptr() as *const c_void,
                    std::mem::size_of_val(params) as u64,
                    bufs.len() as u64,
                );
            }
            // 256 is a comfortable group size for a kernel this register-hungry;
            // `dispatchThreads` handles a grid that is not a multiple of it.
            msg2::<MtlSize, MtlSize, ()>(
                self.enc,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MtlSize { w: threads, h: 1, d: 1 },
                MtlSize { w: 256, h: 1, d: 1 },
            );
        }
        self.n += 1;
    }

    pub fn dispatches(&self) -> usize {
        self.n
    }

    /// Submit, and block until the GPU is done.
    pub fn wait(self) {
        unsafe {
            msg0::<()>(self.enc, sel("endEncoding"));
            msg0::<()>(self.cb, sel("commit"));
            msg0::<()>(self.cb, sel("waitUntilCompleted"));
        }
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn contents(&self) -> *mut c_void {
        unsafe { msg0(self.buf.0, sel("contents")) }
    }

    pub fn as_slice<T: Copy>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.contents() as *const T,
                self.bytes / std::mem::size_of::<T>(),
            )
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.contents() as *mut T,
                self.bytes / std::mem::size_of::<T>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOUBLE: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void twiddle(device const uint* a [[buffer(0)]],
                    device uint* b [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    b[i] = a[i] * 3u + 1u;
}";

    #[test]
    fn a_kernel_runs_and_the_result_comes_back() {
        let Some(dev) = Device::new() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let pso = dev.pipeline(DOUBLE, "twiddle").expect("pipeline");
        let n = 4096usize;
        let mut a = dev.buffer(n * 4);
        let b = dev.buffer(n * 4);
        for (i, v) in a.as_mut_slice::<u32>().iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut batch = dev.batch();
        batch.dispatch(&pso, &[&a, &b], &[n as u32], n as u64);
        batch.wait();
        let got = b.as_slice::<u32>();
        assert!((0..n).all(|i| got[i] == i as u32 * 3 + 1));
    }

    #[test]
    fn a_bad_shader_reports_the_compiler_diagnostic() {
        let Some(dev) = Device::new() else { return };
        let err = match dev.pipeline("kernel void bad() { not metal at all; }", "bad") {
            Err(e) => e,
            Ok(_) => panic!("that should not have compiled"),
        };
        assert!(err.contains("error"), "unhelpful diagnostic: {err}");
    }
}
