//! Minimal PNG writer (8-bit RGB, no filtering, stored deflate blocks) so that
//! frames are viewable anywhere without pulling in an image crate.

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    c ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Bit-level output in deflate's order: the stream is filled from the least
/// significant bit up, while Huffman codes are written most significant bit
/// first.
struct BitWriter {
    out: Vec<u8>,
    buf: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), buf: 0, n: 0 }
    }

    fn bits(&mut self, value: u32, count: u32) {
        self.buf |= value << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, whose bits go out in the opposite order.
    fn code(&mut self, value: u32, count: u32) {
        let mut reversed = 0;
        for i in 0..count {
            reversed |= ((value >> i) & 1) << (count - 1 - i);
        }
        self.bits(reversed, count);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.buf as u8);
        }
        self.out
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// The fixed Huffman code for a literal or length symbol.
fn fixed_symbol(sym: u32) -> (u32, u32) {
    match sym {
        0..=143 => (0b0011_0000 + sym, 8),
        144..=255 => (0b1_1001_0000 + (sym - 144), 9),
        256..=279 => (sym - 256, 7),
        _ => (0b1100_0000 + (sym - 280), 8),
    }
}

const WINDOW: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
/// Cap on how far back to look for a better match. The images are blocks of
/// identical pixels, so the first candidate is nearly always already the long
/// one; a short chain costs almost nothing in ratio.
const MAX_CHAIN: usize = 32;

fn hash3(d: &[u8]) -> usize {
    ((d[0] as usize) << 10 ^ (d[1] as usize) << 5 ^ d[2] as usize) & 0x7FFF
}

/// Deflate with fixed Huffman codes and greedy LZ77 matching.
fn deflate(data: &[u8]) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.bits(1, 1); // final block
    bw.bits(1, 2); // fixed Huffman

    let n = data.len();
    let mut head = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; n.max(1)];

    let mut i = 0;
    while i < n {
        let (mut best_len, mut best_dist) = (0usize, 0usize);
        if i + MIN_MATCH <= n {
            let limit = (n - i).min(MAX_MATCH);
            let mut cand = head[hash3(&data[i..])];
            let mut chain = 0;
            while cand != usize::MAX && chain < MAX_CHAIN && i - cand <= WINDOW {
                let mut l = 0;
                while l < limit && data[cand + l] == data[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = i - cand;
                    if l == limit {
                        break;
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
        }

        let advance = if best_len >= MIN_MATCH {
            let li = LEN_BASE.partition_point(|&b| b as usize <= best_len) - 1;
            let (c, bits) = fixed_symbol(257 + li as u32);
            bw.code(c, bits);
            bw.bits(best_len as u32 - LEN_BASE[li] as u32, LEN_EXTRA[li]);

            let di = DIST_BASE.partition_point(|&b| b as usize <= best_dist) - 1;
            bw.code(di as u32, 5);
            bw.bits(best_dist as u32 - DIST_BASE[di] as u32, DIST_EXTRA[di]);
            best_len
        } else {
            let (c, bits) = fixed_symbol(data[i] as u32);
            bw.code(c, bits);
            1
        };

        for k in i..i + advance {
            if k + MIN_MATCH <= n {
                let h = hash3(&data[k..]);
                prev[k] = head[h];
                head[h] = k;
            }
        }
        i += advance;
    }

    let (c, bits) = fixed_symbol(256); // end of block
    bw.code(c, bits);
    bw.finish()
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let mut framed = Vec::with_capacity(4 + body.len());
    framed.extend_from_slice(kind);
    framed.extend_from_slice(body);
    out.extend_from_slice(&framed);
    out.extend_from_slice(&crc32(&framed).to_be_bytes());
}

/// `rgb` holds `w * h * 3` bytes.
pub fn write_rgb(path: &std::path::Path, w: usize, h: usize, rgb: &[u8]) -> std::io::Result<()> {
    assert_eq!(rgb.len(), w * h * 3);

    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0); // filter type: none
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }

    let mut z = vec![0x78, 0x01];
    z.extend_from_slice(&deflate(&raw));
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, truecolour
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);

    std::fs::write(path, png)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Just enough inflate to read back what `deflate` writes: one fixed
    /// Huffman block. Its only job is to catch the compressor drifting.
    struct BitReader<'a> {
        d: &'a [u8],
        pos: usize,
    }

    impl BitReader<'_> {
        fn bit(&mut self) -> u32 {
            let b = (self.d[self.pos >> 3] >> (self.pos & 7)) & 1;
            self.pos += 1;
            b as u32
        }

        /// Little-endian, as used for the extra bits of lengths and distances.
        fn bits(&mut self, n: u32) -> u32 {
            (0..n).map(|i| self.bit() << i).sum()
        }

        /// Big-endian, as used for Huffman codes.
        fn code(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, _| (acc << 1) | self.bit())
        }
    }

    fn inflate_fixed(data: &[u8]) -> Vec<u8> {
        let mut r = BitReader { d: data, pos: 0 };
        assert_eq!(r.bit(), 1, "expected a final block");
        assert_eq!(r.bits(2), 1, "expected fixed Huffman");

        let mut out: Vec<u8> = Vec::new();
        loop {
            // Walk the fixed literal/length code one bit at a time.
            let mut code = r.code(7);
            let mut len = 7;
            let sym = loop {
                match len {
                    7 if code <= 0b001_0111 => break 256 + code,
                    8 if (0b0011_0000..=0b1011_1111).contains(&code) => {
                        break code - 0b0011_0000
                    }
                    8 if (0b1100_0000..=0b1100_0111).contains(&code) => {
                        break 280 + code - 0b1100_0000
                    }
                    9 => break 144 + code - 0b1_1001_0000,
                    _ => {
                        code = (code << 1) | r.bit();
                        len += 1;
                    }
                }
            };

            if sym == 256 {
                return out;
            }
            if sym < 256 {
                out.push(sym as u8);
                continue;
            }

            let li = (sym - 257) as usize;
            let length = LEN_BASE[li] as usize + r.bits(LEN_EXTRA[li]) as usize;
            let di = r.code(5) as usize;
            let dist = DIST_BASE[di] as usize + r.bits(DIST_EXTRA[di]) as usize;
            for _ in 0..length {
                out.push(out[out.len() - dist]);
            }
        }
    }

    fn roundtrip(data: &[u8]) {
        let packed = deflate(data);
        assert_eq!(inflate_fixed(&packed), data, "round trip failed");
    }

    #[test]
    fn deflate_round_trips() {
        roundtrip(b"a");
        roundtrip(b"abcabcabcabcabcabc");
        roundtrip(&vec![7u8; 5000]); // long matches
        roundtrip(&(0..=255u8).cycle().take(4000).collect::<Vec<_>>());

        // Something shaped like a rendered frame: blocks of repeated pixels
        // laid out in repeated rows.
        let (w, block) = (120usize, 8usize);
        let mut img = Vec::new();
        let mut seed = 1u32;
        let mut row = Vec::new();
        for i in 0..w {
            if i % block == 0 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            }
            row.extend_from_slice(&[(seed >> 16) as u8, (seed >> 8) as u8, seed as u8]);
        }
        for y in 0..40 {
            img.push(0u8); // filter byte
            img.extend_from_slice(&row);
            let _ = y;
        }
        roundtrip(&img);
    }

    #[test]
    fn deflate_actually_compresses() {
        let flat = vec![3u8; 100_000];
        assert!(deflate(&flat).len() < 1000, "runs should collapse");
    }
}
