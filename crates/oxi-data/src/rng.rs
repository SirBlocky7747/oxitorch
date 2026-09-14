//! Seeded MT19937 RNG with NumPy-compatible `random` / `permutation` streams.
//!
//! NumPy's legacy `RandomState` uses MT19937 with `genrand_res53` for
//! `random_sample()` and a Fisher-Yates shuffle for `permutation()`;
//! matching both lets Rust reproduce Python-side shuffle orders exactly
//! (verified against recorded NumPy streams in the tests).

/// MT19937 state (624-word array + index).
pub struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl Mt19937 {
    /// Creates a generator seeded like `np.random.RandomState(seed)` for
    /// non-negative 32-bit seeds.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, mti: 624 }
    }

    /// Draws one full-precision `u32`: the tempered MT19937 output.
    fn next_u32(&mut self) -> u32 {
        if self.mti >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        // Tempering (the step that makes MT19937's output uniform-looking).
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX: u32 = 0x9908_b0df;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut next = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX;
            }
            self.mt[i] = next;
        }
        self.mti = 0;
    }

    /// Uniform `[0, 1)` on the same 53-bit grid as NumPy's `random_sample`.
    #[must_use]
    pub fn random(&mut self) -> f32 {
        // NumPy rk_double: a = first >> 5 (27 bits), b = second >> 6 (26 bits);
        // (a*67108864 + b) / 9007199254740992.
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        let v = (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0;
        v as f32
    }

    /// Uniform `[low, high)` floats.
    #[must_use]
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.random()
    }

    /// Standard normal via Box–Muller on `random()` pairs.
    #[must_use]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.random().max(f32::MIN_POSITIVE); // avoid log(0)
        let u2 = self.random();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// `numpy.random.RandomState.permutation(n)` equivalent: Fisher–Yates
    /// over `0..n` using rejection sampling on 32-bit words.
    #[must_use]
    pub fn permutation(&mut self, n: usize) -> Vec<u64> {
        let mut idx: Vec<u64> = (0..n as u64).collect();
        // Fisher–Yates, exactly NumPy's loop: for i in (1..n).rev().
        for i in (1..n).rev() {
            let j = self.randbelow(i + 1);
            idx.swap(i, j);
        }
        idx
    }

    /// Uniform integer in `[0, bound)`; the public face of the NumPy-exact
    /// masked-rejection sampler (used by per-image crop positions).
    #[must_use]
    pub fn below(&mut self, bound: usize) -> usize {
        self.randbelow(bound)
    }

    /// Uniform integer in `[0, bound)` matching NumPy's masked rejection
    /// (`rk_bounded_uint32`): accept the first masked draw `<= bound - 1`.
    fn randbelow(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let rng = (bound - 1) as u32; // inclusive maximum
        let mut mask = rng;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let buf = self.next_u32() & mask;
            if buf <= rng {
                return buf as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NumPy reference streams (recorded from np.random.RandomState).
    #[test]
    fn matches_numpy_random_stream() {
        let mut r = Mt19937::new(42);
        let got: Vec<f32> = (0..5).map(|_| r.random()).collect();
        // np.random.RandomState(42).random_sample(5)
        let want = [
            0.374_540_12,
            0.950_714_3,
            0.731_993_9,
            0.598_658_5,
            0.156_018_64,
        ];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[test]
    fn matches_numpy_permutation() {
        let mut r = Mt19937::new(7);
        let got = r.permutation(10);
        // np.random.RandomState(7).permutation(10)
        let want = [8, 5, 0, 2, 1, 9, 7, 3, 6, 4];
        assert_eq!(got, want);
    }

    #[test]
    fn normal_is_reasonably_distributed() {
        let mut r = Mt19937::new(0);
        let n = 20_000;
        let sum: f64 = (0..n).map(|_| f64::from(r.normal())).sum();
        let mean = sum / f64::from(n);
        assert!(mean.abs() < 0.05, "mean {mean}");
    }
}
