#![cfg_attr(not(test), no_std)]
#![crate_type = "cdylib"]
#![crate_name = "sample"]

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(not(test))]
use core::num::Wrapping;
#[cfg(test)]
use std::num::Wrapping;

// Simple PRNG, random enough for our purposes
fn xorshift64star(state: &mut Wrapping<u64>) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (*state * Wrapping(0x2545f4914f6cdd1d)).0
}

// Use splitmix64 to seed xorshift
fn seed_prng(state: &mut Wrapping<u64>) {
    *state += Wrapping(0x9e3779b97f4a7c15);
    *state ^= *state >> 30; 
    *state *= Wrapping(0xbf58476d1ce4e5b9);
    *state ^= *state >> 27;
    *state *= Wrapping(0x94d049bb133111eb);
    *state ^= *state >> 31;
    // Ensure that seed is non-zero
    if state.0 == 0 {
        *state += Wrapping(1);
    }
}

// Generate uniform random float between 0 and 1
fn random(state: &mut Wrapping<u64>) -> f64 {
    let res = xorshift64star(state);
    (res >> 12) as f64 / (1u64 << 52) as f64
}

// Generate a uniform random integer between 0 and max (exclusive)
fn randrange(state: &mut Wrapping<u64>, max: u64) -> u64 {
    if max == 0 {
        return 0
    }

    let res = xorshift64star(state);
    res % max
}

fn fib(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

fn powi(mut f: f64, mut n: u64) -> f64 {
    if n == 0 {
        return 1.0
    }

    let mut y = 1.0;
    while n > 1 {
        if n & 1 != 0 {
            y *= f;
        }
        f *= f;
        n >>= 1;
    }
    f * y
}

static PHI: f64 = 1.618033988749894848;

fn sample<const IMAG: bool, const N: usize>(input: u64) -> u64 {
    let mut state = Wrapping(input as u64);
    seed_prng(&mut state);

    // This mirrors the code in preprocessing.py
    let mut bits = [false; N];
    bits[1] = true;
    bits[N - 1] = random(&mut state) < (fib(N as u64 - 1) as f64 / powi(PHI, N as u64 - 2));
    let (mut k, start) = if !bits[N - 1] {
        bits[N - 2] = true;
        (randrange(&mut state, fib(N as u64 - 2)), N - 3)
    } else {
        (randrange(&mut state, fib(N as u64 - 1)), N - 2)
    };

    let mut i = start;
    while i >= 2 {
        let f = fib(i as u64);
        if k >= f {
            bits[i] = false;
            k -= f;
        } else {
            bits[i] = true;
        }
        i -= 1;
    }

    // Pack bits into output 
    let mut output = 0;
    for i in 2..N {
        if bits[i] {
            output |= 1 << (i - 1);
        }
    }
    if IMAG {
        output |= 1
    }
    output
}

#[no_mangle]
pub extern fn init() {}

// Define external functions `sample_real` and `sample_imag` that can be
// accessed from TKET, with the appropriate real/imag part flag and
// number of qubits set.
#[no_mangle]
pub extern fn sample_real(input: u32) -> u32 {
    sample::<false, 16>(input as u64) as u32
}

#[no_mangle]
pub extern fn sample_imag(input: u32) -> u32 {
    sample::<true, 16>(input as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_no_consecutive() {
        let mask = 0b11111111111111111111111111111;
        let a = sample::<false, 32>(0xdeadbfff) >> 1;
        println!("{:b}", a);
        println!("{:b}", a | (a >> 1));
        assert_eq!((a | (a >> 1)) & mask, mask);

        let mask = 0b11111111111111111111111111111;
        let a = sample::<false, 32>(0x12341234) >> 1;
        println!("{:b}", a);
        println!("{:b}", a | (a >> 1));
        assert_eq!((a | (a >> 1)) & mask, mask);

        let mask = 0b11111111111111111111111111111;
        let a = sample::<false, 32>(0x00000000) >> 1;
        println!("{:b}", a);
        println!("{:b}", a | (a >> 1));
        assert_eq!((a | (a >> 1)) & mask, mask);

        let mask = 0b11111111111111111111111111111;
        let a = sample::<false, 32>(0xffffffffffffffff) >> 1;
        println!("{:b}", a);
        println!("{:b}", a | (a >> 1));
        assert_eq!((a | (a >> 1)) & mask, mask);
    }

    #[test]
    fn check_approx_finalbit() {
        let mut count = 0;
        for i in 0..1000 {
            let a = sample::<false, 32>(i);
            if (a & (1 << 30)) != 0 {
                count += 1;
            }
        }
        let p = count as f64 / 1000.0;
        println!("{}", p);
        assert!((0.71 < p) && (p < 0.74));
    }
}
