"""My own FIPS 180-4 SHA-256 with a reduced step count (written fresh for this check; shares no code with the
author's solver, the author's verifiers, or the audit's checker).
Constants come from exact integer roots of the first 64 primes."""
import math, hashlib

M = 0xffffffff

def _primes(n):
    ps, x = [], 2
    while len(ps) < n:
        if all(x % p for p in ps if p * p <= x):
            ps.append(x)
        x += 1
    return ps

def _icbrt(n):                      # floor(cuberoot(n)) by Newton on integers
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x ** 3 > n: x -= 1
    while (x + 1) ** 3 <= n: x += 1
    return x

PR = _primes(64)
K = [_icbrt(p << 96) & M for p in PR]                 # frac(cbrt(p)) * 2^32
H0 = [math.isqrt(p << 64) & M for p in PR[:8]]        # frac(sqrt(p)) * 2^32
assert K[0] == 0x428a2f98 and K[63] == 0xc67178f2 and H0[0] == 0x6a09e667 and H0[7] == 0x5be0cd19

def ror(x, n): return ((x >> n) | (x << (32 - n))) & M

def pad(msg):
    L = len(msg) * 8
    m = msg + b'\x80'
    while len(m) % 64 != 56:
        m += b'\x00'
    return m + L.to_bytes(8, 'big')

def compress(state, block, steps):
    W = [int.from_bytes(block[4*i:4*i+4], 'big') for i in range(16)]
    for t in range(16, max(steps, 16)):
        s0 = ror(W[t-15], 7) ^ ror(W[t-15], 18) ^ (W[t-15] >> 3)
        s1 = ror(W[t-2], 17) ^ ror(W[t-2], 19) ^ (W[t-2] >> 10)
        W.append((W[t-16] + s0 + W[t-7] + s1) & M)
    a, b, c, d, e, f, g, h = state
    for t in range(steps):
        T1 = (h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g & M)) + K[t] + W[t]) & M
        T2 = ((ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & M, c, b, a, (T1 + T2) & M
    return [(x + y) & M for x, y in zip(state, [a, b, c, d, e, f, g, h])]

def hash_steps(msg, steps):
    st = list(H0)
    m = pad(msg)
    for i in range(0, len(m), 64):
        st = compress(st, m[i:i+64], steps)
    return ''.join('%08x' % x for x in st)

def comp_steps(block, steps):
    return ''.join('%08x' % x for x in compress(list(H0), block, steps))

if __name__ == '__main__':
    import random
    rng = random.Random(7)
    n = 0
    for L in list(range(0, 200)) + [255, 256, 1000, 4097]:
        m = bytes(rng.getrandbits(8) for _ in range(L))
        assert hash_steps(m, 64) == hashlib.sha256(m).hexdigest(), L
        n += 1
    m = b'a' * 1000000
    assert hash_steps(m, 64) == hashlib.sha256(m).hexdigest()
    print("self-test OK:", n + 1, "messages (lengths 0..199, 255, 256, 1000, 4097, 1000000) agree with hashlib at 64 steps")
