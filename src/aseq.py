"""
a-sequence toolkit for SHA-256 (scalar Python ints, mod 2^32).

Index convention: a[t] for t >= -7 (a[-7..0] implied by the IV), round t+1 consumes W_t.
  e_s = a_s - S0(a_{s-1}) - Maj(a_{s-1},a_{s-2},a_{s-3}) + a_{s-4}
  W_t = e_{t+1} - a_{t-3} - e_{t-3} - S1(e_t) - Ch(e_t,e_{t-1},e_{t-2}) - K_t
The state after r rounds is (a_r, a_{r-1}, a_{r-2}, a_{r-3}, e_r, e_{r-1}, e_{r-2}, e_{r-3}).
"""
M = 0xFFFFFFFF
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def rotr(x, n): return ((x >> n) | (x << (32 - n))) & M
def S0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def S1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
def s0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def s1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
def ch(e, f, g): return ((e & f) ^ (~e & g)) & M
def maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)
def Q(u): return (s0(u) - u) & M


def window_from_state(st):
    """state (a,b,c,d,e,f,g,h) after r rounds -> [a_{r-7}, ..., a_r]."""
    a, b, c, d, e, f, g, h = st
    A = {0: a, 1: b, 2: c, 3: d}
    E = {0: e, 1: f, 2: g, 3: h}
    for k in range(4):
        A[4 + k] = (E[k] - A[k] + S0(A[k + 1]) + maj(A[k + 1], A[k + 2], A[k + 3])) & M
    return [A[7 - i] for i in range(8)]


IVA = window_from_state(IV)          # a_{-7} .. a_0


class ASeq:
    """dict-like a-sequence with IV prefix; e_s and W_t computed on demand."""

    def __init__(self, r):
        self.r = r
        self.a = {i - 7: IVA[i] for i in range(8)}

    def e(self, s):
        a = self.a
        return (a[s] - S0(a[s - 1]) - maj(a[s - 1], a[s - 2], a[s - 3]) + a[s - 4]) & M

    def W(self, t):
        a = self.a
        e = self.e
        et = e(t)
        return (e(t + 1) - a[t - 3] - e(t - 3) - S1(et) - ch(et, e(t - 1), e(t - 2)) - K[t]) & M

    def set_target(self, state):
        w = window_from_state(state)
        for i in range(8):
            self.a[self.r - 7 + i] = w[i]


def compress(Wmsg, r, iv=IV):
    """r-round compression (no feed-forward) of 16 message words; returns state tuple."""
    W = list(Wmsg)
    for t in range(16, r):
        W.append((s1(W[t - 2]) + W[t - 7] + s0(W[t - 15]) + W[t - 16]) & M)
    a, b, c, d, e, f, g, h = iv
    for t in range(r):
        T1 = (h + S1(e) + ch(e, f, g) + K[t] + W[t]) & M
        T2 = (S0(a) + maj(a, b, c)) & M
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & M, c, b, a, (T1 + T2) & M
    return (a, b, c, d, e, f, g, h)


def digest_words(Wmsg, r, iv=IV):
    st = compress(Wmsg, r, iv)
    return tuple((st[i] + iv[i]) & M for i in range(8))


def pad_single_block(msg: bytes):
    assert len(msg) <= 55
    L = len(msg)
    block = msg + b'\x80' + b'\x00' * (55 - L) + (8 * L).to_bytes(8, 'big')
    return [int.from_bytes(block[4 * i:4 * i + 4], 'big') for i in range(16)]


def sha256_reduced(msg: bytes, r: int):
    """r-step SHA-256 of a single-block message (standard padding, standard IV, feed-forward)."""
    return b''.join(x.to_bytes(4, 'big') for x in digest_words(pad_single_block(msg), r))


if __name__ == '__main__':
    import hashlib, os
    for _ in range(50):
        m = os.urandom(os.urandom(1)[0] % 56)
        assert sha256_reduced(m, 64) == hashlib.sha256(m).digest()
    print('aseq: 64-round single-block SHA-256 matches hashlib on 50 random messages')
