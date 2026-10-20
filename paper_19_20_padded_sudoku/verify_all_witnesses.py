#!/usr/bin/env python3
"""
Standalone Zero-Dependency Verifier for:
  "Practical Preimage Attacks on 19-Step and 20-Step Padded SHA-256
   via Carry-Sudoku and Slot-Sudoku Decomposition"

Verifies:
  1. 20-Step Padded SHA-256 Preimage Witness (Target: All-Zero Digest 0^256)
     + Merkle-Damgard Padding (W13 & 0xff == 0x80, W14 == 0, W15 == 440)
     + Double-Cancel & Carry-Sudoku Invariants (a5 == a6, a10 == a11, e9 == e10 == e15 == 0xffffffff)
  2. 19-Step Padded SHA-256 Preimage Witness (Target: SHA-256("abc"))
     + Merkle-Damgard Padding (W13 & 0xff == 0x80, W14 == 0, W15 == 440)
  3. 20-Step Unpadded SHA-256 Preimage Witness (Target: deadbeef... pattern)
"""

import sys

IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
]

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xffffffff

def Sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def Sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def sigma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def sigma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

def Ch(e, f, g):
    return (e & f) ^ ((~e & 0xffffffff) & g)

def Maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)

def sha256_reduced(W16, rounds):
    W = list(W16)
    for t in range(16, rounds):
        w = (sigma1(W[t - 2]) + W[t - 7] + sigma0(W[t - 15]) + W[t - 16]) & 0xffffffff
        W.append(w)

    state = list(IV)
    trace = [tuple(state)]
    for t in range(rounds):
        a, b, c, d, e, f, g, h = state
        T1 = (h + Sigma1(e) + Ch(e, f, g) + K[t] + W[t]) & 0xffffffff
        T2 = (Sigma0(a) + Maj(a, b, c)) & 0xffffffff
        state = [
            (T1 + T2) & 0xffffffff,
            a, b, c,
            (d + T1) & 0xffffffff,
            e, f, g
        ]
        trace.append(tuple(state))

    digest = [(IV[i] + state[i]) & 0xffffffff for i in range(8)]
    return digest, trace, W

def fmt_words(words):
    return "".join(f"{w:08x}" for w in words)

def main():
    print("=" * 78)
    print("1. VERIFYING 20-STEP PADDED SHA-256 PREIMAGE WITNESS (All-Zero Target 0^256)")
    print("=" * 78)
    W20_padded = [
        0xec531306, 0xc85ad4a4, 0xad211d2e, 0x1a03ece4,
        0xd0ad4a0e, 0x6b4c2b94, 0x36b3babd, 0xcbd52059,
        0x7196f52c, 0x042b8cd9, 0x5f1a39f7, 0x1b58b47b,
        0x0647c78e, 0x65416f80, 0x00000000, 0x000001b8,
    ]
    digest20p, trace20p, W20p_exp = sha256_reduced(W20_padded, 20)
    print(f"Input W[0..15] : {' '.join(f'{w:08x}' for w in W20_padded)}")
    print(f"Expanded W[16..19]: {' '.join(f'{w:08x}' for w in W20p_exp[16:20])}")
    print(f"20-Step Digest : {fmt_words(digest20p)}")

    assert all(x == 0 for x in digest20p), "Digest is not all-zeros!"
    assert (W20_padded[13] & 0xff) == 0x80, "W13 low byte is not 0x80!"
    assert W20_padded[14] == 0x00000000, "W14 is not 0!"
    assert W20_padded[15] == 0x000001b8, "W15 is not 440 (0x1b8)!"

    a5, e5 = trace20p[5][0], trace20p[5][4]
    a6, e6 = trace20p[6][0], trace20p[6][4]
    a9, e9 = trace20p[9][0], trace20p[9][4]
    a10, e10 = trace20p[10][0], trace20p[10][4]
    a11, e11 = trace20p[11][0], trace20p[11][4]
    a15, e15 = trace20p[15][0], trace20p[15][4]

    print(f"Padding Check  : W13 & 0xff = 0x{W20_padded[13] & 0xff:02x}, W14 = 0x{W20_padded[14]:08x}, W15 = {W20_padded[15]} bits (55 bytes) -> PASS")
    print(f"D=0 Invariant  : a5 = 0x{a5:08x}, a6 = 0x{a6:08x} (D = a6 - a5 = {(a6 - a5) & 0xffffffff}) -> PASS")
    print(f"Twin Invariant : a10 = 0x{a10:08x}, a11 = 0x{a11:08x} -> PASS")
    print(f"Ch Mask Check  : e9 = 0x{e9:08x}, e10 = 0x{e10:08x}, e15 = 0x{e15:08x} -> PASS")
    assert a5 == a6 == 0x67978bcd
    assert a10 == a11 == 0xa7b351d7
    assert e9 == e10 == e15 == 0xffffffff

    print("\n" + "=" * 78)
    print("2. VERIFYING 19-STEP PADDED SHA-256 PREIMAGE WITNESS (Target: SHA-256('abc'))")
    print("=" * 78)
    W19_padded = [
        0x8ca254f8, 0xd04dd3fb, 0x04efaad6, 0xb1867588,
        0xffe833de, 0x49a144fe, 0xdbf18394, 0x22892dc9,
        0xf5a9158d, 0x8fc06329, 0x0a14a9a6, 0xc55dfa24,
        0x5642a024, 0x7ec45a80, 0x00000000, 0x000001b8,
    ]
    target_abc = [
        0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
        0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
    ]
    digest19p, trace19p, _ = sha256_reduced(W19_padded, 19)
    print(f"Input W[0..15] : {' '.join(f'{w:08x}' for w in W19_padded)}")
    print(f"19-Step Digest : {fmt_words(digest19p)}")
    print(f"Target ('abc') : {fmt_words(target_abc)}")
    assert digest19p == target_abc, "19-step digest does not match SHA-256('abc')!"
    assert (W19_padded[13] & 0xff) == 0x80 and W19_padded[14] == 0 and W19_padded[15] == 0x1b8
    print("Padding & Digest Check -> PASS")

    print("\n" + "=" * 78)
    print("3. VERIFYING 20-STEP UNPADDED SHA-256 PREIMAGE WITNESS (Target: deadbeef...)")
    print("=" * 78)
    W20_unpadded = [
        0x4334d956, 0x6eccece1, 0x6727a1b2, 0xd85811af,
        0x890c5399, 0x79018a3b, 0xd346b94d, 0x5f26009b,
        0x7562d311, 0x0bfd80d5, 0x1f8c38f4, 0x16d8315d,
        0xe0353aa2, 0x6adf4bd2, 0xa2bf14a2, 0x801ae8f7,
    ]
    target_deadbeef = [
        0xdeadbeef, 0xcafebabe, 0x01234567, 0x89abcdef,
        0xfedcba98, 0x76543210, 0x13579bdf, 0x2468ace0,
    ]
    digest20u, _, _ = sha256_reduced(W20_unpadded, 20)
    print(f"Input W[0..15] : {' '.join(f'{w:08x}' for w in W20_unpadded)}")
    print(f"20-Step Digest : {fmt_words(digest20u)}")
    assert digest20u == target_deadbeef, "20-step unpadded digest mismatch!"
    print("20-Step Unpadded Digest Check -> PASS")

    print("\nALL THREE WITNESSES (20-STEP PADDED, 19-STEP PADDED, 20-STEP UNPADDED) VERIFIED 100%!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
