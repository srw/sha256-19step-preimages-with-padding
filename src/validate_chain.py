"""
Numerical validation of the conditioned-chain identities for r = 19 (and the padded variant).

Conditions on the 'context' (late/middle words):  e_9 = e_10 = 0xFFFFFFFF,  D = a_5 ^ a_6 (small).
Claims (for arbitrary early words a_1..a_4):
  X9  = W_9  + a_2 - R(a_3,a_4)      is constant,  R = Maj(a_5,a_4,a_3) - a_4
  X10 = W_10 + a_3 - (a_4 & D)       is constant
  X11 = W_11 + a_4                   is constant
  W_14 .. W_18                       are constant
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aseq import ASeq, M, S0, maj

rng = random.Random(2026)
R_ROUNDS = 19


def make_context(D=0):
    s = ASeq(R_ROUNDS)
    for t in range(12, 20):
        s.a[t] = rng.getrandbits(32)          # target window a_12..a_19
    v = rng.getrandbits(32)
    s.a[5], s.a[6] = v, v ^ D
    s.a[7], s.a[8], s.a[11] = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
    a = s.a
    a[9] = (M + S0(a[8]) + maj(a[8], a[7], a[6]) - a[5]) & M          # e_9 = -1
    a[10] = (M + S0(a[9]) + maj(a[9], a[8], a[7]) - a[6]) & M         # e_10 = -1
    assert s.e(9) == M and s.e(10) == M
    return s


def check(D, trials=2000):
    s = make_context(D)
    ref = None
    bad = 0
    for _ in range(trials):
        for t in (1, 2, 3, 4):
            s.a[t] = rng.getrandbits(32)
        a = s.a
        R = (maj(a[5], a[4], a[3]) - a[4]) & M
        vals = ((s.W(9) + a[2] - R) & M,
                (s.W(10) + a[3] - (a[4] & D)) & M,
                (s.W(11) + a[4]) & M,
                tuple(s.W(t) for t in range(14, 19)))
        if ref is None:
            ref = vals
        bad += vals != ref
    return bad


if __name__ == '__main__':
    for D in (0, 0x00000001, 0x80000400, 0x0F0F0000):
        print(f'D={D:#010x}: identity violations in 2000 random early assignments: {check(D)}')
    # sparsity of the residual: P(R == 0) and P(a4 & D == 0) for random a3, a4
    n, hitR = 200000, 0
    v = rng.getrandbits(32)
    for _ in range(n):
        a3, a4 = rng.getrandbits(32), rng.getrandbits(32)
        hitR += ((maj(v, a4, a3) - a4) & M) == 0
    print(f'P(R=0) empirical {hitR/n:.2e}  vs (3/4)^32 = {0.75**32:.2e}')
