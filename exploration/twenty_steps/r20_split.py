"""
20-step split test (configuration F = the 19-step shaping applied at 20 steps).

  context : a5 free, a6 = a5 (D = 0), a7, a8 free, a9 from e9 = -1, a10 from e10 = -1, a11, a12 free
  chain   : a1..a4 (any values; the identity below must hold for all of them)
  target  : a13..a20 (random)

With c_k = W_k evaluated with a1..a4 = 0 and
  T16 = W16 - s1(W14) - c9,  T17 = W17 - s1(W15) - c10,  T18 = W18 - s1(W16) - c11,
  T19 = W19 - s1(W17) - W12                 (computed from context + target only)
the 19-step lemma predicts, for every chain,
  rho16 = T16 + a2 - R - s0(W1) - W0,   R = Maj(a5,a4,a3) - a4
  rho17 = T17 + a3     - s0(W2) - W1
  rho18 = T18 + a4     - s0(W3) - W2
  rho19 = T19          - s0(W4) - W3
so the chain side sees the context only through the interface (T16, T17, T18, T19) and a5, and the
20-step system is the 19-step triangular system plus one extra 32-bit equation.

Part 1 checks the identity numerically.
Part 2 measures, on the context side, how much one flipped bit of each free context word moves each
interface word (a6, a9, a10 re-derived after every flip).  A meet-in-the-middle on T19 would need
many contexts sharing (T16, T17, T18); sparse entries here would be the first sign of such structure.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from aseq import ASeq, M, S0, s0, s1, maj

R_STEPS = 20
rng = random.Random(20260929)


def hw(x):
    return bin(x).count('1')


def rho(s, t):
    W = s.W
    return (W(t) - s1(W(t - 2)) - W(t - 7) - s0(W(t - 15)) - W(t - 16)) & M


def derive_F(s):
    a = s.a
    a[6] = a[5]
    a[9] = (M + S0(a[8]) + maj(a[8], a[7], a[6]) - a[5]) & M      # e9  = -1
    a[10] = (M + S0(a[9]) + maj(a[9], a[8], a[7]) - a[6]) & M     # e10 = -1


def random_context():
    s = ASeq(R_STEPS)
    for t in range(1, R_STEPS + 1):
        s.a[t] = rng.getrandbits(32)
    derive_F(s)
    assert s.e(9) == M and s.e(10) == M
    return s


def interface(s):
    """(T16, T17, T18, T19) from context + target only (chain words zeroed on a copy)."""
    z = ASeq(R_STEPS)
    z.a = dict(s.a)
    for k in range(1, 5):
        z.a[k] = 0
    W = z.W
    c9, c10, c11 = W(9), W(10), W(11)
    return ((W(16) - s1(W(14)) - c9) & M,
            (W(17) - s1(W(15)) - c10) & M,
            (W(18) - s1(W(16)) - c11) & M,
            (W(19) - s1(W(17)) - W(12)) & M)


def check_identity(n=4000):
    bad = 0
    for _ in range(n):
        s = random_context()
        T16, T17, T18, T19 = interface(s)
        for k in range(1, 5):
            s.a[k] = rng.getrandbits(32)
        a, W = s.a, s.W
        R = (maj(a[5], a[4], a[3]) - a[4]) & M
        pred = {16: (T16 + a[2] - R - s0(W(1)) - W(0)) & M,
                17: (T17 + a[3] - s0(W(2)) - W(1)) & M,
                18: (T18 + a[4] - s0(W(3)) - W(2)) & M,
                19: (T19 - s0(W(4)) - W(3)) & M}
        bad += sum(pred[t] != rho(s, t) for t in range(16, 20))
    return bad


def context_edges(words=(5, 7, 8, 11, 12), trials=500):
    rows = {}
    for k in words:
        tot = [0, 0, 0, 0]
        zero_tau = 0          # flips leaving T16, T17, T18 all unchanged
        min_tau = 99
        for _ in range(trials):
            s = random_context()
            base = interface(s)
            s.a[k] ^= 1 << rng.randrange(32)
            derive_F(s)
            new = interface(s)
            d = [hw(x ^ y) for x, y in zip(base, new)]
            for i in range(4):
                tot[i] += d[i]
            zero_tau += (d[0] == d[1] == d[2] == 0)
            min_tau = min(min_tau, d[0] + d[1] + d[2])
        rows[k] = ([x / trials for x in tot], zero_tau, min_tau)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    n = 4000
    bad = check_identity(n)
    print(f'part 1: split identity, {n} random (context, chain) samples x 4 constraints: {bad} violations')
    print('part 2: context side, one flipped bit of a free word -> mean bits changed in the interface')
    print('   word    T16    T17    T18    T19   | flips with T16..T18 unchanged | min HW(dT16..dT18)')
    for k, (m, z, mn) in context_edges().items():
        name = 'a5(=a6)' if k == 5 else f'a{k}'
        print(f'   {name:<7} ' + '  '.join(f'{x:5.2f}' for x in m) + f'   | {z:4d} / 500' + f'                  | {mn}')
    print(f'({time.time() - t0:.1f} s)')
