"""
20-step dependency analysis (independent re-implementation, my indexing).

Constraint residuals  rho_t = W_t - s1(W_{t-2}) - W_{t-7} - s0(W_{t-15}) - W_{t-16},  t = 16..19.
Edge weight(a_k -> C_t) = mean Hamming weight of the change in rho_t when one random bit of a_k is
flipped, all other *free* words fixed and all condition-defined words re-derived so that every
imposed condition still holds.  ~16 bits = dense (random), 0 = absent, ~1-2 = sparse/absorbable.

Configurations
  A  4th pivot a5   : a6 = a7 = v,  e10 = e11 = -1           (derived: a7 := a6, a10, a11)
  B  4th pivot a6   : a8 = a7, e9 = -1, e12 = 0 (a6 then enters C19 linearly) (derived: a8, a9, a12)
  C  truncated 224  : a13 free (h output word dropped), measure a13 -> C16..C19
  D  truncated 128  : a13..a16 free (only a,b,c,d kept), measure each
  E  two-block CV   : vary the chaining-value word h (IV[7]) with the digest fixed (feed-forward
                      moves the target window), measure effect on C16..C19
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from aseq import ASeq, M, S0, S1, s0, s1, maj, IV, window_from_state

R = 20
rng = random.Random(20260928)


def rho(s, t):
    W = s.W
    return (W(t) - s1(W(t - 2)) - W(t - 7) - s0(W(t - 15)) - W(t - 16)) & M


def hw(x): return bin(x).count('1')


def derive_A(s):
    a = s.a
    a[7] = a[6]
    a[10] = (M - 0 + S0(a[9]) + maj(a[9], a[8], a[7]) - a[6]) & M          # e10 = -1
    a[11] = (M + S0(a[10]) + maj(a[10], a[9], a[8]) - a[7]) & M            # e11 = -1
    assert s.e(10) == M and s.e(11) == M


def derive_B(s):
    a = s.a
    a[8] = a[7]
    a[9] = (M + S0(a[8]) + maj(a[8], a[7], a[6]) - a[5]) & M               # e9 = -1
    a[12] = (0 + S0(a[11]) + maj(a[11], a[10], a[9]) - a[8]) & M           # e12 = 0
    assert s.e(9) == M and s.e(12) == 0


def fresh(free, derive=None):
    s = ASeq(R)
    for t in range(1, R + 1):
        s.a[t] = rng.getrandbits(32)
    if derive:
        derive(s)
    return s


def edge_table(words, derive=None, trials=400, extra_free=()):
    res = {}
    for k in words:
        tot = {t: 0 for t in range(16, 20)}
        for _ in range(trials):
            s = fresh(None, derive)
            base = {t: rho(s, t) for t in range(16, 20)}
            s.a[k] ^= 1 << rng.randrange(32)
            if derive:
                derive(s)
            for t in range(16, 20):
                tot[t] += hw(base[t] ^ rho(s, t))
        res[k] = {t: tot[t] / trials for t in range(16, 20)}
    return res


def show(title, res):
    print(title)
    print('   word   C16    C17    C18    C19')
    for k, d in res.items():
        print(f'   a{k:<3} ' + '  '.join(f'{d[t]:5.2f}' for t in range(16, 20)))


if __name__ == '__main__':
    show('A: pivots a2,a3,a4,a5; context a6=a7, e10=e11=-1', edge_table([2, 3, 4, 5], derive_A))
    show('B: pivots a2,a3,a4,a6; context a8=a7, e9=-1, e12=0', edge_table([2, 3, 4, 6], derive_B))
    show('no conditions (reference): late words', edge_table([7, 8, 9, 10, 11, 12]))
    show('C/D: truncated outputs -- freed target-side words (no conditions)', edge_table([13, 14, 15, 16]))


def config_E(trials=400):
    """Two-block setting: chaining value CV is the IV of the last block; digest D = CV + state.
    Flip one bit of CV[7] (register h) with D fixed: moves a_{-7} and, through feed-forward, a_13."""
    tot = {t: 0 for t in range(16, 20)}
    lin_w0 = 0
    for _ in range(trials):
        CV = [rng.getrandbits(32) for _ in range(8)]
        D = [rng.getrandbits(32) for _ in range(8)]
        frees = {t: rng.getrandbits(32) for t in range(1, 13)}

        def build(cv):
            s = ASeq(R)
            wiv = window_from_state(cv)
            for i in range(8):
                s.a[i - 7] = wiv[i]
            st = tuple((D[i] - cv[i]) & M for i in range(8))
            wt = window_from_state(st)
            for i in range(8):
                s.a[13 + i] = wt[i]
            for t, v in frees.items():
                s.a[t] = v
            return s
        s0_ = build(CV)
        CV2 = list(CV)
        CV2[7] ^= 1 << rng.randrange(32)
        s1_ = build(CV2)
        for t in range(16, 20):
            tot[t] += hw(rho(s0_, t) ^ rho(s1_, t))
        lin_w0 += hw(s0_.W(0) ^ s1_.W(0))
    print('E: two-block, flip one bit of chaining-value word h with digest fixed')
    print('   C16..C19 residual change (bits):', '  '.join(f'{tot[t]/trials:5.2f}' for t in range(16, 20)),
          f'| W0 change {lin_w0/trials:.2f} bits')


config_E()
