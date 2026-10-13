"""For every padded witness (archived + audit-fresh replay + my fresh run) compute, with my own SHA-256 step function,
a5, a6, D = a5^a6, k = popcount(D), o = popcount(a5 & D); then test the audit's exact joint-probability formula
P ∝ (3/4)^(32-k) (1/2)^(k-o) (1/4)^o  against the observed distribution of o among successful contexts."""
import sys, re, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_sha import K, H0, ror, M, pad
from collections import Counter

def a_seq(block, steps=19):
    W = [int.from_bytes(block[4*i:4*i+4], 'big') for i in range(16)]
    for t in range(16, steps):
        s0 = ror(W[t-15], 7) ^ ror(W[t-15], 18) ^ (W[t-15] >> 3)
        s1 = ror(W[t-2], 17) ^ ror(W[t-2], 19) ^ (W[t-2] >> 10)
        W.append((W[t-16] + s0 + W[t-7] + s1) & M)
    a, b, c, d, e, f, g, h = H0
    A = [a]; E = [e]
    for t in range(steps):
        T1 = (h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g & M)) + K[t] + W[t]) & M
        T2 = ((ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & M, c, b, a, (T1 + T2) & M
        A.append(a); E.append(e)
    return A, E, W

WIT, REPLAY, MINE = sys.argv[1], sys.argv[2], sys.argv[3]     # witnesses dir, replay .out, new-labels .out
msgs = {}
# archived padded witnesses
for fn in ['witnesses_19step.txt', 'demos_19step.txt', 'examples_abc_hello.txt', 'witnesses_fresh_2026-09-29.txt']:
    t = open(f'{WIT}/{fn}').read()
    for chunk in re.split(r'(?=^\[)', t, flags=re.M):
        m = re.search(r'message_hex\s*=\s*([0-9a-f]+)', chunk)
        if m and len(m.group(1)) == 110:                    # 55-byte, single-block
            msgs[m.group(1)] = 'archived'
# fresh runs (audit replay == my Linux replay, and my own 10 labels)
for fn, tag in [(REPLAY, 'audit-fresh'), (MINE, 'mine-fresh')]:
    t = open(fn).read()
    for w in re.findall(r'^W =((?: [0-9a-f]{8}){16})$', t, flags=re.M):
        blk = bytes.fromhex(w.replace(' ', ''))
        msgs.setdefault(blk[:55].hex(), tag)
print("distinct padded 55-byte witnesses:", len(msgs), Counter(msgs.values()))
rows = []
for hx, tag in msgs.items():
    msg = bytes.fromhex(hx); blk = pad(msg)
    A, E, W = a_seq(blk)
    a5, a6 = A[5], A[6]
    assert E[9] == M and E[10] == M and (W[13] & 255) == 0x80 and W[14] == 0 and W[15] == 440     # structure holds
    D = a5 ^ a6; k = bin(D).count('1'); o = bin(a5 & D).count('1')
    a4 = A[4]; a3 = A[3]
    assert (a4 & D) == 0
    rows.append((tag, k, o))
print("k distribution (popcount D):", sorted(Counter(r[1] for r in rows).items()))
for kk in (3, 2, 1, 0):
    sub = [r[2] for r in rows if r[1] == kk]
    if not sub: continue
    # prediction for o given success, with prior Binomial(k,1/2) and success weight (1/2)^(k-o) (1/4)^o
    w = [math.comb(kk, o) * 0.5 ** kk * 0.5 ** (kk - o) * 0.25 ** o for o in range(kk + 1)]
    s = sum(w); pred_mean = sum(o * x for o, x in enumerate(w)) / s
    print(f"k={kk}: n={len(sub)}  observed mean o={sum(sub)/len(sub):.2f}  predicted given success={pred_mean:.2f}  "
          f"(unconditional prior mean = {kk/2:.2f});  observed o histogram={sorted(Counter(sub).items())}")
sub = [r[2] for r in rows if r[1] == 3]
n = len(sub); mu0 = 1.5; sd0 = math.sqrt(3 * 0.25)         # Binomial(3,1/2): mean 1.5, var .75
z = (sum(sub) / n - mu0) / (sd0 / math.sqrt(n))
print(f"z-score of observed mean(o|k=3) against the unconditional prior mean 1.5: {z:.2f}")
