# Padded Preimages for 19-Step SHA-256 on One CPU Core

*Technical note, 2026-09-28 (revised after independent verification). Every result below was
executed and checked by a second implementation, and by an independent agent that re-derived the
method from this text. Nothing here affects full 64-step SHA-256.*

## Summary

We give **preimages for the SHA-256 hash function reduced to 19 of 64 steps**. The setting is:
- standard IV, feed-forward and padding;
- real byte strings: 55-byte single-block messages, and a two-block message with a chosen
  readable first block;
- arbitrary 256-bit targets, all taken as full SHA-256 digests of public labels.

Search takes **1–74 s per target (median 19 s over 12 timed targets) on one CPU core**, after a one-time
18-second, 2 GB table. The same code finds preimages of the 19-step compression function in about
1–3 s.

**What is new here.** Previously demonstrated practical 19- and 20-step preimages are for the
compression function only: 16 message words, no padding. The author of the algebraic line of work
states explicitly that "no padded SHA-256 preimage is claimed", and an earlier 167-context padded
attempt produced none. As far as we can find, these are the first demonstrated preimages of
step-reduced SHA-256 as a hash function at 19 steps.

**What is not new, with credit.** The key algebraic step is to choose internal "context" words so
that two Ch selectors are saturated, which makes the three schedule constraints triangular. It was
published first by K. S. Basra, in the manuscript *Context Shaping for 19- and 20-Round SHA-256
Compression Preimages* and the repository `kamb-code/sha256-r19-preimage`, from about
2026-09-05. We re-derived it independently here, but priority is theirs. Our addition is how to
make it coexist with padding (§5).

**Demonstrations.** The full list, 13 padded and 11 compression-function witnesses, is in
`witnesses/witnesses_19step.txt` and `witnesses/demos_19step.txt`.
- The 55-byte message

  ```
  8ca254f8d04dd3fb04efaad6b1867588ffe833de49a144fedbf1839422892dc9f5a9158d8fc063290a14a9a6c55dfa245642a0247ec45a
  ```

  has 19-step SHA-256 `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`, which
  is the FIPS 180 test-vector digest of `"abc"`.
- The 119-byte message `This first block is plain text; the second one was computed.....`
  followed by 55 computed bytes, `fe409cc930128a72…1c95c0`, has 19-step SHA-256
  `e3b0c442…7852b855`, which is SHA-256 of the empty string.

## 1. Notation: the a-sequence

Round t+1 consumes W_t and produces a_{t+1} and e_{t+1}. Write

    e_s = a_s − Σ0(a_{s−1}) − Maj(a_{s−1}, a_{s−2}, a_{s−3}) + a_{s−4}                   (mod 2^32)
    W_t = a_{t+1} − Σ0(a_t) − Maj(a_t, a_{t−1}, a_{t−2}) − e_{t−3} − Σ1(e_t) − Ch(e_t, e_{t−1}, e_{t−2}) − K_t

These follow from a_{t+1} = T1 + T2 and e_{t+1} = a_{t−3} + T1.
- The state after r steps is equivalent to the window a_{r−7}..a_r.
- The chaining value fixes a_{−7}..a_0.
- The target D fixes the state D − CV, and therefore a_{r−7}..a_r.

For r = 19 the unknowns are a_1..a_11. W_t depends only on a_{t−7}..a_{t+1}, and a preimage is
exactly a solution of the three schedule constraints

    C_t:  W_t = σ1(W_{t−2}) + W_{t−7} + σ0(W_{t−15}) + W_{t−16},     t = 16, 17, 18.

## 2. Pivots (the Q-cancellation)

In C_t the word a_{t−14} appears in two places:
- linearly in W_{t−15} = a_{t−14} − G_{t−15}(a_1..a_{t−15});
- as the oldest element of W_{t−7}'s window, with coefficient −1 and nowhere else.

Hence

    Q(W_{t−15}) = RHS_t,     Q(u) := σ0(u) − u.

So C16, C17 and C18 can be solved in turn for the pivots a_2, a_3 and a_4 by inverting the fixed
map Q. The image of Q covers about 63 % of Z/2^32. Basra's first paper calls these "cancellation
identities".

## 3. Selector saturation (Basra's context shaping, restated in this indexing)

**Lemma.** Choose the context words a_5..a_11 so that e_9 = e_10 = 0xFFFFFFFF, and put
D = a_5 ⊕ a_6. Then for all a_1..a_4:

    W_9  = −a_2 + c_9  + [Maj(a_5, a_4, a_3) − a_4]
    W_10 = −a_3 + c_10 + (a_4 ∧ D)
    W_11 = −a_4 + c_11

Here c_9, c_10 and c_11 depend only on a_5..a_12, and W_14..W_18 do not depend on a_1..a_4.

*Proof.*
- **W_9.** e_9 = −1 makes Σ1(e_9) constant and Ch(e_9, e_8, e_7) = e_8. The early words then enter
  only through

      −e_6 − e_8,  where e_6 = a_6 − Σ0(a_5) − Maj(a_5, a_4, a_3) + a_2  and  e_8 = a_8 − Σ0(a_7) − Maj(a_7, a_6, a_5) + a_4.

- **W_10.** e_10 = −1 makes Σ1(e_10) constant and Ch(e_10, e_9, e_8) = e_9 = −1. The early words
  enter only through −e_7, with e_7 = a_7 − Σ0(a_6) − Maj(a_6, a_5, a_4) + a_3, and

      Maj(a_6, a_5, a_4) = (a_5 ∧ a_6) + (a_4 ∧ D).

- **W_11.** Ch(e_11, −1, −1) = −1.
- **W_14..W_18.** Their windows start at a_7. ∎

Basra's version is a_4 = a_5 with e_8 = e_9 = −1 in his indexing, which is shifted by one. The
residual R = Maj(a_5, a_4, a_3) − a_4 is zero exactly when no bit position has
(a_5, a_3, a_4) ∈ {(1,1,0), (0,0,1)}. That happens with probability (3/4)^32 ≈ 2^−13.3; the
verifier measured 1.03·10^−4. Checks: 0 violations of the lemma over 8,000 random assignments here,
and 8,000 more by the independent verifier.

## 4. Algorithm and cost

**Table.** Store Q(u) for 2^m pseudo-random u, bucketed by the top 24 bits. We use m = 29, which
takes 2 GB and 18 s to build. A random target has about 2^{m−32} = 1/8 matching entries on
average.

**Constants per context.**
- c_9 = W_9, c_10 = W_10 and c_11 = W_11, each evaluated with a_1..a_4 = 0.
- T16 = W16 − σ1(W14) − c_9
- T17 = W17 − σ1(W15) − c_10
- T18 = W18 − σ1(W16) − c_11

**Chain.** Take successive a_1 and branch over every table hit:
1. W_1 ∈ Q^{-1}(T16 + G_1 − W_0), then a_2 = W_1 + G_1.
2. W_2 ∈ Q^{-1}(T17 + G_2 − W_1), then a_3 = W_2 + G_2.
3. W_3 ∈ Q^{-1}(T18 + G_3 − W_2), then a_4 = W_3 + G_3.
4. Accept if R = 0 and a_4 ∧ D = 0.

Here G_k = −W_k evaluated with a_{k+1} = 0.

**Cost.** The expected number of chain starts is about 2^{13.3 + 3(32−m) + popcount(D)}. For m = 29
and D = 0 that is 2^22.3. Measured over 50 compression-function solutions (10 here, 40 by the
verifier), the mean was 4.7·10^6 ≈ 2^22.2, a few seconds each. A full inverse table (m = 32,
16 GiB) lowers this to about 2^13.3, which is Basra's regime.

## 5. Padding

For a message whose last block holds 55 bytes, padding fixes three words:
- W_13 ∧ 0xFF = 0x80;
- (W_14, W_15) = the 64-bit bit-length.

Two of these absorb in closed form:
- W_15 is linear in a_8 (its oldest window word), so a_8 is fixed by a_9..a_11;
- W_14 is linear in a_7, so a_7 is fixed likewise.

This is the same observation as Basra's `padded_preimage.py`.

The difficulty is that shaping also wants a_5 = a_6 and e_9 = e_10 = −1, and the words are no
longer free enough for all of it. Our resolution has three parts:
1. **Use W_13 as a filter, not a pin.** With e_10 = −1 imposed, W_13 no longer depends on a_6. Its
   low-byte condition becomes an 8-bit filter on the context.
2. **Let the saturation conditions set a_6 and a_5.** e_10 = −1 then fixes a_6, and e_9 = −1 fixes
   a_5.
3. **Soft shaping.** D = a_5 ⊕ a_6 is now forced, so we accept contexts with popcount(D) ≤ 3 and
   guess a_4 ∧ D = 0 in the chain. That costs a factor 2^{popcount D}.

The context search costs about 2^8 · 2^19.6 cheap trials, and the chain about 2^{22.3 + popcount D}.
Measured over 12 timed padded targets: 1–74 s, median 19 s. The chaining value and message length are
parameters, which is how the two-block demonstration was made.

## 6. Twenty steps

At 20 steps the chain needs a fourth pivot, a_5. W_9 in C16 contains a_5 densely: through +Σ0(a_5)
in e_6, and inside e_9, which feeds Σ1(e_9) and a Ch selector. So one exact 32-bit condition
remains.

Basra reports this independently. He models the residual as 2^−32, computes a mean cost of about
2^45.4 lookups (about 15 A100-hours), and has found two 20-step compression-function preimages.
Padded 20-step preimages would add the soft-shaping and filter costs to that, and are beyond this
CPU-only setting. Reducing the 20-step residual below 2^32 is open. Basra's notes exclude many
routes within this representation. Our own preliminary checks of three further angles also found
no route:
- freedom in the chaining value of a two-block message;
- output truncation;
- context variations that leave the chain unchanged.

## 7. Prior work (dated; practical line only)

| steps | object | who | method and cost |
|---|---|---|---|
| 16 | compression function | Homsirikamol et al. 2012 (as cited by Basra; not checked) | CDCL SAT |
| 17–18 | compression function, all-zero digest | Davydov et al. 2025 (as cited by Basra; not checked) | SAT |
| 19 | compression function | Zaikin, *Constraints*, Jan 2026 | cube-and-conquer SAT, 18.5 h on 192 cores |
| 19 | compression function | Basra, 2026 (unpublished; repository) | algebraic; GPU; then context shaping, ~12 ms marginal with a 16 GiB table |
| 20 | compression function | Basra, Sept 2026 (unpublished) | context shaping, ~2^45.4, 2.45 and 16.2 GPU-hours |
| **19** | **hash function with padding** | **this note** | **context shaping plus soft shaping plus W_13 filter; about 20 s on one CPU core** |

The theoretical line (meet-in-the-middle and bicliques) now reaches 47 steps, at complexities
close to 2^256: Guo, Li, Liu, Wang and Zhang, EUROCRYPT 2026 (ePrint 2026/353). The previous
record was 45 steps (Khovratovich, Rechberger and Savelieva, FSE 2012).

## 8. Reproduce

```
gcc -O3 -march=native -o pre19 src/pre19.c
# padded, 55-byte messages:
./pre19 padded <target_hex> 29 <seed> 3 26
# two-block, chosen first block:
./pre19 padded <target_hex> 29 <seed> 3 26 iv=<chaining value after block 1> len=119
# compression function:
./pre19 comp <target_hex> 29 <seed> 0 26
python3 verify/verify_standalone.py witnesses/witnesses_19step.txt    # stdlib only; also checks 64 steps = hashlib
```

**Scope.**
- The padded mode assumes the final block carries 55 message bytes, i.e. total length ≡ 55 mod 64.
  Other lengths change the constraints: 52–54 bytes make the W_13 condition 32, 24 or 16 bits, and
  ≤ 51 bytes pins further words.
- Search times are self-reported by the solver. The independent verifier reproduced them within a
  factor of about 2, including 5 padded preimages of its own from this description.
