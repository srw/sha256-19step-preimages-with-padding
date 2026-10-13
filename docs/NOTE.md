# Padded Preimages for 19-Step SHA-256 on One CPU Core

*Technical note, 2026-09-28; revised 2026-10-13 after an independent audit (see
`checks/recheck_2026-10-13/`). The results below were executed and checked by a second
implementation, by an independent agent that re-derived the method from this text, by an audit that
wrote its own SHA-256, and by a third implementation. These are computational cross-checks by AI
agents, not human peer review. Nothing here affects full 64-step SHA-256.*

## Summary

We give **preimages for the SHA-256 hash function reduced to its first 19 of 64 steps** (H19). The
setting is:
- standard IV, standard padding, steps 0–18 on every block, and the usual feed-forward;
- real byte strings: 55-byte single-block messages, and a two-block message with a chosen
  readable first block;
- prescribed 256-bit targets: the full SHA-256 digests of public labels, and the all-ones and
  all-zeros digests.

The solver succeeded on every target we recorded (37 distinct padded solutions). Search time on one
CPU core, with the one-time table build excluded, had medians of 7 to 22 s depending on the CPU
(overall range 0.3–86 s across the datasets listed in the README, Results). The table takes 8 to
28 s to build and about 2.1 GiB. The same code finds preimages of the 19-step compression function in about 1–3 s.
Success for every possible target is not proved.

**What is new here.** Previously demonstrated practical 19- and 20-step preimages are for the
compression function only: 16 message words, no padding. Basra's manuscript says of his
construction: "Although the repository contains an exploratory closed-form way to impose those word
values, no padded preimage was obtained; we make no padded-hash claim." To our knowledge, as of
2026-10-13, these are the first publicly documented full 256-bit preimages of SHA-256 reduced to its
first 19 steps, with the standard IV and padding. That is a statement about what we could find, not a
proof of priority (§7 lists what was and was not searched).

**What is not new, with credit.** The key algebraic step is to choose internal "context" words so
that two Ch selectors are saturated, which makes the three schedule constraints triangular. It was
published first by K. S. Basra, in the manuscript *Context Shaping for 19- and 20-Round SHA-256
Compression Preimages* and the repository `kamb-code/sha256-r19-preimage` (the commit that introduced
it is dated 2026-09-05). We re-derived it independently here, but priority is his. The closed-form
absorption of the length words was also his observation, made earlier for his unshaped construction
(`code/padded_preimage.py`, commit `c97f093`, 2026-08-13; a 167-context run produced no witness; that
history is no longer in the public repository, see the README). Our addition is how to make context
shaping coexist with padding once the third padding word can no longer be pinned (§5): the 8-bit
filter on W_13, the relaxed shaping condition, and the demonstration.

**Demonstrations.** The full list, 13 padded and 11 compression-function witnesses, is in
`witnesses/witnesses_19step.txt` and `witnesses/demos_19step.txt`; counting all four witness files
there are 28 records and 27 distinct witnesses (15 padded, 12 compression).
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

**Table.** Store Q(u) for the 2^m distinct values u = i·0x9E3779B1 (mod 2^32), i < 2^m, bucketed by
the top 24 bits of Q(u). We use m = 29. The table is 2^m 32-bit words (2 GiB) plus 2^24 bucket
offsets (64 MiB), and the build uses a further temporary 64 MiB; it took 8.5 to 28 s depending on
the CPU. A lookup returns at most eight roots (a fixed cap in the code). A random target has about
2^{m−32} = 1/8 matching entries on average. Each context gets at most 2^26 chain starts; if none
succeeds, a new context is drawn.

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

**Cost.** The expected number of chain starts is about 2^{13.3 + 3(32−m) + popcount(D)} on average
over contexts (see §5 for what "on average" hides). For m = 29 and D = 0 that is 2^22.3. Measured over
50 compression-function solutions (10 here, 40 by the verifier), the mean was 4.7·10^6 ≈ 2^22.2, a
few seconds each. A full inverse table (m = 32, 16 GiB) lowers this to about 2^13.3, which is
Basra's regime. The model is a heuristic: it treats the table lookups and the accepted contexts as
independent and uniform, which we have not proved. A chain start is not one SHA-256 evaluation, so
chain starts alone are not the total work; the context search is counted separately (§5).

## 5. Padding

For a message whose last block holds 55 bytes, padding fixes three words:
- W_13 ∧ 0xFF = 0x80;
- (W_14, W_15) = the 64-bit bit-length.

Two of these absorb in closed form:
- W_15 is linear in a_8 (its oldest window word), so a_8 is fixed by a_9..a_11;
- W_14 is linear in a_7, so a_7 is fixed likewise.

This is Basra's observation (`padded_preimage.py`, commit `c97f093`, 2026-08-13, for his unshaped
construction, where it pins W_13 as well, through the context word that is a_6 in our indexing; see
the README, Credit).

The difficulty is that shaping also wants a_5 = a_6 and e_9 = e_10 = −1, and the words are no
longer free enough for all of it: a_6 is spent on e_10 = −1, so W_13 can no longer be pinned. Our
resolution has three parts:
1. **Use W_13 as a filter, not a pin.** With e_10 = −1 imposed, W_13 no longer depends on a_6. Its
   low-byte condition becomes an 8-bit filter on the context.
2. **Let the saturation conditions set a_6 and a_5.** e_10 = −1 then fixes a_6, and e_9 = −1 fixes
   a_5.
3. **Soft shaping.** D = a_5 ⊕ a_6 is now forced, so we accept contexts with popcount(D) ≤ 3 and
   require a_4 ∧ D = 0 in the chain.

Imposing a_5 = a_6 exactly is not impossible (the code accepts `kmax = 0`), only expensive: P(D = 0)
is 2^−32 per context trial, against 5489·2^−32 ≈ 2^−19.6 for popcount(D) ≤ 3, and it would buy no
saving.

**Exact cost of soft shaping.** Put v = a_5, k = popcount(D) and o = popcount(v ∧ D). For independent
uniform bits of a_3 and a_4, the condition R = 0, that is Maj(v, a_4, a_3) = a_4, holds on one bit
with probability 3/4. On a bit where D = 1 we also need a_4 = 0, and the joint probability is 1/2
if v = 0 and 1/4 if v = 1. So for a fixed context

    Pr[R = 0 and a_4 ∧ D = 0] = (3/4)^{32−k} · (1/2)^{k−o} · (1/4)^o.

Averaging over fair bits of v on D gives (3/4)^32 · 2^{−k}, which is the "factor 2^{popcount D}" of
the earlier version of this note. That is right on average but not for each context: at k = 3 the
exact value differs from it by a factor between 8/27 and 64/27. The audit checked the formula by
exhaustive enumeration at 6 bits. In the data it shows as a bias: the solver's successful contexts
with k = 3 (30 of 36 distinct padded 55-byte witnesses; the other 6 have k = 2) have mean o = 1.00, as
the formula predicts for successes, and not the 1.50 that an unconditional Binomial(3, 1/2) would
give (z = −3.2; `checks/recheck_2026-10-13/results.txt`).

The context search costs about 2^8 · 2^19.6 cheap trials per usable context (the 8-bit filter and
the popcount test), and the chain about 2^{22.3 + popcount D} chain starts on average. Measured over
32 label targets: mean 2^25.3 chain starts and 2^27.9 context trials per solution (README, Results);
2^27.9 divided by the mean 1.31 contexts used is 2^27.5 trials per context, close to the 2^27.6
predicted. Measured times are in the README. The chaining value and message length are parameters,
which is how the two-block demonstration was made.

**Correctness of an output.** Whatever the search does, an output is a valid preimage if and only if
recomputing H19 gives the target, and every output reported here was recomputed with at least one
independent SHA-256. The deterministic part of the argument is this. (i) The context is built so
that e_9 = e_10 = −1 and W_13's low byte is 0x80, W_14 and W_15 equal the length words (the code
checks the last three before accepting a block). (ii) Given such a context, the three table
inversions make C16, C17 and C18 hold exactly, by the identities of §2 and §3. (iii) The final
check R = 0 and a_4 ∧ D = 0 is what the lemma of §3 needs for W_9, W_10 and W_11 to take the stated
forms. (iv) A block that satisfies C16–C18 and the three padding conditions is a valid padded
message. The solver also recomputes H19 and the padding conditions before printing `ok=1`. The
heuristic part is only how long the search takes.

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

## 7. Prior work (dated 2026-10-13; practical line, then the theoretical line)

| steps | object | who | method and cost |
|---|---|---|---|
| 16 | hash function, with padding | Homsirikamol, Morawiecki, Rogawski, Srebrny, CISIM 2012 (ePrint 2012/421) | SAT (CryptoMiniSat 2.9.0), about one hour on a 2.67 GHz Core i7; a 30-hour limit did not reach a 17th round. We read the ePrint version ("Extension", 20 pages) on 2026-10-13. Section 2.2 fixes the hash and the padding bits in a single 512-bit block with the message length set to 446 bits; Table 1 gives SHA-256 at 16 of 64 rounds; Table 3 lists the compression function separately (also 16). The paper does not say how its target digests were chosen and prints no message, so the result cannot be re-checked from the paper. Its message is a 446-bit string, not a whole number of bytes. |
| 17–18 | compression function, all-zero digest | Davydov et al. 2025 (abstract read 2026-10-13: "17- and 18-round SHA-256 compression functions", plus a weakened 19-round problem) | SAT |
| 19 | compression function | Zaikin, *Constraints*, Jan 2026 (preprint read; padding removed; Table 7 witness re-verified) | cube-and-conquer SAT, one all-ones target, 18 h 33 min on 192 cores |
| 19 | compression function | Basra, 2026 (unpublished; repository) | algebraic; GPU; then context shaping, ~12 ms marginal with a 16 GiB table |
| 20 | compression function | Basra, Sept 2026 (unpublished; all-ones witness re-verified) | context shaping, ~2^45.4, 2.45 and 16.2 GPU-hours |
| **19** | **hash function with padding** | **this note** | **context shaping plus soft shaping plus W_13 filter; medians of 7–22 s on one CPU core** |

An unverified claim also exists of padded preimages at 17 rounds (McDair, *cryptography* mailing
list, April 2024; the post names a target but gives no witness, and the replies dispute terminology
rather than showing a concrete mismatch). It is below 19 steps and we have not validated it.

The theoretical line (meet-in-the-middle and bicliques) reaches 46 and 47 steps (Guo, Li, Liu, Wang
and Zhang, EUROCRYPT 2026, ePrint 2026/353). Its Table 1 gives, for 46 steps, a pseudo-preimage cost of
2^251.1 and a preimage cost of 2^254.5, and for 47 steps 2^253.2 and 2^255.6. Its 44-step attack
accounts for the padding (65 degrees of freedom from W_14, W_15 and one bit of W_13), and the authors
implemented that attack's procedure with reduced matching bits to check the complexity (their
repository `desert-oasis/dual-sync-MITM`). So the line has been partly run, but it has not produced a
demonstrated full-output preimage, padded or not. The previous record was 45 steps (Khovratovich,
Rechberger and Savelieva, FSE 2012).

What was searched for earlier work: web search, arXiv, ePrint, GitHub, and the references of the
papers above. What was not: CNKI, Wanfang, CiNii, RISS, ProQuest and Russian thesis databases were
not accessible. "To our knowledge" in the Summary is a statement about that search, not a proof.

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
- Search times are the solver's own output (C `clock()`, process CPU seconds, rounded to 0.1 s), on
  machines that were not isolated. The independent verifier of 2026-09-29 reproduced them within a
  factor of about 2, including 5 padded preimages of its own from this description. The datasets of
  2026-10-13 are listed in the README, with the raw outputs in `checks/recheck_2026-10-13/timing/`.
- Second preimages for H19 follow by setting the target to H19(M') for any message M' and searching
  for a different 55-byte message. That is a reduction, not a separate experiment, and it says
  nothing about SHA-256 itself. Collisions are not claimed.
