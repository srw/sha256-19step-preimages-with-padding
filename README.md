# Preimages for 19-step SHA-256 with standard padding

This repository gives working preimages for **SHA-256 reduced to its first 19 of 64 steps, as a hash
function** (written H19 below): standard initial value, standard padding, only steps 0–18 on every
block, and the usual feed-forward, with all 256 output bits matching a prescribed target.

Given a target digest, the solver searches for a 55-byte message M with H19(M) = target. It succeeded
on every target we recorded (37 distinct padded solutions; see Results). On one core a search took
from under a second to about a minute and a half in our runs. Median values were 7 to 22 s depending
on the CPU, after a one-time table build of 8 to 28 s and about 2.1 GiB of RAM. We have no proof that
the search succeeds for every target.

**This builds directly on K. S. Basra's "context shaping"** for the 19- and 20-round SHA-256
*compression function* ([kamb-code/sha256-r19-preimage](https://github.com/kamb-code/sha256-r19-preimage)).
The key algebraic step is his. What is added here is a way to make it coexist with SHA-256's
message padding, so that the preimages are real messages rather than free 512-bit blocks. (Basra had
earlier noted, for his unshaped construction, that the padding words can be pinned in closed form;
see Credit.)

Nothing here affects full 64-step SHA-256.

## Example

```
target  = SHA-256("abc")
        = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

message = 8ca254f8d04dd3fb04efaad6b1867588ffe833de49a144fedbf1839422892dc9f5a9158d8fc063290a14a9a6c55dfa245642a0247ec45a
          (55 bytes)

H19(message) = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

The target here is the ordinary 64-step SHA-256 digest of "abc". So the message is a preimage of that
digest under H19. It is not an H19 collision with "abc", because H19("abc") is a different value.

## What is new, and what is not

| steps | object | work |
|---|---|---|
| 16 | hash function, with padding (one block, 446-bit message) | Homsirikamol, Morawiecki, Rogawski, Srebrny, CISIM 2012, ePrint 2012/421 (SAT, CryptoMiniSat 2, about an hour on a 2.67 GHz Core i7). We read the ePrint version on 2026-10-13; it does not say how its targets were chosen and prints no message. |
| 17 | hash function, with padding (messages of 31, 55 and 55 bytes) | McDair, *cryptography* mailing list, 28 March 2024 (not peer-reviewed; no method described; we recomputed all three at 17 steps) |
| 17–18 | compression function, all-zero digest | Davydov et al., 2025 (SAT) |
| 19 | compression function | Zaikin, *Constraints* 2026 (SAT, one all-ones target, 18 h 33 min on 192 cores); Basra 2026 (algebraic) |
| 20 | compression function | Basra 2026 (GPU, about 2^45.4 table lookups); [srw/sha256-20step-preimages-with-no-padding](https://github.com/srw/sha256-20step-preimages-with-no-padding) (CPU, ~2^33 via Zero-a3 Bi-Directional Q0/Q1 Bridge, 21.1 min on 16 vCPUs) |
| **19** | **hash function, with padding** | **this repository** |

To our knowledge, as of 2026-10-13, this is the first publicly documented computation of full
256-bit preimages for SHA-256 reduced to its first 19 steps with the standard IV and padding. That is
a statement about what we could find, not a proof of priority. Our searches (general web search and
the reference lists of the papers above) did not cover CNKI, Wanfang, CiNii, RISS, ProQuest or
Russian thesis databases, which we could not access. We also know of three padded 17-step
preimages posted to the *cryptography* mailing list by McDair on 28 March 2024 ("SHA-256 challenge",
`2024-March/038315.html`): one message of 31 bytes and two of 55 bytes, all for the digest
`deb360ae...6b85`. We recomputed them with our own SHA-256 (`checks/recheck_2026-10-13/results_mcdair.txt`):
each equals that digest at exactly 17 steps, with the standard IV and padding. The post describes no
method. It also makes claims at 64 rounds about the compression function without padding, which we
did not evaluate. The only peer-reviewed padded result we know of is at 16 steps (Homsirikamol et al.),
and we know of no padded result above 17 steps. If you know of earlier work, please open an issue
and we will correct this.

The theoretical meet-in-the-middle line reaches 46 and 47 steps (Guo, Li, Liu, Wang, Zhang,
EUROCRYPT 2026; preimage cost about 2^254.5 and 2^255.6). Its authors implemented a 44-step version
with reduced matching bits to validate the complexity. No full-output preimage has been computed with
that line, padded or not.

## Verify (Python 3, standard library only)

```bash
python3 verify/verify_standalone.py witnesses/witnesses_19step.txt
```

The verifier first checks itself against `hashlib` at 64 steps. It then recomputes every witness
and checks that each target is the full SHA-256 of the witness's public label.

A second, independently written checker also confirms that each witness fails at 18 and 20 steps,
which checks the round-count convention:

```bash
cd verify/independent_checker && python3 v1.py ../../witnesses/witnesses_19step.txt
```

The best check is to write your own reduced-step SHA-256: run the standard compression loop for
19 rounds instead of 64, keep the feed-forward, and apply it to every block. A third, from-scratch
implementation (constants derived from integer roots of primes, not copied) is in
`checks/recheck_2026-10-13/scripts/my_sha.py`; it agrees with `hashlib` at 64 steps on 205 messages,
and `my_witness_check.py` uses it to count and verify the archived witnesses.

## Reproduce

Build the solver:

```bash
gcc -O3 -march=native -o pre19 src/pre19.c
```

Run it with the target given as 64 hex characters, here SHA-256 of a label of your choice. The
table takes ~2.1 GiB of RAM at size 2^29.

```bash
T=$(python3 -c "import hashlib;print(hashlib.sha256(b'my label').hexdigest())")
./pre19 padded $T 29 1 3 26     # 55-byte message, standard padding
./pre19 comp   $T 29 1 0 26     # compression function, 16 free words
```

The printed `W = ...` words form the final block. In padded mode the message is its first 55
bytes; the block ends in `...80 00000000 000001b8`.

Usage:

```
pre19 comp|padded <target>[,<target>...] <table_bits> <seed> [kmax] [log2chains] [iv=<64 hex>] [len=<bytes>]
```

- `iv=` sets the chaining value and `len=` the total message length.
- Together they allow a chosen first block. For example, `witnesses/demos_19step.txt` has a
  119-byte message whose first 64 bytes are readable text.
- `iv=` and `len=` are read only from the 7th argument on, so give `kmax` and `log2chains` first.

## Results

**Padded mode, search time** (CPU seconds as printed by the solver, one thread, table build excluded;
medians vary with the CPU, so each dataset is listed separately):

| dataset | targets | median | range |
|---|---|---|---|
| public logs of 2026-09-28/29, the author's machine (`witnesses/raw_solver_logs/`) | 13 (12 single-block, 1 two-block) | 21.7 s | 1.2–86.4 s |
| independent audit, 2026-10-13, Apple M3, 10 fresh labels | 10 | 6.95 s (mean 11.0 s) | 0.6–28.0 s |
| same audit, all-ones and all-zeros targets | 2 | – | 0.3 s and 1.0 s |
| recheck, 2026-10-13, Intel Xeon 2.1 GHz, two jobs running at once; replay of the audit's 12 targets | 12 | 10.5 s | 0.7–63.0 s |
| same machine, 10 new labels with a fresh random nonce | 10 | 14.3 s (mean 17.8 s) | 1.6–52.4 s |

The replay of the audit's targets gave the same 12 messages and the same context and chain counters
as the audit's own macOS/M3 run, so the solver is deterministic across platforms. Table build took
8.5 s (M3), 17–19 s (the author's machine) and 27–28 s (the Xeon, two builds at once); peak memory is
about 2.1 GiB.

**Padded mode, work per solution** (machine independent; 32 solutions for label targets: the 12 single-block
ones from the public logs, the audit's 10, our 10 new ones): mean 2^25.3 chain starts (standard error
about 20 %) and 2^27.9 context trials. 23 of the 32 were solved with the first context, 8 needed a
second and 1 a third. A context is abandoned after 2^26 chain starts (about 40 s on the Xeon), so a
target that needs four contexts costs close to three minutes there. An earlier reviewer reported a
182.5 s run; its raw log was not kept, so we treat it as unverified, and none of the 35 distinct timed
solves above needed more than three contexts. A chain start is not one SHA-256 evaluation, and chain
starts alone are not the total work.

**Compression-function mode:**
- 0.2–3.6 s per target.
- Measured cost: 2^22.17 chain starts per solution (pooled over 50 solutions), against 2^22.3
  predicted by the cost model.

**Witnesses** are in `witnesses/`: 28 records, 27 distinct (the "abc" message appears twice):
- 15 distinct padded messages: 14 of 55 bytes (11 labelled tests, the "abc" and "hello" examples, and
  1 fresh solve of 2026-09-29), and one 119-byte two-block message with a readable first block, whose
  19-step digest is SHA-256("");
- 12 compression-function witnesses (11 labelled tests and 1 from 2026-09-29).

All pass the checks of `checks/recheck_2026-10-13/` (28 records, 0 failures). The 22 solves made on
2026-10-13 are in `checks/recheck_2026-10-13/timing/`.

**Raw solver output** is in `witnesses/raw_solver_logs/`.

## How it works (short)

Full text: `docs/NOTE.md`.

1. **The state as a window.** Write the computation in terms of the a-register sequence. The state
   after r steps is the window a_{r−7}..a_r. A preimage is exactly a solution of the three message
   schedule equations for W_16, W_17 and W_18.
2. **Solving the equations in turn.** Each equation has a "pivot" word that enters only through
   Q(u) = σ0(u) − u. So the three equations can be solved in sequence by lookups in a table of Q.
3. **Basra's context shaping.** Choose middle words so that e_9 = e_10 = 0xFFFFFFFF. This makes the
   system triangular (each equation involves only words already solved for), up to one residual
   condition that holds with probability (3/4)^32.
4. **Padding.**
   - The two length words are absorbed in closed form (Basra's earlier observation, carried over to
     the shaped setting).
   - **First-generation solver (`src/pre19.c`):** The `0x80` end-marker byte is treated as an 8-bit
     filter (`2^-8`), and `a_5 = a_6` is relaxed to "soft shaping" (`popcount(a_5 ^ a_6) <= 3`) so
     contexts can be found in `~2^27.6` trials, at the cost of a `2^popcount` factor in chain starts.
   - **Exact algebraic inversion & exact `D = 0` shaping (`exploration/twenty_steps/solve20_padded_fast.c`,
     `test_speedup.c`):** Because `Sigma1(e_13) & 0xff` is independent of bits `18..31` of `e_13`
     (which rotate to bit positions `>= 12, >= 7, >= 19`), fixing the 8 pivot bits `18, 25..31` of `e_13`
     Allows solving `(W_13 & 0xff) == 0x80` in `O(1)` time by a single byte lookup while stepping the
     6 free bits `19..24` inside the inner loop (`256x` faster, `100%` branchless validity). Combined
     with algebraic `Ch`-annihilation (`e_15[8..31] = 0xffffff`, `e_14[8..31] = 0`, another `2.3x`),
     this finds **exact `D = a_5 ^ a_6 = 0` padded contexts in ~0.15 s**, completely eliminating the
     soft-shaping penalty. With 16-bit algebraic `Q_0^-1` table compression (`REC_HI`), a 32 GiB
     4-way exact table (`98.71%` 3-level chain survival), and Level-2 Hamming-weight pruning on
     `popcount(~(a_3 ^ a_5))`, 19-step padded preimages are generated in **under 10 ms** (hundreds
     per second on 16 cores), and the 20th padded step reduces to a single terminal 32-bit check
     `sigma0(W_4) + W_3 == W_19 - sigma1(W_17) - W_12` on the stream of 19-step padded solutions at
     `~2^45.4` expected chain starts.

## Scope and limits

- **19 steps (demonstrated padded witnesses) and 20 steps (unpadded witness + unified padded solver).**
  While `exploration/twenty_steps/r20_split.py` recorded why keeping the 19-step shaping (`a5 = a6`,
  `e9 = e10 = -1`) makes going below Basra's ≈2^45.4 hard:
  - For **20-step unpadded compression**, switching to the Zero-$a_3$ Manifold (`a3 = a5 = a6 = a8 = a9 = 0,
    a10 = -1`, in `exploration/twenty_steps/solve20_zero_a3.c` and the companion repository
    [srw/sha256-20step-preimages-with-no-padding](https://github.com/srw/sha256-20step-preimages-with-no-padding))
    eliminates the $(3/4)^{32} \approx 2^{-13.28}$ residual filter and solves the 20-step compression
    function in ~2^33 (21.1 min on 16 CPU vCPUs).
  - For **19- and 20-step padded messages**, `exploration/twenty_steps/solve20_padded_fast.c` removes the
    soft-shaping and `W_13` filter penalties via exact `O(1)` algebraic padding inversion (`D = 0`),
    producing 19-step padded preimages in milliseconds and reducing 20-step padded preimage search to
    the same `~2^45.4` chain-start regime as unpadded context shaping.
- **Message length.** Padded mode handles final blocks carrying 55 bytes (total length ≡ 55 mod
  64). That is enough for preimages, since the attacker chooses the message.
- **Second preimages.** Because the target can be any 256-bit value, one can take any message M' of
  any length, set the target to H19(M'), and search for a different message with the same H19. That
  is a second preimage for H19, not for SHA-256, and we have not run it as a separate experiment.
  We do not claim collisions: much stronger collision results are known at this step count.
- **Heuristic.** The search is probabilistic and its cost model is a heuristic supported by
  measurements. It succeeded on every target we recorded, but success for every possible target is not
  proven. The correctness of any output is a different matter: each one is checked by recomputing H19.
- **Not peer-reviewed.** Timings are printed by the solver (process CPU time).

## Checks so far

- **Independent re-implementation.** A separate agent was given only the note and the witnesses,
  with no code. It reimplemented the method and found 40 compression-function and 5 padded
  preimages for its own target. It confirmed the cost model and verified all witnesses.
- **Re-checks on 2026-09-29.** All archived witnesses and the key identities were checked again;
  see `checks/recheck_log_2026-09-29.txt`.
- **Audit and re-check on 2026-10-13.** Another AI agent audited the repository from the FIPS text
  and wrote its own SHA-256 (it counted 28 records and 27 distinct witnesses), ran the unchanged
  solver on 12 new targets including all-ones and all-zeros, and listed corrections to this
  repository's text: the classification of Homsirikamol et al. 2012, the cost model for soft
  shaping, the timing datasets, the "never run" wording for the meet-in-the-middle line, and the
  second-preimage wording. They are applied here; the repository owner then supplied the Homsirikamol
  et al. paper, and it confirms the audit's reading (padded, 16 rounds). We then wrote a third implementation, replayed
  the audit's 12 targets on a different platform (identical messages and counters), solved 10 more
  new labels, and found the audit's cost-model correction visible in the data
  (`checks/recheck_2026-10-13/results.txt`). We also recomputed the three 17-step padded messages in
  McDair's 2024 mailing-list post (see above; `results_mcdair.txt`). These are computational
  cross-checks by AI agents, not human peer review.

## How this was produced

This work was carried out with substantial AI assistance: Claude (Anthropic) agents did the
derivations, implementation and experiments, and the checking was done by separate agents. It was
directed and reviewed by the repository owner. Everything here is meant to be checked, not trusted:
the witnesses are self-verifying, and corrections are welcome.

## Layout

| path | contents |
|---|---|
| `src/` | `pre19.c` (first-generation solver), `aseq.py` (a-sequence toolkit), `validate_chain.py` (numerical check of the key identities) |
| `verify/` | stdlib verifier, and an independently written checker |
| `witnesses/` | all witnesses and raw solver logs |
| `docs/NOTE.md` | technical note: method, proof, costs, prior work |
| `checks/` | re-check logs; `recheck_2026-10-13/` has an independent SHA-256, witness counts, the 22 new solves and their statistics |
| `exploration/twenty_steps/` | `solve20_padded_fast.c` & `test_speedup.c` (exact `O(1)` `W_13` padding inversion, exact `D = 0` shaping, `<10 ms` 19-step padded & unified 20-step padded solver), `solve20_padded.c`, preliminary 20-step split analysis, and `solve20_zero_a3.c` (Zero-$a_3$ Bi-Directional Bridge solver; full repo at [srw/sha256-20step-preimages-with-no-padding](https://github.com/srw/sha256-20step-preimages-with-no-padding)) |

## Credit and licence

Context shaping is due to K. S. Basra; please cite his work together with this one (see
`CITATION.cff`). The code and data here are released under the MIT licence.

Two further points about Basra's earlier work, because they are not visible in his repository today:
- His repository at commit `c97f093` (2026-08-13) contained `code/padded_preimage.py`, which pins
  the three padding words W_13, W_14 and W_15 in closed form through the context words, in his
  original (unshaped) construction. A run of 167 contexts with it produced no witness, and the
  README at commit `102f701` (2026-09-10) said: "No padded SHA-256 preimage is claimed." So the
  closed-form handling of the length words is his observation; we carried it over to the shaped
  setting, where the third word cannot be pinned and becomes a filter.
- That history is no longer in the public repository, which today is a single commit (`f75b87a`,
  dated 2026-09-30). We read the earlier history (HEAD `102f701`) on 2026-09-28/29 and do not know
  how or when it was removed. His current manuscript still says: "Although the repository
  contains an exploratory closed-form way to impose those word values, no padded preimage was
  obtained; we make no padded-hash claim." (`paper_r19_final.tex`). We hold a shallow copy of the
  earlier history (the last 50 commits up to `102f701`, from 2026-08-07; earlier commits, if any,
  were never fetched); ask us if you need to check these statements.
