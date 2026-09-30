# Preimages for 19-step SHA-256 with standard padding

This repository gives working preimages for **SHA-256 reduced to 19 of its 64 steps, as a hash
function**: standard initial value, standard padding and feed-forward, with all 256 output bits
matching an arbitrary target.

Given any target digest, the solver finds a 55-byte message whose 19-step SHA-256 equals it. That
takes about 20 s (median) on one CPU core, after an 18 s precomputation.

**This builds directly on K. S. Basra's "context shaping"** for the 19- and 20-round SHA-256
*compression function* ([kamb-code/sha256-r19-preimage](https://github.com/kamb-code/sha256-r19-preimage)).
The key algebraic step is his. What is added here is a way to make it coexist with SHA-256's
message padding, so that the preimages are real messages rather than free 512-bit blocks.

Nothing here affects full 64-step SHA-256.

## Example

```
target  = SHA-256("abc")
        = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

message = 8ca254f8d04dd3fb04efaad6b1867588ffe833de49a144fedbf1839422892dc9f5a9158d8fc063290a14a9a6c55dfa245642a0247ec45a
          (55 bytes)

SHA-256 reduced to 19 steps (message) = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
```

## Padded vs. unpadded: what is new

| object | 19 steps | 20 steps |
|---|---|---|
| compression function (16 free message words, no padding) | Zaikin, SAT, *Constraints* 2026; Basra 2026 | Basra 2026 (GPU, about 2^45.4) |
| **hash function (real messages, standard padding)** | **this repository** | open |

As far as we could find (searched 2026-09-28/29), no padded preimage of 19-step SHA-256 had been
demonstrated before. Basra's repository states that no padded preimage is claimed there. If you
know of earlier work, please open an issue and we will correct this.

The theoretical meet-in-the-middle line reaches 47 steps, but at costs close to 2^256; none of it
has been run.

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
19 rounds instead of 64, keep the feed-forward, and apply it to every block.

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

**Padded mode:**
- 1.2–86 s per target on one core, median ≈ 20 s, over 13 timed targets.
- Measured cost: ≈ 2^25.9 chain starts per solution.

**Compression-function mode:**
- 0.2–3.6 s per target.
- Measured cost: 2^22.17 chain starts per solution (pooled over 50 solutions), against 2^22.3
  predicted by the cost model.

**Witnesses** are in `witnesses/`:
- 11 padded and 11 compression-function witnesses for public labels;
- a padded preimage of SHA-256("abc");
- a two-block message with a readable first block, whose 19-step digest is SHA-256("");
- fresh solves made on 2026-09-29;
- the "abc" and "hello" examples.

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
4. **Padding (added here).**
   - The two length words are absorbed in closed form.
   - The 0x80 end-marker byte becomes an 8-bit filter.
   - The shaping condition a_5 = a_6, which padding makes impossible to impose exactly, is relaxed
     to "differs in at most 3 bits". That costs a factor 2^popcount in extra tries.

## Scope and limits

- **19 steps only.** At 20 steps only compression-function results exist (Basra's).
  `exploration/twenty_steps/` records a preliminary analysis of why going below Basra's ≈2^45.4
  looks hard.
- **Message length.** Padded mode handles final blocks carrying 55 bytes (total length ≡ 55 mod
  64). That is enough for preimages, since the attacker chooses the message. It also gives second
  preimages of arbitrary messages, and collisions.
- **Heuristic.** It is a probabilistic algorithm. It succeeded on every target tried, but it is not
  proven for every possible target.
- **Not peer-reviewed.** Timings are self-reported by the solver.

## Checks so far

- **Independent re-implementation.** A separate agent was given only the note and the witnesses,
  with no code. It reimplemented the method and found 40 compression-function and 5 padded
  preimages for its own target. It confirmed the cost model and verified all witnesses.
- **Re-checks on 2026-09-29.** All archived witnesses and the key identities were checked again;
  see `checks/recheck_log_2026-09-29.txt`.

## How this was produced

This work was carried out with substantial AI assistance: Claude (Anthropic) agents did the
derivations, implementation and experiments, and the checking was done by separate agents. It was
directed and reviewed by the repository owner. Everything here is meant to be checked, not trusted:
the witnesses are self-verifying, and corrections are welcome.

## Layout

| path | contents |
|---|---|
| `src/` | `pre19.c` (solver), `aseq.py` (a-sequence toolkit), `validate_chain.py` (numerical check of the key identities) |
| `verify/` | stdlib verifier, and an independently written checker |
| `witnesses/` | all witnesses and raw solver logs |
| `docs/NOTE.md` | technical note: method, proof, costs, prior work |
| `checks/` | re-check log |
| `exploration/twenty_steps/` | preliminary 20-step analysis (negative result) |

## Credit and licence

Context shaping is due to K. S. Basra; please cite his work together with this one (see
`CITATION.cff`). The code and data here are released under the MIT licence.
