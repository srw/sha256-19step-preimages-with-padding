# Re-check of 2026-10-13

A third, independent check of the padded 19-step results, made after another AI agent audited the
repository (its counts and corrections are applied in `README.md` and `docs/NOTE.md`). Everything
here is standard-library Python except the solver, which is `src/pre19.c` unchanged (commit
`3130a9a`).

## What was done

1. `scripts/my_sha.py`: SHA-256 written from FIPS 180-4 with a reduced step count. Constants come from
   exact integer roots of the first 64 primes. Self-test: agrees with `hashlib` at 64 steps on 205
   messages (lengths 0–199, 255, 256, 1000, 4097 and one million bytes).
2. `scripts/my_witness_check.py witnesses`: parses the four witness files and checks every record
   (target = SHA-256 of the label; length; 19-step hit; 18-, 20- and 64-step misses; padding
   omitted or message misread as a raw block misses). Result: 28 records, 27 distinct (15 padded:
   14 of 55 bytes and one of 119 bytes; 12 compression), 0 failures.
3. The solver was run twice at the same time on two cores of an Intel Xeon at 2.1 GHz:
   - a replay of the 12 targets of the audit (all-ones, all-zeros, 10 labels containing a nonce chosen by
     the audit) with the audit's seed, where the audit had run on an Apple M3. The 12 messages and the
     12 triples (context trials, contexts used, chain starts) are identical to the audit's, which
     shows that the solver is deterministic across platforms;
   - 10 new labels with a nonce generated on 2026-10-13 (`timing/new_labels_manifest.json`).
   All 22 outputs verify with `scripts/my_run_verify.py`.
4. `scripts/my_ko_stats.py`: for every padded 55-byte witness, D = a_5 xor a_6, k = popcount(D),
   o = popcount(a_5 and D). Among the 30 witnesses with k = 3 the mean of o is 1.00, as the exact joint
   acceptance probability predicts for successful contexts, against 1.50 without the correction.
5. `scripts/my_pool_stats.py`: per-dataset timing and work per solution.

`results.txt` has the output of all of these, run from the repository root:

    python3 -I checks/recheck_2026-10-13/scripts/my_sha.py
    python3 -I checks/recheck_2026-10-13/scripts/my_witness_check.py witnesses
    python3 -I checks/recheck_2026-10-13/scripts/my_run_verify.py \
        checks/recheck_2026-10-13/timing/xeon_replay_of_audit_manifest.out \
        checks/recheck_2026-10-13/timing/audit_manifest_targets.txt
    python3 -I checks/recheck_2026-10-13/scripts/my_ko_stats.py witnesses \
        checks/recheck_2026-10-13/timing/xeon_replay_of_audit_manifest.out \
        checks/recheck_2026-10-13/timing/xeon_new_labels.out

To rerun the solver (about 2.1 GiB of RAM): `gcc -O3 -march=native -o pre19 src/pre19.c`, then
`./pre19 padded <targets, comma separated> 29 <seed> 3 26`. The seed of the replay is in the audit's
manifest and was 3540575513; the seed of the new labels is in `timing/new_labels_manifest.json`.

## Limits

- The two jobs shared the machine, so the times are indicative. Counts of context trials and chain
  starts do not depend on the machine.
- The last row of the `results.txt` statistics ("chain/ctx pooled over 32 label targets") mixes
  machines in its time columns; only its chain and context counts mean anything.
- These checks are computational cross-checks by AI agents, not human peer review.
- The audit's own Apple M3 timings are quoted in the README as the audit reported them; we reproduced
  its counters, not its clock.
