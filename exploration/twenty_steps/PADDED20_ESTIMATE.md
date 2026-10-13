# Padded 20-step preimage: cost estimate (2026-10-13)

Back-of-envelope, from docs/NOTE.md (sections 4-6) and Basra's manuscripts (kamb-code/sha256-r19-preimage,
paper_submask_r20.tex, paper_r20_barrier.tex). A model, not a measurement.

## Inputs
- Basra, compression function, 20 steps: about 2^45.4 swept a_0 per preimage, with a 16 GiB table and 3 stored
  roots. That is ~2^32 a_0 per context, ~2^13.7 contexts, ~15 A100-hours mean. Two verified examples, one of
  them the all-ones digest, which we re-verified independently.
- 19-step padded (ours): padding pins W_13 low byte, W_14, W_15.
  - W_14 and W_15 are absorbed linearly (a_7, a_8).
  - W_13 is an 8-bit context filter.
  - Saturation forces a_5, a_6, so D = a_5 xor a_6 is forced.
  - Soft shaping accepts popcount(D) <= 3 and pays 2^popcount(D) in the chain.
  - Context search: about 2^8 * 2^19.6 cheap trials per usable context.
- At 20 steps the window argument is unchanged:
  - The target fixes a_13..a_20 and the unknowns are a_1..a_12.
  - W_15 is linear in a_8, W_14 in a_7, W_13 is a filter.
  - The 4th pivot a_5 is already fixed by e_9 = -1 in padded mode, so C19 is a pure 32-bit check, as in
    Basra's model.

## Estimate
- Per context: 19-step completions = 2^32 / 2^(15.26 + popD).
- C19 passes with probability about 3 * 2^-32.
- Contexts needed: about 2^(13.7 + popD).
- Chain work: about 2^(45.7 + popD). Context search: about 2^(41.3 + popD), so it is not dominant.

| policy | popD (typical) | total work | ratio to Basra's 2^45.4 | A100-hours (Basra rate) |
|---|---|---|---|---|
| accept popcount(D) <= 3 | ~2.9 | ~2^48.6 | ~9x | ~140 |
| accept popcount(D) <= 2 | ~1.9 | ~2^48.2 (context search ~2^46.6) | ~7x | ~105 |

=> Padded 20 steps is about 2^48-2^49: roughly 100-150 A100-hours, or about $150-400 rented. This is an
engineering and compute project, and needs no new idea: port Basra's 20-round GPU sweep to padded mode.

## What a solver-driven shaping search (option 3) can and cannot buy
- CAN: remove the soft-shaping overhead 2^popD (up to ~8x) by finding contexts or shapings with D = 0 that are
  compatible with padding. That brings padded 20 down to Basra's unpadded ~2^45.4.
- CANNOT (per Basra's barrier paper): make C19 triangular. The obstruction is Sigma0(a_4) subtracted directly
  inside W_9, a bijective linear term rather than a Ch/Maj selector, so no context choice removes it. The
  dependency graph acquires a heavy cycle at R = 20 (three at R = 21).
- Our own probe (r20_split.log) agrees: 0/500 single-bit context flips leave T16..T18 unchanged, so there is no
  context-side meet-in-the-middle structure on T19.
- Odds of a 2^32-scale win are revised down from <= 10% to a few percent.

## Caveats
- Assumes the cost of a "cheap trial" in the context search is comparable to one a_0 sweep step.
- Assumes Basra's 3-root, 16 GiB table regime carries over unchanged to padded contexts.
- Assumes C19 behaves like a uniform 32-bit condition in padded contexts. Basra models it as uniform and does not
  prove it.
- Needs a small validation (padded 20-step pipeline on a GPU, measuring per-context yield) before a full campaign.
