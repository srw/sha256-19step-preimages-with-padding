# Practical Preimages for 19- and 20-Step Reduced SHA-256: Solvers & Verified Witnesses

This folder (`paper_19_20_padded_sudoku/`) contains the standalone zero-dependency Python 3 verifier (`verify_all_witnesses.py`), all **46 distinct verified 19- and 20-step padded and unpadded witnesses (48 checks)**, and the multi-threaded C / CUDA GPU solvers discovered and developed using **Google DeepMind Gemini 4 Argon**.

---

## Solvers, Verification Suite & Witnesses in This Folder

| File | Description |
| :--- | :--- |
| **[`verify_all_witnesses.py`](verify_all_witnesses.py)** | Standalone zero-dependency Python 3 verifier that checks all **46 distinct 19- and 20-step SHA-256 preimage witnesses (48 checks)** (16 distinct 20-step padded + 16 distinct 19-step padded + 13 distinct 20-step unpadded compression + 1 Stage-2 completeness regression witness), verifying exact FIPS 180-4 Merkle–Damgård padding (`W13 & 0xff == 0x80`, `W14 == 0`, `W15 == 440` or `952`) and all internal Carry-Sudoku invariants (`a5 == a6`, `a10 == a11`, `e9 == e10 == e15 == 0xffffffff`). |
| **[`witnesses_20step.txt`](witnesses_20step.txt)** | All 14 single-block 55-byte 20-step FIPS 180-4 padded preimages and 12 20-step unpadded compression preimages across the standard public-label benchmark suite (`"abc"`, `"hello"`, `"padded test 1"`..`"padded test 11"`, `"unpadded test 1"`..`"unpadded test 11"`, `"hand-off cross-check 2026-09-29"`). |
| **[`demos_20step.txt`](demos_20step.txt)** | Flagship 19- and 20-step padded demonstration witnesses, including the 2-block (119-byte, $L = 952$) plain-ASCII prefix messages (`"This first block is plain text; the second one was computed....."`) for `SHA-256("")`, the single-block 20-step padded preimage for `SHA-256("abc")`, the 19-step exact-shaping $0^{256}$ witness, and the Stage-2 completeness regression witness. |
| **[`examples_abc_hello_20step.txt`](examples_abc_hello_20step.txt)** | 20-step padded preimages for `SHA-256("abc")` and `SHA-256("hello")`. |
| **[`witnesses_fresh_20step_2026-09-29.txt`](witnesses_fresh_20step_2026-09-29.txt)** | Fresh-label 20-step padded and unpadded preimage witnesses (`"hand-off cross-check 2026-09-29"`). |
| **[`FOUND_20ROUND_PREIMAGE.txt`](FOUND_20ROUND_PREIMAGE.txt)** | Discovery record and full 16-word message block for the all-zero target $H^* = 0^{256}$ 20-step padded SHA-256 preimage (`ec531306 c85ad4a4 ... 65416f80 00000000 000001b8`). |
| **[`solve20_padded_fast.c`](solve20_padded_fast.c)** | Host-verifiable multi-threaded CPU solver implementing 7-Stage Causal `Carry-Sudoku` (`266.8 ctx/s` on 10-core Apple M4, 16 threads) and 6.00 GiB / 32.00 GiB `Slot-Sudoku` tables (`6,242.8 sol19/s`, `191.1 node-h` / $2^{46.56}\text{ CPU-ref}$ per 20-step padded preimage). |
| **[`solve20_padded_cuda.cu`](solve20_padded_cuda.cu)** | CUDA GPU solver implementing `Carry-Sudoku`, Zero-Table $\mathbb{F}_2$ Inverse (`reconstruct_u_dev`), and 6.00 GiB / 16.00 GiB `Slot-Sudoku` tables (`53,847.5–132,888.7 sol19/s` and `29,310–114,507 ctx/s` on 1× NVIDIA L4 GPU on genuine contexts). |
| **[`solve20_padded_cuda_prod.cu`](solve20_padded_cuda_prod.cu)** | Multi-target production CUDA GPU solver (`twin2` / `single4`) used to generate the 20-step benchmark witnesses. |
| **[`solve20_zero_a3.c`](solve20_zero_a3.c)** | 20-step unpadded compression solver using the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge (`33.14 min` expected / `21.10 min` observed on 16 CPU cores, $2^{38.13}\text{ CPU-ref}$). |
| **[`test_all_12_theories.c`](test_all_12_theories.c)** | 14-Theory algebraic and empirical verification suite (`[E20]`–`[E21]`, Theorem 2 `[E28]`–`[E29]`, `[E37]`, and Table 9.4 directions). |

---

## Quick Verification (Standard Python 3, Zero Dependencies)

```bash
python3 verify_all_witnesses.py
```
