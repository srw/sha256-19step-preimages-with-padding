# Practical Preimages for 19- and 20-Step Reduced SHA-256: Publication Package, Solvers & Verified Witnesses

This folder (`paper_19_20_padded_sudoku/`) contains the complete publication paper (`PAPER_19_20_PADDED_SHA256_final-v1.pdf` and `PAPER_19_20_PADDED_SHA256.pdf`), the companion **ELI5 & ELI-Undergraduate** pedagogical guide (`COMPANION_SUDOKU_ELI5_ELIU.pdf`, referenced for equations `[E1]`–`[E41]`), the standalone zero-dependency Python 3 verifier (`verify_all_witnesses.py`), all 29 verified 20-step padded and unpadded witnesses (plus the 19-step padded witnesses), and the multi-threaded C / CUDA GPU solvers.

---

## Documents & Page Counts

| File | Format | Pages | Description |
| :--- | :---: | :---: | :--- |
| **[`PAPER_19_20_PADDED_SHA256_final-v1.pdf`](PAPER_19_20_PADDED_SHA256_final-v1.pdf)** | PDF | **20 pages** | **Main Research Paper (IACR ePrint / Journal Version):** *Practical Preimages for 19- and 20-Step Reduced SHA-256: Exact Algebraic Padding Inversion, Context Shaping, and Causal Bit-Slice Pruning*. Features upfront 19-step and 20-step FIPS 180-4 padded preimages and prior-work comparison table in Section 1 (Pages 1–2), full untruncated 110-hex-char 55-byte padded benchmark preimages in Section 8.4 (Tables 8.2 & 8.3), Theorems 1 & 2, equations `[E1]`–`[E41]`, complete 20-step padded trajectory table, Zero-$a_3$ unpadded bridge, and Apple M3 CPU / NVIDIA L4 GPU benchmarks. |
| **[`PAPER_19_20_PADDED_SHA256.pdf`](PAPER_19_20_PADDED_SHA256.pdf)** | PDF | **20 pages** | Synchronized build of the main research paper. |
| **[`COMPANION_SUDOKU_ELI5_ELIU.pdf`](COMPANION_SUDOKU_ELI5_ELIU.pdf)** | PDF | **15 pages** | **Companion Pedagogical Document:** *Carry-Sudoku & Slot-Sudoku in Reduced SHA-256: A Self-Contained ELI5 and ELI-Undergraduate Companion Guide*. Provides a self-contained glossary of all symbols and equations `[E1]`–`[E41]`, plus dual **ELI5 (Explain Like I'm 5)** and **ELI-Undergraduate (ELIU)** derivations. |
| [`PAPER_19_20_PADDED_SHA256_final-v1.md`](PAPER_19_20_PADDED_SHA256_final-v1.md) | Markdown + LaTeX | — | Full Markdown/LaTeX source of the main paper (`final-v1`). |
| [`PAPER_19_20_PADDED_SHA256.md`](PAPER_19_20_PADDED_SHA256.md) | Markdown + LaTeX | — | Synchronized Markdown/LaTeX source of the main paper. |
| [`COMPANION_SUDOKU_ELI5_ELIU.md`](COMPANION_SUDOKU_ELI5_ELIU.md) | Markdown + LaTeX | — | Full Markdown/LaTeX source of the companion ELI5 & ELI-Undergraduate document. |

---

## Supporting Code, Solvers & Witnesses in This Folder

| File | Description |
| :--- | :--- |
| **[`verify_all_witnesses.py`](verify_all_witnesses.py)** | Standalone zero-dependency Python 3 verifier that checks all **29 20-step SHA-256 preimage witnesses** (17 padded + 12 unpadded compression) as well as the **19-step padded SHA-256 preimage witnesses**, verifying exact FIPS 180-4 Merkle–Damgård padding (`W13 & 0xff == 0x80`, `W14 == 0`, `W15 == 440` or `952`) and all internal Carry-Sudoku invariants (`a5 == a6`, `a10 == a11`, `e9 == e10 == e15 == 0xffffffff`). |
| **[`witnesses_20step.txt`](witnesses_20step.txt)** | All 14 single-block 55-byte 20-step FIPS 180-4 padded preimages and 12 20-step unpadded compression preimages across the standard public-label benchmark suite (`""`, `"abc"`, `"hello"`, `"SHA-256"`, `"padded"`, `"preimage"`, `"19-step"`, `"crypto"`, `"sat-c"`, `"bitcoin"`, `"test"`, `"padded test 1"`, `"padded test 2"`, `"padded test 3"`). |
| **[`demos_20step.txt`](demos_20step.txt)** | Flagship 20-step padded demonstration witnesses, including the 2-block (119-byte, $L = 952$) plain-ASCII prefix message (`"This first block is plain text; the second one was computed....."`) for `SHA-256("")` and the single-block padded preimage for `SHA-256("abc")`. |
| **[`examples_abc_hello_20step.txt`](examples_abc_hello_20step.txt)** | 20-step padded preimages for `SHA-256("abc")` and `SHA-256("hello")`. |
| **[`witnesses_fresh_20step_2026-09-29.txt`](witnesses_fresh_20step_2026-09-29.txt)** | Fresh-label 20-step padded and unpadded preimage witnesses (`"fresh-2026-09-29"`). |
| **[`FOUND_20ROUND_PREIMAGE.txt`](FOUND_20ROUND_PREIMAGE.txt)** | Discovery record and full 16-word message block for the all-zero target $H^* = 0^{256}$ 20-step padded SHA-256 preimage (`ec531306 c85ad4a4 ... 65416f80 00000000 000001b8`). |
| **[`solve20_padded_fast.c`](solve20_padded_fast.c)** | Multi-threaded CPU solver implementing 7-Stage Causal `Carry-Sudoku` (`86.4B trials/s` on 12-core Apple M3) and 6 GB / 32 GB `Slot-Sudoku` tables (`6,242.8 sol19/s`). |
| **[`solve20_padded_cuda.cu`](solve20_padded_cuda.cu)** | CUDA GPU solver implementing `Carry-Sudoku`, Zero-Table 5-Instruction $\mathbb{F}_2$ Inverse (`reconstruct_u_dev`), 6 GB / 16 GB `Slot-Sudoku`, Multi-Way Single-$T_{16}$ Fusion, and Warp-Compacted Shared-Memory Level-3 Queues (**`180,831.9 sol19/s`** and **`29,310.3 ctx/s`** on 1× NVIDIA L4 GPU). |
| **[`solve20_padded_cuda_prod.cu`](solve20_padded_cuda_prod.cu)** | Multi-target production CUDA GPU solver used to generate the 29 20-step benchmark witnesses across 8× NVIDIA L4 GPUs. |
| **[`solve20_zero_a3.c`](solve20_zero_a3.c)** | 20-step unpadded compression solver using the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge (`21.1 min` on 16 CPU cores). |
| **[`render_pdfs.py`](render_pdfs.py)** | Script to regenerate `PAPER_19_20_PADDED_SHA256_final-v1.pdf`, `PAPER_19_20_PADDED_SHA256.pdf`, and `COMPANION_SUDOKU_ELI5_ELIU.pdf` from the Markdown sources via KaTeX + Headless Chrome + Quartz/ASCII85 distillation. |

---

## Quick Verification (Standard Python 3, Zero Dependencies)

```bash
python3 verify_all_witnesses.py
```
