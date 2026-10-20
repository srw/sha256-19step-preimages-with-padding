# Practical Preimages for 19- and 20-Step Reduced SHA-256: Carry-Sudoku & Slot-Sudoku Publication Package

This folder (`paper_19_20_padded_sudoku/`) contains the complete publication paper, the companion **ELI5 & ELI-Undergraduate** pedagogical guide (referenced throughout the paper for equations `[E1]`–`[E41]`), both rendered PDFs, the standalone zero-dependency Python verifier, the verified 19-step and 20-step padded witnesses, and the C/CUDA solvers.

---

## Documents & Page Counts

| File | Format | Pages | Description |
| :--- | :---: | :---: | :--- |
| **[`PAPER_19_20_PADDED_SHA256.pdf`](PAPER_19_20_PADDED_SHA256.pdf)** | PDF | **14 pages** | **Main Research Paper:** *Practical Preimages for 19- and 20-Step Reduced SHA-256: Exact Algebraic Padding Inversion, Context Shaping, and the Carry/Slot-Sudoku Framework*. Covers 19-step and 20-step padded SHA-256, Theorems 1 & 2, equations `[E1]`–`[E41]`, complete 20-step padded trajectory table, Zero-$a_3$ unpadded bridge, and Apple M3 CPU / NVIDIA L4 GPU benchmarks. |
| **[`COMPANION_SUDOKU_ELI5_ELIU.pdf`](COMPANION_SUDOKU_ELI5_ELIU.pdf)** | PDF | **11 pages** | **Companion Pedagogical Document:** *Carry-Sudoku & Slot-Sudoku in Reduced SHA-256: A Self-Contained ELI5 and ELI-Undergraduate Companion Guide*. Explicitly referenced in the main paper; provides a self-contained glossary of all symbols and equations `[E1]`–`[E41]`, plus dual **ELI5 (Explain Like I'm 5)** and **ELI-Undergraduate (ELIU)** derivations using the Sudoku metaphor. |
| [`PAPER_19_20_PADDED_SHA256.md`](PAPER_19_20_PADDED_SHA256.md) | Markdown + LaTeX | — | Full Markdown/LaTeX source of the main paper. |
| [`COMPANION_SUDOKU_ELI5_ELIU.md`](COMPANION_SUDOKU_ELI5_ELIU.md) | Markdown + LaTeX | — | Full Markdown/LaTeX source of the companion ELI5 & ELI-Undergraduate document. |

*(Combined total across both PDFs: **25 pages** — 14 pages for the research paper + 11 pages for the companion ELI5/ELIU guide.)*

---

## Supporting Code, Solvers & Witnesses in This Folder

| File | Description |
| :--- | :--- |
| **[`verify_all_witnesses.py`](verify_all_witnesses.py)** | Standalone zero-dependency Python 3 verifier that checks: (1) the **20-step padded SHA-256 preimage** (`H20 = 0000...0000`) and all internal Carry-Sudoku invariants (`a5 == a6`, `a10 == a11`, `e9 == e10 == e15 == 0xffffffff`, `W13 & 0xff == 0x80`, `W14 == 0`, `W15 == 440`); (2) the **19-step padded SHA-256 preimage** (`H19 = SHA-256("abc")`); and (3) the **20-step unpadded compression preimage** (`H20 = deadbeef...`). |
| **[`FOUND_20ROUND_PREIMAGE.txt`](FOUND_20ROUND_PREIMAGE.txt)** | Discovery record and full 16-word message block for the first demonstrated 20-step padded SHA-256 preimage (`ec531306 c85ad4a4 ... 65416f80 00000000 000001b8`). |
| **[`solve20_padded_fast.c`](solve20_padded_fast.c)** | Multi-threaded CPU solver implementing 7-Stage Causal `Carry-Sudoku` (`86.4B trials/s` on 12-core Apple M3) and 6 GB / 32 GB `Slot-Sudoku` tables (`6,242.8 sol19/s`). |
| **[`solve20_padded_cuda.cu`](solve20_padded_cuda.cu)** | CUDA GPU solver implementing `Carry-Sudoku`, Zero-Table 5-Instruction $\mathbb{F}_2$ Inverse (`reconstruct_u_dev`), 6 GB / 16 GB `Slot-Sudoku`, Multi-Way Single-$T_{16}$ Fusion, and Warp-Compacted Shared-Memory Level-3 Queues (**`180,831.9 sol19/s`** and **`29,310.3 ctx/s`** on 1× NVIDIA L4 GPU). |
| **[`solve20_zero_a3.c`](solve20_zero_a3.c)** | 20-step unpadded compression solver using the Zero-$a_3$ Bi-Directional $Q_0/Q_1$ Bridge (`21.1 min` on 16 CPU cores). |
| **[`render_pdfs.py`](render_pdfs.py)** | Script to regenerate `PAPER_19_20_PADDED_SHA256.pdf` and `COMPANION_SUDOKU_ELI5_ELIU.pdf` from the Markdown sources via KaTeX + Headless Chrome. |

---

## Quick Verification (Standard Python 3, Zero Dependencies)

```bash
python3 verify_all_witnesses.py
```
