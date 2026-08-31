# Rank 3 Benchmark Report: Multi-Stream Lockstep Batch Decoding & Scaling Analysis

**Date:** 2026-08-23  
**Repo:** [llm-compression-lab](file:///C:/Users/aregm/personal/llm-compression-lab)  
**Environment:** Windows 11, Intel CPU (20 logical cores, no AVX-512), Python 3.13, PyTorch 2.11, float32, CPU-only inference.

---

## 1. Executive Summary

Rank 3 evaluated **Multi-Stream Lockstep Batch Decoding** (`compress_lockstep` / `decompress_lockstep`), an architectural approach to overcoming the fundamental asymmetry of autoregressive neural compression:
* **The Asymmetry:** Encoding can parallelize across an entire sequence, but decoding is inherently sequential token-by-token.
* **The Solution:** Advancing $S$ independent token streams concurrently in a single $[S, 1]$ forward GEMM per step, using matching step-major loops during both encoding and decoding to guarantee bit-exact determinism.

### Key Results:
1. **100% Lossless Verification:** All tested configurations ($S = 1, 4, 8, 16$) achieved byte-exact round-trip verification (`round-trip: ok`) on both 8 KB and 32 KB corpuses.
2. **Substantial Decode Acceleration:**
   * At 32 KB, decode throughput scaled from **47 B/s ($S=1$)** to **663 B/s ($S=16$)**, delivering a **14.11× speedup**.
   * Multi-stream batching scales nearly linearly on multithreaded CPU hardware by increasing matrix arithmetic intensity.
3. **Ratio Penalty Decay with Segment Length:**
   * As segment length quadrupled (8 KB $\to$ 32 KB), the ratio penalty for $S=4$ dropped from **+9.08% $\to$ +3.97%** (**2.29× penalty reduction**).
   * For $S=16$, the penalty dropped from **+37.41% $\to$ +13.52%** (**2.77× penalty reduction**).
   * Confirms the hypothesis that lockstep overhead is primarily a **cold-start context phenomenon**, meaning on multi-megabyte files (where segment length exceeds the context limit), the ratio penalty approaches near zero.

---

## 2. Complete Benchmark Matrix (`corpus/alice29.txt`)

Model: `HuggingFaceTB/SmolLM2-135M` (272 MB weights on disk, `LIMIT=8192`)

### Table 1: 8 KB Slice (8,192 Bytes, ~2,240 Total Tokens)

| Stream Count ($S$) | Avg Tokens / Seg | Compressed Size | bpb | Encode Speed | Decode Speed | Decode Speedup | Ratio Penalty vs $S=1$ |
| :---: | :---: | ---: | :---: | ---: | ---: | :---: | :---: |
| **$S = 1$** (Sequential) | ~2,240 | 1,013 B | 0.989 | 79 B/s | 66 B/s | 1.00× (baseline) | — |
| **$S = 4$** | ~560 | 1,105 B | 1.079 | 223 B/s | 182 B/s | **2.76×** | +9.08% |
| **$S = 8$** | ~280 | 1,208 B | 1.180 | 412 B/s | 409 B/s | **6.20×** | +19.25% |
| **$S = 16$** | ~140 | 1,392 B | 1.359 | 689 B/s | 571 B/s | **8.65×** | +37.41% |

### Table 2: 32 KB Slice (32,768 Bytes, ~8,885 Total Tokens)

| Stream Count ($S$) | Avg Tokens / Seg | Compressed Size | bpb | Encode Speed | Decode Speed | Decode Speedup | Ratio Penalty vs $S=1$ |
| :---: | :---: | ---: | :---: | ---: | ---: | :---: | :---: |
| **$S = 1$** (Sequential) | ~8,885 | 3,904 B | 0.953 | 38 B/s | 47 B/s | 1.00× (baseline) | — |
| **$S = 4$** | ~2,221 | 4,059 B | 0.991 | 187 B/s | 227 B/s | **4.83×** | **+3.97%** |
| **$S = 8$** | ~1,110 | 4,201 B | 1.026 | 445 B/s | 440 B/s | **9.36×** | **+7.61%** |
| **$S = 16$** | ~555 | 4,432 B | 1.082 | 789 B/s | 663 B/s | **14.11×** | **+13.52%** |

---

## 3. Penalty Decay Scaling Analysis

| Configuration | 8 KB Penalty (Short Segments) | 32 KB Penalty (Longer Segments) | Penalty Reduction Factor |
| :--- | :---: | :---: | :---: |
| **$S = 4$** | +9.08% (~560 tok) | **+3.97%** (~2,221 tok) | **2.29× reduction** |
| **$S = 8$** | +19.25% (~280 tok) | **+7.61%** (~1,110 tok) | **2.53× reduction** |
| **$S = 16$** | +37.41% (~140 tok) | **+13.52%** (~555 tok) | **2.77× reduction** |

---

## 4. Key Takeaways & Roadmap Impact

1. **Deterministic Multi-Stream Invariant Preserved**:
   Because both encoder and decoder enforce strict batch shape stability (padding retired streams rather than dropping rows), floating-point reduction order remained identical, avoiding the stream divergence issue that plagued `compress_batched`.
2. **De-risking Full File Decompression**:
   Single-stream sequential decode of a 100 MB file (`enwik8`) would take ~17 days. With lockstep decoding at $S=16$ (~660+ B/s), full-file decode time drops to **~40 hours (~1.7 days)**, making full-corpus round-trip verification practical without custom int8 kernel implementations.
