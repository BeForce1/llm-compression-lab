# Rank 1 Benchmark Report: Model Scaling & Cross-Corpus Evaluation

**Date:** 2026-08-23  
**Repo:** [llm-compression-lab](file:///C:/Users/aregm/personal/llm-compression-lab)  
**Environment:** Windows 11, Intel CPU (20 logical cores, no AVX-512), Python 3.13, PyTorch 2.11, float32, CPU-only inference.

---

## 1. Executive Summary

Rank 1 focused on zero-code-change experiments that sweep model capacity and test cross-corpus generalization:
1. **Model Capacity Sweep (`SmolLM2-360M`)**: Tested whether a 2.7x larger model improves compression on `alice29.txt`. Result: **0.798 bpb / 15,179 bytes** (a **12.8% reduction** over `SmolLM2-135M`), achieving **10.02x raw compression** and beating `xz -9` by **3.19x**.
2. **Cross-Corpus Literary Benchmark (`book1`)**: Tested `SmolLM2-135M` on the full 768,771-byte `book1` corpus against `ts_zip`'s published figure. Result: **1.270 bpb / 122,034 bytes**, beating `ts_zip`'s published **1.431 bpb** by **11.3%** and `xz -9` by **2.14x**.

---

## 2. Experiment 1: Scaling to `SmolLM2-360M` on `alice29.txt`

- **Input File:** `corpus/alice29.txt` (152,089 bytes, 41,933 tokens)
- **Model:** `HuggingFaceTB/SmolLM2-360M` (727 MB weights on disk)
- **Path:** Batched encode (`compress_batched`) with LIMIT=8192 context window.

### Results:

| Codec / Model | Compressed Size | Bits per Byte (bpb) | Raw Ratio | vs `xz -9` | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`llm_ptc` + `SmolLM2-360M`** | **15,179 B** | **0.798** | **10.02x** | **3.19x smaller** | **409 B/s** |
| `llm_ptc` + `SmolLM2-135M` | 17,400 B | 0.915 | 8.74x | 2.79x smaller | ~1,050 B/s |
| `ts_zip` (RWKV-169M, published) | ~21,711 B | 1.142 | 7.00x | 2.23x smaller | - |
| `bz2 -9` | 43,202 B | 2.272 | 3.52x | 1.12x smaller | ~2 MB/s |
| `xz -9` | 48,492 B | 2.551 | 3.14x | 1.00x (baseline) | ~3 MB/s |
| `ptc` (pure math context mixer) | 49,543 B | 2.606 | 3.07x | 1.02x larger | ~24 KB/s |

### Progression:
```
  ...  8,192 / 41,933 tokens: 0.819 bpb
  ... 16,386 / 41,933 tokens: 0.822 bpb
  ... 24,580 / 41,933 tokens: 0.795 bpb
  ... 32,774 / 41,933 tokens: 0.794 bpb
  ... 41,933 / 41,933 tokens: 0.798 bpb (Final)
```

---

## 3. Experiment 2: Cross-Corpus Evaluation on `book1`

- **Input File:** `corpus/book1` (768,771 bytes, 205,456 tokens)
- **Model:** `HuggingFaceTB/SmolLM2-135M` (272 MB weights on disk)
- **Path:** Batched encode (`compress_batched`) with LIMIT=8192 context window.

### Results:

| Codec / Model | Compressed Size | Bits per Byte (bpb) | vs `ts_zip` | vs `xz -9` | Throughput |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`llm_ptc` + `SmolLM2-135M`** | **122,034 B** | **1.270** | **-11.3% smaller** | **2.14x smaller** | **577 B/s** |
| `ts_zip` (RWKV-169M, published) | ~137,495 B | 1.431 | baseline | 1.90x smaller | - |
| `bz2 -9` | 232,598 B | 2.420 | +69.1% larger | 1.12x smaller | ~2 MB/s |
| `ptc` (pure math) | 257,433 B | 2.679 | +87.2% larger | 1.01x larger | ~24 KB/s |
| `xz -9` | 261,116 B | 2.717 | +89.9% larger | 1.00x (baseline) | ~3 MB/s |

### Progression:
```
  ...  40,968 / 205,456 tokens: 1.304 bpb
  ...  81,938 / 205,456 tokens: 1.303 bpb
  ... 122,908 / 205,456 tokens: 1.287 bpb
  ... 163,878 / 205,456 tokens: 1.277 bpb
  ... 205,456 / 205,456 tokens: 1.270 bpb (Final)
```

---

## 4. Key Findings

1. **Model Parameter Scaling**:
   Increasing parameters from 135M to 360M yielded a **12.8% reduction** in bits/byte on `alice29.txt`, dropping from `0.915 bpb` to `0.798 bpb` while maintaining a CPU throughput of ~409 B/s.
2. **Consistency Across Literary Texts**:
   The model beats published state-of-the-art neural compressors (`ts_zip`) on multiple diverse literary corpora:
   - `alice29.txt`: **0.915 bpb** (135M) / **0.798 bpb** (360M) vs ts_zip's `1.142 bpb`.
   - `book1`: **1.270 bpb** (135M) vs ts_zip's `1.431 bpb`.
3. **Long Context Synergy**:
   On `book1`, the running bpb improved progressively over the length of the book from `1.340 bpb` (first 8K tokens) down to `1.270 bpb` as long-range matches and context stabilized.
