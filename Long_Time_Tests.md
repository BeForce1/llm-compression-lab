# Long-running tests

Everything that costs more than a coffee break, with what it buys and what it costs.
Nothing here is blocked on ideas — only on wall clock.

**Cost estimates come from measured throughput on this machine** (20-core CPU, no GPU,
SmolLM2-135M, float32). Scale them if you run elsewhere.

| path | measured | notes |
|---|---:|---|
| `llm_ptc` batched encode | **~1,050 B/s** | one forward per window; the fast path |
| `llm_ptc` sequential encode | ~70 B/s | one forward per token |
| `llm_ptc` sequential decode | ~65 B/s | **cannot be batched, ever** — needs token N to predict N+1 |
| `ptc` (pure maths) | ~24 KB/s | pure Python |
| Qwen3-0.6B | ~23 B/s | 4.4× the params, ~15× slower than SmolLM2 batched |

> **The asymmetry that dominates this whole file:** encoding parallelises because the
> token sequence is already known. Decoding cannot. Any test needing a verified
> round-trip costs ~16× more than the same test measuring ratio only.

---

## Tier 1 — worth doing first (1–2 hours each)

### 1.1 Prove the headline is actually lossless — **~75 min** ⭐ highest value per hour
The published 0.939 bpb on `alice29.txt` came from the batched encoder and **was never
decompressed**. This is the single biggest credibility gap in the repo.

```bash
export LLM_PTC_MODEL=HuggingFaceTB/SmolLM2-135M
python llm_ptc.py corpus/alice29.txt 152089        # sequential, encode AND decode
```
- **Buys:** turns caveat #1 in the README into a verified result. ~37 min encode + ~37 min decode.
- **Expect:** 0.939 ± 0.002 and `round-trip: ok`. If the bpb differs materially from the batched figure, the batched path is a worse approximation than the 8 KB test suggested — which is itself worth knowing.

### 1.2 Trend check before committing to enwik8 — **~1 hour**
We have 0.917 bpb on a 262 KB slice and are extrapolating to a 100 MB claim across
**380×** the data. Check the trend at 16× first.

```bash
python scripts/fetch_corpus.py                     # if enwik8 isn't there yet
python - <<'EOF'
data = open('corpus/enwik8','rb').read()[50_000_000:50_000_000+4*1024*1024]
while True:
    try: data.decode('utf-8'); break
    except UnicodeDecodeError: data = data[:-1]
open('corpus/enwik8_4m','wb').write(data)
EOF
python llm_ptc.py corpus/enwik8_4m 4194304 enc batch
```
- **Buys:** de-risks the 26-hour run. Also the first real test of whether the match model's gain *grows* with file size — its hash table sees 1.2M tokens here vs 75K in the ablation, and that's the main reason to expect the full run to beat the extrapolation.
- **Expect:** ≤ 0.917. If it doesn't improve, the full-enwik8 claim is weaker than assumed and Tier 2.1 should wait.

### 1.3 Does base really beat instruct? — **~1 hour**
The README states this as an untested hypothesis for why Qwen3-0.6B lost to a model 4.4×
smaller. It's a clean A/B.

```bash
LLM_PTC_MODEL=Qwen/Qwen3-0.6B-Base python llm_ptc.py corpus/enwik8_mid 262144 enc batch
# compare against the instruct-tuned figure already in results.json: 1.110 on 8 KB alice29
```
- **Buys:** either confirms a genuinely useful rule (*use base models for compression*) or removes a claim from the README. Both are wins.
- **Careful:** match samples exactly — use the same corpus and byte count for both, not the numbers already recorded against different samples.

### 1.4 SmolLM2-360M — **~1.5 hours** on full alice29
The model is the dominant lever: swapping it was worth 45%, all hand-built modelling ~1%.

```bash
LLM_PTC_MODEL=HuggingFaceTB/SmolLM2-360M python llm_ptc.py corpus/alice29.txt 152089 enc batch
```
- **Buys:** the cheapest remaining ratio gain. 2.7× params, expect ~2.5× slower.
- **Watch:** whether it beats 0.939 by more than it costs in speed. 135M already beat a 600M model, so bigger is *not* guaranteed better here.

### 1.5 book1 and the rest of the corpus — **~15 min** each
`llm_ptc` has only ever been measured on alice29, post2026 and enwik8 slices. `book1` has
a published ts_zip figure (1.431) and a published `xz` figure (2.717) we already match.

```bash
python llm_ptc.py corpus/book1 768771 enc batch
```
- **Buys:** a second directly-comparable data point against ts_zip. Cheap. Do it while something else runs.

---

## Tier 2 — overnight

### 2.1 Full enwik8 encode — **~26 hours** ⭐ the headline claim
The only way the ts_zip comparison stops being an extrapolation. 100,000,000 bytes at
~1,050 B/s.

```bash
python llm_ptc.py corpus/enwik8 100000000 enc batch > enwik8_full.log 2>&1 &
```
- **Buys:** a number that sits directly beside ts_zip's published **1.106 bpb** on the identical file. Current standing: 0.917 on a representative slice.
- **Do 1.2 first.** Don't spend a day confirming a trend you can check in an hour.
- **Progress:** prints running bpb to stderr every 2,000 tokens, so it's checkable mid-flight.
- **Memory:** logits are `[window, vocab]` float32 — ~200 MB per batch for SmolLM2. Watch it if you switch to a large-vocab model (Qwen3's 151K vocab is ~620 MB per batch).
- **No resumption.** If it dies at hour 20 you start over. Writing a checkpoint (token index + coder state + match tables) is a prerequisite if this proves flaky.

### 2.2 A large genuinely-unseen corpus — **collection + ~2 hours**
The contamination test rests on a **50 KB** sample from one 2026 arXiv paper. That's thin
for the load-bearing claim that our advantage isn't memorisation.

- **Do:** gather ~5 MB of text published after SmolLM2's cutoff, from several genres (papers, news, forum posts, code), not one source.
- **Buys:** turns "2.13× over xz on unseen text" from suggestive into solid, and separates *genre difficulty* from *memorisation* — currently a confound we admit but haven't removed.
- **Keep** the fetch script and the exact sources in the repo, or the result isn't reproducible.

---

## Tier 3 — days, or blocked on engineering

### 3.1 Full enwik8 verified round-trip — **~17 days** ❌ infeasible as built
100 MB decoded at ~65 B/s, and decoding **cannot** be parallelised. Do not start this.
Blocked on 4.1. Listed so nobody plans around it thinking it's just a long weekend.

### 3.2 enwik9 — **~11 days encode**
ts_zip publishes **1.084 bpb** on enwik9 and the Hutter record is **0.886**. 1 GB at
~1,050 B/s. Only sensible after 4.1 and a C/SIMD port; realistically wants a GPU
(ts_zip does 1 MB/s on an RTX 4090 — ~1,000× our rate).

---

## Tier 4 — engineering projects that unblock the above

### 4.1 Deterministic integer inference ⭐ the real blocker
Quantise weights to int8 and make inference bit-exact regardless of tensor shape,
thread count or library version. This is the load-bearing choice in ts_zip's design, and
we proved why by breaking it: our batched stream diverged from the sequential decoder at
**byte 253**.

Unblocks, all at once:
- the 16× batched path becomes a **valid codec**, not just a measurement → 3.1 becomes ~2 hours instead of 17 days
- cross-machine decompression, i.e. the thing that makes it a real format
- ~4× less weight traffic, since we're memory-bandwidth bound

Nothing else on this list is worth as much.

### 4.2 Checkpoint/resume for long encodes
Serialise token index, coder state (`lo`/`hi`/output), match tables and mixer weights.
Cheap to write, and it's what makes 2.1 and 3.2 survivable.

### 4.3 C/SIMD port — *after* the format is pinned
Order is already recorded in `ptc.py`'s docstring: int8 GEMM kernels (AVX-512 VNNI) are
worth 4–10×; the arithmetic coder is worth ~5%. Under a fixed time budget, speed is the
currency you buy ratio with — a faster inner loop affords a bigger model in the same wall
clock. **Not before the format is stable**; hand-tuning a moving target is the waste.

### 4.4 2D contexts in `ptc` for images
`ptc` loses to `xz` on `ptt5` (0.834 vs 0.655) because its contexts are 1-D while the
strongest predictor of a pixel is the one *above* it — `row_width` bytes back, and the
stride detector only tries (2,3,4,8,16). Add left/above/above-left contexts, then
benchmark against PNG. Build is short; the value is that it makes `ptc` competitive on a
whole data class it currently fails.

---

## Running these

- **Always `batch` for ratio-only runs.** 16× faster, byte-identical output on the 8 KB test, and the caveat is decodability, not accuracy.
- **Always background long jobs** and tee to a log. Progress goes to stderr every 2,000 tokens.
- **Slice mid-file.** The first 256 KB of enwik8 is XML preamble and scored 0.811 vs 0.917 mid-file — a 13% bias that nearly went into the README.
- **Match sample sizes exactly** when comparing configurations. Our bpb improves with file size, so a bigger sample flatters the newer run and manufactures fake wins.
- **Record every result in `results/results.json`**, then `python scripts/make_charts.py`. The README is generated from that file, not hand-edited.
- **Re-verify round-trip after any model or coder change.** A 26-hour encode of a broken codec is the worst outcome available.
