# Long-running tests

Everything that costs more than a coffee break, with what it buys and what it costs.
Nothing here is blocked on ideas — only on wall clock.

**Cost estimates come from measured throughput on this machine** (20-core CPU, no GPU,
SmolLM2-135M, float32). Scale them if you run elsewhere.

| path | measured | notes |
|---|---:|---|
| `llm_ptc` batched encode | **~1,050 B/s** | one forward per window; the fast path |
| `llm_ptc` sequential encode | ~87 B/s | one forward per token |
| `llm_ptc` sequential decode | ~85 B/s | needs token N to predict N+1 — but see the lockstep note |
| `ptc` (pure maths) | ~24 KB/s | pure Python |
| Qwen3-0.6B | ~23 B/s | 4.4× the params, ~15× slower than SmolLM2 batched |

> ⚠️ **Every estimate below was computed at ~70 B/s and is now ~20% pessimistic.**
> The 2026-07-31 audit found the thread-count default (`os.cpu_count()` = 20) was the
> *slowest* of five settings for the sequential path — 42 B/s vs 87 at 6 threads, a 2.07×
> penalty paid on every encode and decode since the project began. Defaults are now set
> per-path (6 for sequential, all cores for batched, which has the opposite optimum).
> Divide the sequential wall-clock numbers below by ~2 where they were derived from the
> old rate; they have not all been recomputed.

> **The asymmetry that dominates this whole file:** encoding parallelises because the
> token sequence is already known. Decoding cannot — *within one stream*. Across several
> independent streams advanced in lockstep it can, which is the one idea in this file
> that would change Tier 3 without solving 4.1. See 4.6.

---

## Tier 1 — worth doing first (1–2 hours each)

### ~~1.1 Prove the headline is actually lossless~~ — **done 2026-07-31**
```bash
export LLM_PTC_MODEL=HuggingFaceTB/SmolLM2-135M
python llm_ptc.py corpus/alice29.txt 152089        # sequential, encode AND decode
```
**Result: `round-trip: ok`, 17,873 B, 0.940 bpb.** Prediction was 0.939 ± 0.002; it landed
inside, so the batched path is a faithful approximation (0.15%) and stays usable for
ratio-only runs. The README, charts and `results.json` now quote the verified figure.

Two corrections to this entry's own estimates, worth keeping visible:
- **~75 min was wrong; it took ~3 hours.** The estimate came from small-sample throughput,
  where the context window is mostly empty. See trap 3 in `handoff.md`.
- **Decode is not the mirror of encode in cost**, and the run's own 30/42 B/s figures can't
  settle it — unrelated benchmarking was running on the machine at the time. Anyone wanting
  that number needs a quiet box.

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
- **Compare against 0.939, not the 0.940 headline.** That command runs `batch`, so the batched 135M figure is the like-for-like baseline. Mixing the two paths manufactures a 0.001 bpb difference that is the encoder, not the model.

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

⚠️ **Do not start with `torch.ao.quantization.quantize_dynamic`** — audited 2026-07-31 and
it cannot deliver either half of what this item needs:
- **Not deterministic.** It computes activation scales *per batch at runtime*, so the same
  row gives different outputs alone vs inside a batch — reproduced in under 10 s on a toy
  `nn.Linear` stack: max abs diff **0.067**, `torch.equal` False. That is the exact failure
  class that broke `compress_batched`, surviving quantisation untouched.
- **Not faster here.** This torch ships only the `onednn` quantized engine (no fbgemm), and
  its dynamic-quant linear at M=1 does not parallelise — it gets *slower* with more threads.
  Best int8 measured ~1.0–1.15× over tuned fp32, and it loses at every multithreaded setting.

What the requirement actually implies: **static, export-time activation scales**, integer
accumulation, and fixed-point softmax/RMSNorm/RoPE — i.e. pinned custom kernels, which is
what ts_zip and NNCP ship. Weeks, not days. Before committing to that, price it against
**4.6**, which buys most of the same wall-clock win with none of the kernel work.

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

### 4.4 Record-level columnarisation for SQLite ⭐ the Part 2 follow-up
Page-kind grouping measured 0.5–4.1% and is a dead end. But the *same* columnar mechanism
gave **17–28%** on SQL dumps, and the only reason it fails on SQLite is that the fields sit
behind a binary record format instead of in plain text.

⚠️ **Rescoped by the audit: do not target `wiki.db`.** Leaf-cell payload is 90% of its
compressed size but it is one TEXT column, so the ceiling there is ~0.1%. The dbs where
dump-level columnarisation paid have payload as a *minority* of bytes (chinook.db: 36% raw,
indexes alone 47%), putting the honest ceiling at **~10%** over page-sorted zstd, on
OLTP-shaped databases only.

⚠️ **Build the churn fixture first.** All three sample dbs are freshly imported and contain
**zero** freeblocks, zero overflow cells and zero stale bytes in unallocated gaps. Every
hard byte-exact-rebuild path is therefore unexercised: a columnariser that gets the overflow
spill formula, freeblock contents or stale-gap preservation wrong will pass every round-trip
assert here and corrupt the first real database with delete/update history. Generate a db
with mixed INSERT/DELETE/UPDATE, no VACUUM, and rows over 4061 B, and assert against that.

A cheaper intermediate step exists: parse leaf cells **read-only** (no rebuild risk) and
compress the index-page group with the payload as a zstd dictionary. Measured −15.7% on
wiki_meta.db's index pages, ~3.8% of the file — pays only where indexes are text keys.

The upside here grew: on dumps, per-column value re-spelling was worth more than the
regrouping itself (17.8% → 28.6% on chinook, and 17.4% → 22.2% on wiki_meta). SQLite already stores ints as binary rather than decimal
ASCII, so the *planes* half of that win is partly pre-collected — but the *delta* half is
not, and rowid/foreign-key ramps are exactly what a b-tree is full of. Reuse `encode_col`
directly once the fields are reachable; it takes a list of byte values and needs nothing
SQL-specific.

The work: parse cells inside leaf-table pages, split record payloads column-major, and
keep enough page metadata (cell pointer array, free blocks, overflow chains) to rebuild
each page byte-exactly. Byte-exactness is the hard part and the whole point — a logical
dump-and-reload is **not** a lossless compressor, because page layout and vacuum state are
unrecoverable from the rows.

Worth doing because the mechanism is now proven rather than hypothesised, and SQLite is one
of the highest-volume structured formats on earth with no format-aware compressor. Days,
not hours.

### 4.6 Multi-stream lockstep decode ⭐ the cheap alternative to 4.1
**PILOT BUILT AND RUN, 2026-07-31 — mechanism confirmed, cost not yet priced.**
"Decoding cannot be batched" is true *within* one stream and false *across*
streams. Split the token sequence into S segments, give each its own KV cache, match bank
and mixer, and advance all S by one token per step in a single `[S,1]` forward.

Why it stays lossless where `compress_batched` did not: the encoder runs the **same
step-major loop** as the decoder, so both sides execute identical batched GEMMs in
identical order. The divergence at byte 253 came from encoder and decoder using *different*
batch shapes; here they use the same one.

- **Buys:** ~5–10× on decode *and* on encode-of-a-decodable-stream at S=16–64. Traffic model
  at ctx 1024, S=16: (540 MB weights + 16×47 MB KV)/16 tokens ≈ 81 MB/token vs 587 MB
  sequential. That drops item **3.1 from ~17 days to ~2 days** without solving 4.1.
- **Ratio cost:** S context restarts (measured as noise — see `negative_results`) plus lost
  cross-segment matches. Bounded by the match model's total contribution, −1.2%.
- **Format:** S and the segment lengths go in the header.
**Pilot result** (scratch prototype, 8,185 B of alice29, byte-exact round-trip at every S):

| S | bpb | decode | vs sequential |
|---:|---:|---:|---|
| 1 | 0.989 | 76 B/s | — |
| 4 | 1.077 | 187 B/s | **2.5× decode, +8.9% size** |
| 8 | 1.178 | 271 B/s | 3.6× decode, +19.1% size |
| 16 | 1.350 | 672 B/s | 8.8× decode, +36.5% size |

The speedup is real and lands in the predicted 5–10× band at S=16. **The ratio cost does
not** — predicted <0.5%, measured 8.9–36.5%. It scales with S, which is the signature of a
per-segment *cold start*, not of lost cross-segment matches: at 8 KB an S=16 segment is only
~140 tokens, so each stream spends its whole life in the low-context regime the design exists
to escape.

The <0.5% prediction came from extrapolating "post-slide tokens cost the same as
deep-in-window ones" — but that is about window *slides*, where the model still gets WINDOW
tokens of context, not about starting from zero. (And that finding is itself now flagged; see
`negative_results`.)

**And the control confirms it.** Same S=4, same corpus, 4× the segment length:

| sample | segment | sequential | lockstep S=4 | penalty |
|---|---:|---:|---:|---:|
| 8 KB | ~560 tok | 0.989 | 1.077 | +8.9% |
| 32 KB | ~2,221 tok | 0.974 | 0.999 | **+2.6%** |

4× the segment length cut the penalty **3.5×**. Lost cross-segment matches would not shrink
that fast — so the cost is a function of **segment length, not of S**, and S is close to free
once segments are long. That flips the recommendation: this is worth engineering, on files
big enough to give long segments.

- **Still unmeasured, deliberately:** enwik8 at S=16 gives ~1.7M-token segments, ~770× the
  32 KB test, where the penalty should be far below 1%. Two points is a direction, not a
  curve, and trap 3 in `handoff.md` is exactly about extrapolating this kind of thing. Run it
  before quoting it.
- **Also unresolved:** the pilot uses equal-length segments and drops the remainder. A real
  implementation needs per-segment lengths in the header and streams that retire at different
  steps without changing the batch shape mid-run.

### 4.5 2D contexts in `ptc` for images
`ptc` loses to `xz` on `ptt5` (0.834 vs 0.655) because its contexts are 1-D while the
strongest predictor of a pixel is the one *above* it — `row_width` bytes back, and the
stride detector only tries (2,3,4,8,16). Add left/above/above-left contexts, then
benchmark against PNG. Build is short; the value is that it makes `ptc` competitive on a
whole data class it currently fails.

---

## Documentation owed (minutes, not hours — but load-bearing for honesty)

**D.1, D.2 and D.3 are DONE** — all three are in the README now. D.4 still waits on 1.1.

### ~~D.1 Separate "measured" from "decodable" in the README~~ — done
The headline says 8.5×. What is actually **verified round-trip** is smaller, and the
README does not currently draw that line. Add a table:

| | largest verified round-trip | compression |
|---|---:|---:|
| `ptc.py` | 1,029,744 B, any bytes incl. binary | 3.1× |
| `llm_ptc` + SmolLM2 | 152,089 B — whole of `alice29.txt` | 8.5× |

The distinction readers care about has since resolved in the good direction: 1.1 landed, so
**8.5× is now a verified round-trip rather than a measurement** — on one machine.

### ~~D.2 The break-even line~~ — done
Against `xz` we save ~0.2 bytes per input byte, so a 272 MB model repays itself after
**~1.35 GB** of text — which at 21–65 B/s decode takes 8 months to 2 years to read back.
One sentence, and it converts "8.5× smaller!" into a claim that survives scrutiny.

### ~~D.3 "Decodable" ≠ "portable"~~ — done
Even after 1.1 passes, a verified round-trip holds only for the same machine, thread
count and library versions. That is not a format. Say so next to the headline, not only
in caveat #3.

### ~~D.4 Re-point the headline at the shipped config~~ — done
The 0.939 figure predated the match model and mixer, so it was not reproducible with the
shipped defaults. Task 1.1's **0.940 bpb / 17,873 B** replaced it across `results.json`,
the charts and the README. Two hardcoded copies of the old numbers turned up during the
swap — `chart_shapes` held `17.8` and `chart_speed_ratio` held `963, 0.939` — which is why
"single source of truth" needs the charts to actually *read* the JSON. Both now do.

## Running these

- **Always `batch` for ratio-only runs.** 16× faster, byte-identical output on the 8 KB test, and the caveat is decodability, not accuracy.
- **Always background long jobs** and tee to a log. Progress goes to stderr every 2,000 tokens.
- **Slice mid-file.** The first 256 KB of enwik8 is XML preamble and scored 0.811 vs 0.917 mid-file — a 13% bias that nearly went into the README.
- **Match sample sizes exactly** when comparing configurations. Our bpb improves with file size, so a bigger sample flatters the newer run and manufactures fake wins.
- **Record every result in `results/results.json`**, then `python scripts/make_charts.py`. The README is generated from that file, not hand-edited.
- **Re-verify round-trip after any model or coder change.** A 26-hour encode of a broken codec is the worst outcome available.
