# Where this stands, and what to do next

Written 2026-08-01, after a full audit of the codebase. Companion to
[`handoff.md`](handoff.md) (orientation, traps, hard rules) and
[`Long_Time_Tests.md`](Long_Time_Tests.md) (everything costed and ordered).

---

## What we can actually claim

**0.915 bpb on `alice29.txt`, verified byte-exact round-trip on the whole file.** That is
8.74×, or 2.79× smaller than `xz -9`, and 19.9% under ts_zip's published figure on the
identical file. *(Was 0.940 until 2026-08-03, when the context sweep below moved the
default `LIMIT` from 1024 to 8192 and the headline was re-earned at shipped defaults.)*

Three caveats that are load-bearing, not throat-clearing:

1. **The margin over ts_zip is a model-vintage artifact.** We used a 2024 model against
   their 2023 one. It is not an algorithmic advance and should never be presented as one.
2. **"Verified" means this machine.** Same thread count, same library versions. It is a
   verified round-trip, not a portable format, and no amount of further running fixes that.
3. **The break-even is unreachable, and moved the wrong way.** A 272 MB model repays itself
   after ~1.33 GB of text, which at the measured 32 B/s takes **481 days** of continuous
   reading. The better ratio made this worse, not better — 2.65% of ratio cost 2.7× of
   speed, pushing break-even out from ~180 days. Quote the ratio with that attached.

The honest summary is the one `handoff.md` already gives: **a well-measured reproduction,
not a contribution.** What is genuinely worth keeping is the method and the negative
results — sixteen refuted predictions with measurements, and instrument bugs caught by
sanity checks rather than luck.

---

## What the audit changed (2026-07-31 → 08-01)

| | before | after |
|---|---|---|
| encode **and** decode throughput | baseline | **2.07× faster** |
| known silent-corruption / crash bugs | 9 | 0 |
| refuted predictions on record | 9 | 10 (+3 moved to sql-compression; 14 as of 2026-08-31) |
| charts with hardcoded numbers | 3 | 0 |

**The single biggest win was one line.** `LLM_PTC_THREADS` defaulted to `os.cpu_count()`,
which had never been swept, and which the README described as "the measured optimum". It
was the *slowest* of five settings:

| threads | 20 | 10 | 8 | 6 | 4 |
|---|---:|---:|---:|---:|---:|
| sequential encode | 42 B/s | 82 | 86 | **87** | 83 |

A single-token step is ~210 tiny GEMVs, one parallel region each, on a
memory-bandwidth-bound workload — past ~6 threads you buy fork-join barriers, not
throughput. That 2.07× was paid on every encode and decode this project ever ran, including
the three-hour full-file verification. The batched path has the **opposite** optimum (one
big GEMM per window, so it wants every core), and the two are now set separately.

Four latent bugs were fixed in `llm_ptc.py`, all no-ops for the current model — the
4,096-byte round-trip is byte-identical at 515 B — and all armed by the model swaps below:
the `WINDOW`/`LIMIT` clamp, `add_special_tokens` (which would have broken any BOS-inserting
tokeniser *after* a long run was paid for), the batched blob being indistinguishable from a
real stream, and the `clean_up_tokenization_spaces` default.

Two instruments were producing evidence for conclusions they could not support:
`probe.py`'s "warm" bucket contained provably **zero** post-slide tokens (it was 64 tokens
of memorised `alice29` opening), and four chart values were hardcoded — including classical
codec speeds understated by 5–33×, all in the direction that flattered this repo.

---

## What to do next

### ~~0. Sweep the thread count for lockstep~~ — **DONE 2026-08-31.** Premise refuted; the sweep paid for itself anyway

Fastest-of-3, interleaved, S=8 on 16,384 B:

| threads | 4 | 6 | 8 | 12 | 20 |
|---|---:|---:|---:|---:|---:|
| lockstep encode | **350 B/s** | 288 | 276 | 269 | 205 |

**Monotonically decreasing — the *sequential* shape, not the batched one.** So the
inherited default of 6 was approximately right, and there was no 2.07×-style win here. The
within-config spread (1.29–2.18×) is wider than the gap between 4, 6 and 8 (1.27×), so
those three are **not separable** at n=3 and no optimum should be quoted among them.

What it was worth: the plausible change — *lockstep batches a GEMM, so give it every core
like `compress_batched` does* — would have been **1.71× slower**. Preventing that was the
payoff, not a speedup.

**And the column the sweep wasn't looking at was the real find.** Output size varied with
thread count: 2,209 B at 4/6/8 and 2,210 at 12/20, consistently across all three passes.
Cross-decoding those was **silent corruption** — a 12-thread blob read at 6 threads
returned 8,692 bytes for an 8,192-byte input, with no exception raised. The thread count is
a header byte now and `_apply_threads` restores it at decode; cross-decode verified in both
directions.

The instrument was validated before it was trusted: the sequential control reproduced the
known 6-vs-20 result at 1.66× and 2.31×, bracketing the recorded 2.07×.

### ~~1. Sweep the context length~~ — **DONE 2026-08-03.** Worth −11.3%

Predicted "+1–3%". Measured, on SmolLM2's native 8192:

| LIMIT | 1024 | 2048 | 4096 | 8192 | total |
|---|---:|---:|---:|---:|---:|
| alice29 (memorised) | 0.958 | 0.946 | 0.938 | 0.934 | **−2.5%** |
| post2026 (unseen) | 1.331 | 1.260 | 1.220 | 1.181 | **−11.3%** |

The old default came from a sweep that found context worth −0.7% — **on GPT-2, whose
maximum *is* 1024.** It measured a ceiling and called it a property of context.

Two things worth carrying: on unseen text the curve **has not converged** at 8192 (last
doubling paid 3.20% against the previous 3.17%), so 8192 is not enough context, it is all
this model has. And the split between the two rows is the cleanest memorisation evidence
here — on text the model has read, context is redundant with the weights.

Default now 8192, clamped per model. Cost: encode 87 → 32 B/s. Taken, because ratio is
this codec's only deliverable and its speed was already far past usable.

### ~~2. Run the model A/Bs~~ — **DONE 2026-08-03.** Three claims died

| tested at matched sample and context | outcome |
|---|---|
| base beats instruct | **confirmed** — −16.7% alice29, −14.6% arXiv |
| bigger loses (the recorded finding) | **refuted** — 135M → 360M is −13%, everywhere |
| Qwen3-Base generalises better to unseen text | **refuted by a genre control** — its 12.2% arXiv win is a 0.7% loss on unseen narrative |
| unseen text costs ~38%, and that is memorisation | **refuted** — genre held constant, it costs −0.8% to +5.7% |

Full matrix in `results.json` → `model_comparison_2026_08_03`. Still open:
**`gemma-3-270m` is a gated repo** (needs a licence acceptance + `HF_TOKEN`), so the
big-vocab counter-test is untested; **Qwen3-Base never reached its context** — native
32,768, tested to 4096, still improving, capped by 4.3 GB of RAM not by the model; and the
genre control needs **post-cutoff fiction** to be airtight, since Victorian narrative vs
2026 journalism is two narrative genres rather than one.

<details><summary>the original plan, for the record</summary>

### 2. Run the model A/Bs — **~1 hour each** ⭐ now the best value per hour

The model is worth 45%; everything hand-built is worth ~1%. Unblocked by the
`add_special_tokens` fix — and the context sweep just made one of these urgent.

**`model_comparison` is not currently a fair test.** Every row ran at `LIMIT=1024`. That
was fine for GPT-2 vs SmolLM2 (1024 is GPT-2's maximum, so both were at a real ceiling),
which is why the 45% model finding stands. It is *not* fine for Qwen3-0.6B, measured at
1.11 bpb against a **32,768-token native context** — at 1/32nd of its context, that row is
partly a context measurement wearing a model's name. Re-run each model at its own native
maximum, and report the LIMIT beside every figure.

- `Qwen/Qwen3-0.6B-Base` — settles base-vs-instruct, *and* re-tests the "not worth its
  size" verdict now known to have been measured under a handicap.
- `HuggingFaceTB/SmolLM2-360M` — compare against **0.934** (batched, LIMIT=8192), not the
  0.915 headline; mixing paths and LIMITs manufactures differences that are the harness,
  not the model.
- `google/gemma-3-270m` — the counter-test to "big vocab wastes capacity". Needs a
  detokenise pre-flight; it is SentencePiece, not byte-level BPE.

**Watch the memory.** The batched path materialises `LIMIT × vocab × 4` bytes of logits:
1.6 GB for SmolLM2 at 8192, but **5.0 GB for Qwen3's 151,936 vocab** and ~8.6 GB for
Gemma's 262,144. This box has 15.7 GB with ~2.4 GB typically free. Cap `LLM_PTC_LIMIT` for
the wide-vocab models or use the sequential path, and *say which* in the results.

</details>

### 2b. The headline is pointed at the wrong model — **the extrapolation has been run**

Predicted "near 0.80 bpb, and must not be quoted until it is run". It was run on
2026-08-23 and landed at **0.798 bpb / 15,179 B** on the whole file — 10.0× versus raw,
3.19× versus `xz -9`, at 409 B/s. Prediction inside its own band, which is the rare case.

**It is still not the headline, and the reason is the same one that demoted 0.939.** That
run used the *batched* path, so it is an entropy measurement, not a decodable stream. What
would make it the headline is a full-file sequential encode **and** decode, same as the
0.915 run — call it 7–10 h at 360M's slower rate, or a weekend at S=16 lockstep now that
the slide path works.

~~One caveat the run itself carries: 0.798 batched against 0.915 sequential crosses paths.~~
**Closed 2026-08-31.** The like-for-like baseline is **17,406 B / 0.916 bpb** (135M,
batched, `LIMIT=8192`, whole file, 781 B/s), so the model gain is **−12.79%** against the
−12.76% previously quoted. The cross-path comparison was harmless — which is a measurement
now, not an assumption.

It also refuted its own prior: this file expected ~0.934, the figure already recorded for
that configuration. That figure came from a **65,536 B sample**, and bpb improves with file
size, so it would have inflated the 360M gain to ~14.6%. And the batched path measured
**6 bytes / 0.034%** from the verified sequential figure, against 27 bytes / 0.15% at
`LIMIT=1024` — more faithful with more context, and still not decodable.

Price it honestly first: 360M is 2.7× the parameters, so it is slower than the 32 B/s that
already put break-even at 481 days, and the model on disk grows from 272 MB to ~720 MB.
The ratio improves and the economics get worse again.

### ~~3. Build lockstep decode properly~~ — **DONE 2026-08-03**, and it took hours not days

Shipped in `llm_ptc.py` as `compress_lockstep` / `decompress_lockstep`, ~130 lines.
Byte-exact round-trip at **S=1, 2, 4, 8, 16**; `decompress()` routes on a header bit.

| S | bpb | decode | speedup | ratio cost |
|---:|---:|---:|---:|---:|
| 1 | 0.989 | 73 B/s | — | — |
| 4 | 1.079 | 257 B/s | 3.50× | +9.1% |
| 16 | 1.359 | 777 B/s | **10.59×** | +37.4% |

**Cost shrinks with segment length, as predicted — but by less than predicted.** 4×
the segment length took the S=4 penalty from **9.08% → 3.97%**. The pilot claimed that
same step gave 8.9% → 2.6%; it does not replicate, and 3.97% is the number to quote.

Two costs are conflated at these sizes and it matters: 560- and 2,221-token segments are
both *shorter* than `LIMIT=8192`, so a segment never fills its context. The penalty mixes
one cold start per segment with less available context per segment. On a file where each
segment exceeds `LIMIT`, only the first applies — which predicts further shrinkage, and
that is a **prediction, not a measurement.**

**This removes the largest blocker on the roadmap.** A verified full-`enwik8` round-trip
was costed at 17 days and declared infeasible, blocked behind multi-week int8 determinism
work. At 10.59× it is roughly a weekend, and needs no int8 at all.

<details><summary>the original plan, for the record</summary>

### 3. Build lockstep decode properly — **days** ⭐ the only novel idea here

"Decoding cannot be batched" is true *within* one stream and false *across* streams. A
scratch pilot already round-trips **byte-exactly** at S=4/8/16 and decodes **2.5–8.8×**
faster, because encoder and decoder run the same step-major loop — precisely what
`compress_batched` got wrong.

| S | bpb | decode | vs sequential |
|---:|---:|---:|---|
| 1 | 0.989 | 76 B/s | — |
| 4 | 1.077 | 187 B/s | 2.5×, +8.9% size |
| 16 | 1.350 | 672 B/s | 8.8×, +36.5% size |

The predicted <0.5% ratio cost was wrong, but the control explains it: **4× the segment
length cut the penalty 3.5×** (8.9% → 2.6%). The cost scales with *segment length*, not
with S — so S is close to free on large files, and enwik8 at S=16 gives ~1.7M-token
segments. Not extrapolated further on purpose; trap 3 is about exactly that.

This converts item 3.1 (full-enwik8 verified round-trip) from **17 days, declared
infeasible** to roughly a weekend — **without** the multi-week int8 kernel project it was
supposedly blocked behind.

</details>

### 3b. Now actually run the full-enwik8 verification — **~a weekend**

Unblocked by the item above and by one thing that turned out not to be true. Run it at
S=16 with segments long enough that the ratio cost is the cold-start term only, and
**measure** the penalty there rather than inheriting either the 3.97% or the pilot's
discredited 2.6%.

**The blocker found 2026-08-31, by reading rather than running.** `_LockPredictor.feed`
rebuilds the cache with an `[S, WINDOW]` forward when segments pass `LIMIT`, and
`transformers` materialises logits for every position unless `logits_to_keep` is set.
That is `S × 4096 × 49,152 × 4` bytes — 0.8 GiB at S=1, 3.0 at S=4, **12.0 GiB at S=16**,
on a box with ~2.4 GB free. Every lockstep test so far was 8 KB or 32 KB, where a segment
at S>1 is 140–2,221 tokens and never reaches `LIMIT`, so the branch had **only ever run at
S=1**. The first run large enough to slide is exactly this one.

Fixed with `logits_to_keep=1` on both slide forwards: 12.0 GiB → 3.0 MiB. Verified at
`LIMIT=128` on 4,096 B, which slides sixteen times per segment — round-trip ok sequential,
S=1 and S=4. **Treat it as a format change**: the streams measured byte-identical, but the
logits underneath differ by ~4e-5 and ~7% of probability buckets move, which is the same
mechanism that made `compress_batched` undecodable. Encoder and decoder must match.

**Do the 1 MB rehearsal first — ~1 hour.** `enwik8_1m` at S=16 gives 20,057-token segments,
comfortably past `LIMIT=8192`, so it is the first run where segments exceed the context
window. It is the only test that turns "the penalty approaches zero on big files" from a
prediction into a measurement, and it exercises the fixed slide path at scale before a
weekend is committed to it.

### Do not start with int8

Today's repro: `torch.ao.quantization.quantize_dynamic` fails both requirements. It computes
activation scales *per batch*, so the same row differs alone vs in a batch (max abs diff
**0.067**) — the exact failure class that broke `compress_batched`, surviving quantisation
untouched — and it is not faster here either (onednn-only engine, gets *slower* with more
threads). Real determinism needs static export-time scales and pinned integer kernels:
weeks. Price that against lockstep first.

---

## The pattern worth naming

Every significant win this project produced came from **measuring something nobody had
measured**, not from being clever:

- Swapping the model: **45%**. Every hand-built modelling improvement combined: **~1%**.
- An unswept default cost **2.07×** on every run, for the project's entire life.
- The audit caught two of this repo's own instruments producing invalid evidence.
- Then re-measurement caught **the audit itself**: its `ptc` rewrite measured 1.35× when run
  A-then-B and **1.00×** when interleaved on an idle machine, and its <0.5% lockstep cost
  measured 8.9%.

Three layers deep, each one found something real at the layer above. The cleverness kept
being wrong; the discipline kept being right. That is the transferable result, and it is
worth more than the bpb figure.
