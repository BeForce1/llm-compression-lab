# Where this stands, and what to do next

Written 2026-08-01, after a full audit of the codebase. Companion to
[`handoff.md`](handoff.md) (orientation, traps, hard rules) and
[`Long_Time_Tests.md`](Long_Time_Tests.md) (everything costed and ordered).

---

## What we can actually claim

**0.940 bpb on `alice29.txt`, verified byte-exact round-trip on the whole file.** That is
8.5×, or 2.71× smaller than `xz -9`, and 17.7% under ts_zip's published figure on the
identical file.

Three caveats that are load-bearing, not throat-clearing:

1. **The margin over ts_zip is a model-vintage artifact.** We used a 2024 model against
   their 2023 one. It is not an algorithmic advance and should never be presented as one.
2. **"Verified" means this machine.** Same thread count, same library versions. It is a
   verified round-trip, not a portable format, and no amount of further running fixes that.
3. **The break-even is unreachable.** A 272 MB model repays itself after ~1.35 GB of text,
   which at ~85 B/s takes months to read back. Quote the ratio with that attached.

The honest summary is the one `handoff.md` already gives: **a well-measured reproduction,
not a contribution.** What is genuinely worth keeping is the method and the negative
results — ten refuted predictions with measurements, and instrument bugs caught by sanity
checks rather than luck.

---

## What the audit changed (2026-07-31 → 08-01)

| | before | after |
|---|---|---|
| encode **and** decode throughput | baseline | **2.07× faster** |
| known silent-corruption / crash bugs | 9 | 0 |
| refuted predictions on record | 9 | 10 (+3 moved to sql-compression) |
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

### 1. Sweep the context length — **~1 hour** ⭐ best value per hour

`LIMIT` defaults to 1024. SmolLM2-135M natively supports **8192**. The sweep behind the
"+3%, context barely matters" conclusion stopped at 1024, on a 4,096-byte sample that
tokenises to 1,098 tokens — so the 1024 row **never slid even once**, and the curve was
still descending at the last point (1.024 → 0.999 → 0.990).

```bash
for L in 2048 4096 8192; do
  LLM_PTC_LIMIT=$L python llm_ptc.py corpus/enwik8_mid 262144 enc batch
done
```

Same sample size every run (trap 2). Expect +1–3% bpb for ~+17% time at 2048. **This was
not runnable before the `WINDOW` fix** — `LLM_PTC_LIMIT` above the model max crashed or
silently ran a full-window forward per token.

### 2. Run the model A/Bs — **~1 hour each**

The model is worth 45%; everything hand-built is worth ~1%. Three queued, all now unblocked
by the `add_special_tokens` fix:

- `Qwen/Qwen3-0.6B-Base` — settles the base-vs-instruct hypothesis the README states as
  untested. Either confirms a useful rule or removes a claim.
- `HuggingFaceTB/SmolLM2-360M` — compare against **0.939** (the batched figure), not the
  0.940 headline; that command runs `batch` and mixing paths manufactures a 0.001 difference
  that is the encoder, not the model.
- `google/gemma-3-270m` — the counter-test to the "big vocab wastes capacity" hypothesis.
  Needs a detokenise pre-flight; it is SentencePiece, not byte-level BPE.

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
