# Handoff

For whoever picks this up next, including future-me. Read this before running anything.

**Repo:** `github.com/BeForce1/llm-compression-lab` (private)
**Working copy:** `C:\Users\aregm\personal\llm-compression-lab`
**Started:** 2026-07-30. **This doc:** 2026-07-31.

---

## What this is in one paragraph

Compression is prediction: an arithmetic coder spends `-log2(p)` bits per outcome, so a
file's compressed size *is* the model's total surprise. The repo builds that twice — once
with a hand-built context-mixing model (`ptc.py`), once with a pretrained language model
(`llm_ptc.py`) — over one shared, verified arithmetic coder. A harness (`bench.py`)
validates itself against published figures before reporting anything. The format-aware
transforms that used to sit here as "Part 2" are now a separate repo,
[sql-compression](https://github.com/BeForce1/sql-compression); nothing in here depends on
them.

## The honest state, up front

**This is a well-measured reproduction, not a contribution.** `llm_ptc` rebuilds what
ts_zip shipped in 2023; our 17.7% margin over it is a *model-vintage artifact* (we used a
2024 model against their 2023 one), not an algorithmic advance. Nothing here is novel.

**What is genuinely worth keeping** is the method and the negative results: a harness that
reproduces published `xz` figures to three decimals, nine refuted predictions with
measurements, and two instrument bugs caught by sanity checks rather than luck.

**Effort-to-result is humbling and worth internalising before you plan work:** swapping the
model was worth **45%**; every hand-built modelling improvement combined is worth about
**1%**; and in the transform work now split into sql-compression, a *codec flag* beat an
afternoon of transform code. Nothing you build by hand here is likely to be worth what
choosing a better model is worth — spend the time on the model, or on measuring what you
already have.

## What works, and what is merely measured

| claim | status |
|---|---|
| `ptc.py` lossless on arbitrary bytes incl. binary, up to 1,029,744 B | **verified** |
| `ptc.py` → 3.1× on text (2.606 bpb) | **verified** |
| `llm_ptc` → 8.5× on alice29 (0.940 bpb) | **verified** — full-file sequential round-trip |
| `llm_ptc` round-trip | **verified to 152,089 B**, one machine only — see below |
| SQL dump transform → −28.6% chinook / −22.2% wiki_meta | **verified**, round-trip asserted every run, plus a `selfcheck()` over 12 dump shapes |
| SQLite page grouping → −0.5% | verified, and a **dead end** |
| OCI layer → −34.9% (`zstd -19 --long=27`) | **re-encode**, not byte-lossless. Identical tar, new digest. |
| enwik8 → 0.917 bpb | 262 KB slice only. Full-file figures are **extrapolations**. |

**`compress_batched` still has never been decoded, and the headline no longer depends on
it.** It is 16× faster and produced byte-identical output on an 8 KB sample, but its stream
does **not** round-trip through the sequential decoder (diverges at byte 253 — batched and
single-token GEMMs reduce floats in a different order). Keep it for measurement, not output.

## The verification landed

Done, 2026-07-31: a full sequential encode **and decode** of `alice29.txt` (task
`bzdrzq9j0`, pid 5428, ~3 hours) returned `round-trip: ok` at **0.940 bpb / 17,873 B**.
The README headline, charts and `results/results.json` were updated to the verified figure
and item **D.4** is closed.

Three things to carry forward:

- **The batched path was faithful** — 27 bytes, 0.15% from the verified figure. It was
  never decodable, but it was not inflating anything either.
- **Verified ≠ portable.** Same machine, same thread count, same library versions. The next
  real step is still deterministic int8 inference (item 4.1), not another long run.
- **Ignore the 30 B/s encode / 42 B/s decode it printed.** Heavy `xz`/`zstd` benchmarking
  ran on this machine through the decode half — trap 6 below, committed *again*, by the
  person who wrote trap 6. The bpb is deterministic and unaffected.

## Traps that will bite you

These all cost me real time. They are the most valuable part of this document.

1. **Slice `enwik8` mid-file.** The first 256 KB is XML preamble and `<siteinfo>` and scores
   **0.811 vs 0.917** mid-file — a 13% bias that nearly went into the README. Use
   `corpus/enwik8_mid`.
2. **Match sample sizes exactly when comparing configs.** Our bpb *improves* with file size,
   so a bigger sample flatters the newer run and manufactures wins that aren't there.
3. **Never extrapolate throughput from small samples.** I estimated a 85-minute run from
   2–8 KB measurements; it is taking 3× that. On small samples the context window is mostly
   *empty*, so per-token attention is cheap. Real files sit at full context.
4. **A surprising measurement is usually a broken instrument.** Twice confirmed. `probe.py`
   once reported that tokens *after* a window slide were cheaper than deep-context ones —
   the bug was that its "cold" bucket held the *start of the file*, and GPT-2 has Alice's
   opening lines memorised.
5. **bf16 is slower here, not faster.** No AVX512-BF16 on this CPU, so torch converts to
   fp32 per matmul and you pay conversion on top. 64 vs 84 B/s. Don't "optimise" it back.
6. **CPU contention corrupts timings, not ratios.** Ratios are deterministic; any MB/s
   figure measured while another job runs is fiction. I polluted my own numbers this way.
7. **`llm_ptc` is text-only** — it calls `.decode('utf-8')` and rejects binary outright.
8. **SSE/APM and extra match orders are already tested and rejected.** Neutral to harmful
   against a well-calibrated LLM. Losing configs are recorded in comments next to `ORDERS`
   in `llm_ptc.py`. Don't re-run them.
9. **A default is not a measurement.** The thread count was `os.cpu_count()` from day one and
   the README called the defaults "the measured optimum". It had never been swept, and it was
   the **slowest** of five settings — 42 B/s vs 87 at 6 threads. Every encode and decode in
   this project's history, including the 3-hour full-file verification, paid 2.07× for it.
   Anything you inherited rather than measured is a candidate.
10. **A speedup measured as "run A, then run B" is measuring cache state.** The audit's
   1.35×/1.41× `ptc` rewrite measured **1.00×/1.07×** when re-run interleaved (A,B,A,B) on an
   idle box, and was reverted. Interleave, take fastest-of-N, and run nothing else.

## Hard rules for this repo

- **No Claude co-author in commits.** Owner's explicit requirement. No `Co-Authored-By`
  trailer, no "Generated with" line. Audited clean across all history.
- **Commits must use the GitHub noreply email.** The account blocks pushes exposing its
  real address. `user.email` is set **locally in this repo** to
  `80689854+BeForce1@users.noreply.github.com`. Global git config is untouched — don't
  "fix" the local override.
- **Never commit corpus or test data.** Two reasons, both real: they are other people's
  content under their own licences (this repo is MIT — redistributing would be
  relicensing), and the 2026 arXiv control text carries its author's contact email. An
  earlier commit did include it and had to be purged with a force-push. Everything is
  regenerated by `scripts/fetch_corpus.py`.
- **`results/results.json` is the single source of truth.** Charts are generated from it;
  the README quotes it. Add measurements there, then run `scripts/make_charts.py`. Don't
  hand-edit numbers into the README.
- **Re-verify round-trip after any model or coder change** before starting a long run. A
  26-hour encode of a broken codec is the worst outcome available.

## Getting running from zero

```bash
pip install -r requirements.txt
python scripts/fetch_corpus.py        # Canterbury, Calgary, enwik8, 2026 control text

python bench.py                       # ~2 min. Confirms harness reproduces published xz.

export LLM_PTC_MODEL=HuggingFaceTB/SmolLM2-135M
python llm_ptc.py corpus/alice29.txt 4096           # ~2 min, round-trip verified
```

**Sanity check that you're set up correctly:** `bench.py` must print `xz -9` at **2.551**
bpb on alice29 and **2.717** on book1, matching the published column. If it doesn't, your
corpus files are wrong and every other number will be too.

## Findings you must not re-derive

All in `results/results.json` with numbers; the short version:

- **The model is the whole game.** SmolLM2-135M beats GPT-2 124M by 45.6% at the same
  parameter count — data, not size. And it beats **Qwen3-0.6B**, which is 4.4× larger, on
  every axis. Untested hypothesis: base models compress better than instruct-tuned ones,
  because post-training decalibrates the next-token distribution. Testable with
  `Qwen3-0.6B-Base` (item 1.3).
- **The arithmetic coder wastes <1%.** So a C rewrite buys *ratio* nothing. Speed still
  matters under a fixed time budget, but that's a different argument.
- **12-bit quantisation makes GPT-2 5.3% smaller** by smoothing an overconfident model.
  Worth ~0 against well-calibrated SmolLM2.
- **Context length is worth ~0% for GPT-2 and +3% for SmolLM2.** Don't assume from one model.
- **Columnar regrouping is the transferable win**, and it materialises only where you can
  *reach* the fields: plain text in a SQL dump (14–18% from the regrouping alone),
  unreachable behind SQLite's binary record format (0.4%). Regrouping is the smaller half —
  re-spelling *values* (delta for ID ramps, byte-planes for wide numbers, epoch seconds for
  ISO-8601 timestamps, picked per column) took chinook 17.8% → 28.6% and wiki_meta
  17.4% → 22.2% with no change to the grouping.
- **Don't predict which encoding wins; measure it.** Delta beats byte-planes 45 B to
  328 B on a monotonic ID column and loses 11,594 to 8,510 on a wide value column, and
  plain ASCII beats both on some. A rule would mis-call the columns where they're close.
- **The wiki.sql −0.1% never tested the hypothesis it was cited for.** Only 1,176 of 68,576
  lines match the line-based INSERT parser (SQLite `.dump` emits raw newlines inside string
  values), so 97.7% of the file bypasses the transform entirely. `wiki_meta` is the valid
  controlled test; wiki.sql is a parser-coverage measurement wearing a shape-argument label.
- **The Hutter Prize is structurally closed to pretrained models** — entrants are charged
  for the decompressor, so a 272 MB model can't chase a 110 MB record. Only online-trained
  models qualify. Don't plan around it; I wasted several planning cycles before checking.

## What to do next

`Long_Time_Tests.md` has everything costed and ordered. With D.4 closed, the three that matter:

1. **Multi-stream lockstep decode** (item 4.6) — new, and now the best-value item on the
   list. A scratch pilot already round-trips byte-exactly at S=4/8/16 with **2.5–8.8× decode**,
   and the ratio cost falls with segment length (8.9% at 560-token segments, 2.6% at 2,221),
   so it is close to free on large files. This is what makes a full-enwik8 *verified*
   round-trip feasible without solving int8 determinism first.
2. **Record-level SQLite columnarisation** (item 4.4) — rescoped: OLTP-shaped dbs only
   (~10% ceiling; wiki.db's is ~0.1%), and build the churn fixture first, because all three
   sample dbs have zero freeblocks/overflow/stale bytes and cannot catch a rebuild bug.
   `encode_col` is reusable as-is, and rowid ramps are exactly what delta is for.
3. **Deterministic int8 inference** (item 4.1) — the sole path from "measurement demo" to
   "working codec". Collapses a 17-day verified round-trip to ~2 hours and makes
   cross-machine decompression possible.

**Explicitly not worth doing:** a C port of `ptc` (its ratio already loses to `bz2`, so
speed isn't the constraint), SSE/APM (tested, harmful), more match orders (0.2% for 25%
speed), and chasing full-`enwik8` before the 1-hour trend check in item 1.2.

## Decisions for the owner

- **Publish public?** The repo is private. It's above-average as a portfolio piece
  *specifically because* it leads with refuted predictions. If it goes public, the
  `ff4501d` orphan commit is unreachable but may survive GitHub GC — delete-and-recreate is
  the belt-and-braces option.
- **Spend 26 hours on full enwik8?** It makes the ts_zip comparison a measurement rather
  than an extrapolation, but that comparison is a model-vintage artifact either way. Low
  value for the cost.
- ~~**Repo name.**~~ Resolved by the split: the non-LLM half is its own repo now, so this
  one's name finally describes its contents.
