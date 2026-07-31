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
(`llm_ptc.py`) — over one shared, verified arithmetic coder. Then Part 2 abandons better
modelling for cheap **structural** transforms (`shapes/`). A harness (`bench.py`) validates
itself against published figures before reporting anything.

## The honest state, up front

**This is a well-measured reproduction, not a contribution.** `llm_ptc` rebuilds what
ts_zip shipped in 2023; our 17.8% margin over it is a *model-vintage artifact* (we used a
2024 model against their 2023 one), not an algorithmic advance. Nothing here is novel.

**What is genuinely worth keeping** is the method and the negative results: a harness that
reproduces published `xz` figures to three decimals, nine refuted predictions with
measurements, and two instrument bugs caught by sanity checks rather than luck.

**Effort-to-result is humbling and worth internalising before you plan work:** swapping the
model was worth **45%**; every hand-built modelling improvement combined is worth about
**1%**; and in Part 2 a *codec flag* beat an afternoon of transform code, 32% to 18%.

## What works, and what is merely measured

| claim | status |
|---|---|
| `ptc.py` lossless on arbitrary bytes incl. binary, up to 1,029,744 B | **verified** |
| `ptc.py` → 3.1× on text (2.606 bpb) | **verified** |
| `llm_ptc` → 8.5× on alice29 (0.939 bpb) | **measured, not decoded** — see below |
| `llm_ptc` round-trip | **verified only to 4,096 B** |
| SQL dump transform → −17.8% vs `xz` | **verified**, round-trip asserted every run |
| SQLite page grouping → −0.4% | verified, and a **dead end** |
| OCI layer → −32.3% | **re-encode**, not byte-lossless. Identical tar, new digest. |
| enwik8 → 0.917 bpb | 262 KB slice only. Full-file figures are **extrapolations**. |

**The 8.5× headline came from `compress_batched`, which has never been decoded.** It is 16×
faster and produced byte-identical output on an 8 KB sample, but its stream does **not**
round-trip through the sequential decoder (diverges at byte 253 — batched and single-token
GEMMs reduce floats in a different order). Treat 0.939 as a measured entropy.

## In flight right now

A full sequential encode+decode of `alice29.txt` (task `bzdrzq9j0`, pid 5428). At the time
of writing: encode 40,000/41,933 tokens, tracking **0.935 bpb** — which matches the batched
0.939 closely and suggests the batched path is faithful. The decode half prints nothing, so
silence is expected, not a hang.

**When it finishes:** if `round-trip: ok`, update `results/results.json`, regenerate charts,
and close item **D.4** in `Long_Time_Tests.md` — the README headline is currently the
*pre-match-model* figure and is not reproducible with the shipped defaults.

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
  regenerated by `scripts/fetch_corpus.py` and `scripts/fetch_shape_data.py`.
- **`results/results.json` is the single source of truth.** Charts are generated from it;
  the README quotes it. Add measurements there, then run `scripts/make_charts.py`. Don't
  hand-edit numbers into the README.
- **Re-verify round-trip after any model or coder change** before starting a long run. A
  26-hour encode of a broken codec is the worst outcome available.

## Getting running from zero

```bash
pip install -r requirements.txt
python scripts/fetch_corpus.py        # Canterbury, Calgary, enwik8, 2026 control text
python scripts/fetch_shape_data.py    # Docker layer, SQLite dbs, SQL dumps

python bench.py                       # ~2 min. Confirms harness reproduces published xz.
python shapes/baseline.py sql         # seconds. Part 2 baselines.
python shapes/sqldump.py shapes/data/chinook.sql    # the 17.8% win

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
  *reach* the fields: plain text in a SQL dump (14–18%), unreachable behind SQLite's binary
  record format (0.4%).
- **The Hutter Prize is structurally closed to pretrained models** — entrants are charged
  for the decompressor, so a 272 MB model can't chase a 110 MB record. Only online-trained
  models qualify. Don't plan around it; I wasted several planning cycles before checking.

## What to do next

`Long_Time_Tests.md` has everything costed and ordered. The three that matter:

1. **Finish/confirm the in-flight verification** (D.4) — closes the biggest credibility gap.
2. **Record-level SQLite columnarisation** (item 4.4) — the only Part 2 follow-up with a
   *proven* mechanism behind it. Days of work.
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
- **Repo name.** Part 2 has nothing to do with LLMs. The unifying theme is "measure the
  incumbent before inventing." One `gh repo rename` if you want the name to say that.
