"""A language model as the predictor, ptc's arithmetic coder as the mouth.

Same architecture as ptc - predictor feeds probabilities to a coder that spends
-log2(p) bits per outcome. We swapped four hash tables for a pretrained model.
bench.py established the coder was never the bottleneck (<1% waste).

Two predictors, blended the way ptc blends its own:
  LLM    - the pretrained model's next-token distribution
  match  - long-range exact repeats over the whole token stream, which the LLM
           cannot see past its LIMIT-token window

The match model is the ONLY added predictor that pays (-1.2%), and it pays
because it supplies information the LLM structurally cannot have. The rest of
the context-mixing toolbox - extra match orders, SSE/APM - measured neutral to
harmful here: that machinery exists to prop up weak predictors, and this
predictor is not weak. See the numbers next to ORDERS below.

They are mixed in the LOGISTIC domain with weights learned online, starting at
full trust in the LLM and zero in the match. That matters: naive linear
interpolation of a point mass into the LLM's distribution measured 2% WORSE on
enwik8 (0.833 vs 0.817), because it steals mass from the LLM even when the LLM
is already right. A learned mixer weights a useless input out instead.

The coder is reused unchanged by binarising the token id: VBITS binary
decisions walk down the id space, each probability read off the cumulative
distribution. Exact, and no multi-symbol coder to get wrong.

ponytail: v1 does not mix in ptc's byte-level hash contexts - that needs
        marginalising a multi-byte tokenizer over bytes, which is real work.
ponytail: same-process round-trip only. Cross-machine needs quantised weights
        and pinned kernels (ts_zip does this); float32 on one box is
        reproducible enough to measure a ratio.
"""
import os
import sys
import time
from array import array

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ptc

MODEL = os.environ.get('LLM_PTC_MODEL', 'gpt2')
VBITS = 16                            # set by _load() from the model's vocab size
SCALE = 1 << 30                       # integer resolution of the distribution
# Context budget. _load() clamps it to the model's own maximum, so 8192 reads as
# "all the context this model has": gpt2 gets 1024, SmolLM2 8192. Swept on
# SmolLM2-135M, batched encode, bpb by LIMIT:
#                       1024    2048    4096    8192
#   alice29 (memorised) 0.958   0.946   0.938   0.934    -2.5%
#   post2026 (unseen)   1.331   1.260   1.220   1.181   -11.3%
# The old default of 1024 rested on a sweep that found context worth -0.7% - on
# GPT-2, whose maximum IS 1024, so it measured a ceiling and called it a property
# of context. On unseen text the curve has not converged even at 8192.
# It costs the sequential path: 70 -> 49 B/s encode, 66 -> 22 B/s decode over
# 40,960 B (decode figure measured under contention, so treat it as a bound).
# Taken anyway: ratio is this codec's only deliverable, its speed is already far
# past unusable, and the batched measurement path is unaffected (flat, 585 -> 582).
# ponytail: the batched path materialises LIMIT x vocab x 4 bytes of logits -
#         1.6 GB for SmolLM2 here, but 5.0 GB for Qwen3's 151,936 vocab. Set
#         LLM_PTC_LIMIT down for big-vocab models rather than growing a
#         memory-aware default before anything needs one.
LIMIT = int(os.environ.get('LLM_PTC_LIMIT', 8192))
WINDOW = int(os.environ.get('LLM_PTC_WINDOW', LIMIT // 2))   # re-derived in _load()
_WINDOW_SET = 'LLM_PTC_WINDOW' in os.environ          # honour an explicit choice
# Measured on a representative enwik8 slice (262,144 B, mid-file):
#   LLM only ............................ 0.928 bpb   1088 B/s
#   + match order 4 ..................... 0.917       1056      <- default, best trade
#   + match orders 2,4,6,8 .............. 0.915        789      +0.2% for -25% speed
#   + APM/SSE at 25% weight ............. 0.918       1062      no help
#   + APM/SSE at 75% weight ............. 0.926        887      actively harmful
ORDERS = [int(x) for x in os.environ.get('LLM_PTC_ORDERS', '4').split(',') if x]
BATCHED_FLAG = 1 << 31                # header bit: measured by compress_batched
LOCKSTEP_FLAG = 1 << 30               # header bit: S-segment stream, decodable
MAXLEN = 31                           # match length doubles as a confidence bucket
MMASK = (1 << 22) - 1                 # match index size
# OFF by default, and that is a result not an oversight: SSE exists to fix a
# miscalibrated mixer, and a well-calibrated LLM has nothing to recalibrate, so
# the APM's own estimation noise costs more than it saves. Tested at two weights.
APM_ON = os.environ.get('LLM_PTC_APM', '0') != '0'
APM_W = int(os.environ.get('LLM_PTC_APM_W', 3))   # APM share of the final p, /4
LR = 12                               # mixer learning rate (right-shift)
RATE = 5                              # match counter adaptation (right-shift)
_LOADED = {}


def _load():
    if MODEL not in _LOADED:
        # NOT os.cpu_count(). A single-token step is ~210 tiny GEMVs, each its own
        # parallel region, and the work is memory-bandwidth-bound - so past ~6
        # threads you buy only fork-join barriers and hyperthread contention.
        # Measured sequential encode on this 20-core box, fastest of each:
        #   20 threads 42 B/s | 10: 82 | 8: 86 | 6: 87 | 4: 83
        # The old default was therefore the SLOWEST setting on the sweep, by 2.07x.
        # Sequential streams measured interchangeable across thread counts at
        # 3 KB - byte-identical output, and cross-decode ok in both directions.
        # Do NOT generalise that: the BATCHED path at 16 KB does differ by thread
        # count (1,990 vs 1,992 B), so reduction order is visible there. Treat
        # this as "safe for the sizes tested", and re-verify before a long run.
        torch.set_num_threads(int(os.environ.get(
            'LLM_PTC_THREADS', min(6, os.cpu_count() or 4))))
        torch.set_grad_enabled(False)
        # These checkpoints ship in 16-bit (272 MB / 135M params = 2.0 B/param), but
        # float32 measured FASTER here: this CPU has no AVX512-BF16, so torch
        # converts bf16 to fp32 per matmul anyway and we pay the conversion.
        dtype = getattr(torch, os.environ.get('LLM_PTC_DTYPE', 'float32'))
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
        model.eval()
        # Token ids must fit the binarised id space, and gpt2's 50257 is not a
        # safe assumption - Qwen3 has 151936, which needs 18 bits not 16.
        global VBITS, LIMIT, WINDOW
        VBITS = max(1, (model.config.vocab_size - 1).bit_length())
        LIMIT = max(2, min(LIMIT, getattr(model.config, 'max_position_embeddings', LIMIT)))
        # WINDOW is derived from LIMIT at import, BEFORE the clamp above knows the
        # model. Re-derive it here or the slide stops shrinking the context: with
        # gpt2 and LLM_PTC_LIMIT=4096 the rebuild asks for position 1024 of a
        # 1024-entry table (crash), and with WINDOW == LIMIT every token past the
        # limit triggers a full-window forward (~100x slower, silently).
        WINDOW = max(1, min(WINDOW if _WINDOW_SET else LIMIT // 2, LIMIT - 1))
        _LOADED[MODEL] = (tok, model)
    return _LOADED[MODEL]


class Predictor:
    """Wraps the model so it looks like ptc's Model: feed a symbol, get a
    distribution for the next one. Holds the KV cache so each step is O(1)."""

    def __init__(self):
        self.tok, self.model = _load()
        self.V = self.model.config.vocab_size
        self.cdf = np.zeros((1 << VBITS) + 1, dtype=np.int64)
        self.prob = None
        self.past = None
        self.ids = []

    def feed(self, tid):
        """Advance one token; refresh the distribution for the next one.

        Models with absolute position embeddings die past LIMIT tokens, so on
        overflow we rebuild the cache from the last WINDOW tokens. Depends only
        on the token sequence, so both sides slide at exactly the same place.
        """
        self.ids.append(tid)
        if len(self.ids) > LIMIT:
            # logits_to_keep=1 slices hidden_states before lm_head, so the slide
            # allocates one position of logits instead of WINDOW. That is 0.8 GiB
            # -> 0.2 MiB here, and it is what makes the lockstep slide possible at
            # all: S x WINDOW x 49,152 x 4 B is 12.0 GiB at S=16 on a 15.7 GB box.
            # NOT backward compatible, and do not assume it is: the last-row logits
            # differ by ~4e-5 between the two shapes and ~7% of 30-bit probability
            # buckets move, which is the same reduction-order effect that made
            # compress_batched undecodable. It measured byte-identical on 4,096 B
            # at LIMIT=128 - that is one sample agreeing, not a guarantee.
            self.ids = self.ids[-WINDOW:]
            out = self.model(input_ids=torch.tensor([self.ids]), use_cache=True,
                             logits_to_keep=1)
        else:
            out = self.model(input_ids=torch.tensor([[tid]]),
                             past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.prob = torch.softmax(out.logits[0, -1].double(), -1).numpy()


class _Match:
    """One exact-repeat predictor at a fixed context order."""
    __slots__ = ('order', 'table', 'mtab', 'ptr', 'len', 'idx')

    def __init__(self, order):
        self.order = order
        self.table = array('i', [0]) * (MMASK + 1)   # context hash -> next position
        self.mtab = array('H', [ptc.ONE >> 1]) * (2 * (MAXLEN + 1))
        self.ptr = 0
        self.len = 0
        self.idx = -1


class MatchBank:
    """Exact-repeat predictors at several context orders, over one shared token
    history.

    The LLM only sees LIMIT tokens. These reach back across the whole file,
    where wiki markup keeps its repeated templates, tags and link syntax -
    structure xz eats wholesale and a bounded-context model cannot see at all.

    Long orders fire rarely but confidently, short orders fire constantly and
    are often wrong. Nobody tunes that tradeoff: each reports a per-BIT opinion
    through a counter learned on (match length, expected bit), and the mixer
    learns how far to trust each one.
    """

    def __init__(self, orders):
        self.hist = array('i')
        self.models = [_Match(o) for o in orders]

    def _hash(self, order):
        h = 0
        for t in self.hist[-order:]:
            h = (h * 0x100000001B3 ^ t) & 0xFFFFFFFFFFFFFFFF
        return (h >> 20) & MMASK

    def opinions(self, lo, hi):
        """One 12-bit probability per model; None where it has nothing to say."""
        mid = (lo + hi) >> 1
        h, n, out = self.hist, len(self.hist), []
        for m in self.models:
            if m.len and m.ptr < n:
                t = h[m.ptr]
                if lo <= t < hi:                 # still consistent with bits so far
                    m.idx = min(m.len, MAXLEN) * 2 + (1 if t >= mid else 0)
                    out.append(m.mtab[m.idx])
                    continue
            m.idx = -1
            out.append(None)
        return out

    def learn(self, bit):
        target = bit << ptc.BITS
        for m in self.models:
            if m.idx >= 0:
                v = m.mtab[m.idx] + ((target - m.mtab[m.idx]) >> RATE)
                m.mtab[m.idx] = 1 if v < 1 else ptc.ONE - 1 if v > ptc.ONE - 1 else v

    def best_len(self):
        return max((m.len for m in self.models), default=0)

    def push(self, tid):
        h = self.hist
        n = len(h)
        for m in self.models:
            if m.len and m.ptr < n:
                if h[m.ptr] == tid:
                    m.ptr += 1
                    m.len = min(m.len + 1, MAXLEN)
                else:
                    m.len = 0
        h.append(tid)
        n += 1
        for m in self.models:
            if n >= m.order:
                k = self._hash(m.order)
                if not m.len:
                    cand = m.table[k]
                    if cand:
                        m.ptr, m.len = cand, 1
                m.table[k] = n


class Mixer:
    """Blends N predictions per binary decision, in the logistic domain, with
    weights learned online.

    Initialised to full trust in input 0 (the LLM) and zero elsewhere, so it
    starts exactly equal to LLM-only and can only improve: a useless input gets
    weighted out rather than dragging the distribution around. Naive linear
    interpolation lacks that property and measured 2% WORSE on enwik8.
    """

    def __init__(self, n):
        self.w = [1 << 16] + [0] * (n - 1)
        self.st = [0] * n
        self.p = ptc.ONE >> 1

    def mix(self, probs):
        total = 0
        for i, p in enumerate(probs):
            s = ptc._STRETCH[p] if p is not None else 0
            self.st[i] = s
            total += self.w[i] * s
        self.p = ptc._squash(total >> 16)
        return self.p

    def update(self, bit):
        err = (bit << ptc.BITS) - self.p
        for i in range(len(self.w)):
            self.w[i] += (self.st[i] * err) >> LR   # st = 0 means no vote, no update


class APM:
    """Adaptive probability map, a.k.a. SSE. Takes the mixer's output and
    recalibrates it against a context by interpolating between learned buckets
    in the stretched domain.

    This is the stage every serious context-mixing compressor has and ptc does
    not. It exists because a mixer's output is systematically miscalibrated in
    ways that depend on state - here, on how long the current match is and how
    deep into the token we are. Initialised to the identity so it starts neutral.
    """

    BUCKETS = 33                       # 32 stretch intervals plus the right edge

    def __init__(self, n_ctx):
        self.t = array('H', [0]) * (n_ctx * self.BUCKETS)
        for c in range(n_ctx):
            base = c * self.BUCKETS
            for b in range(self.BUCKETS):
                self.t[base + b] = ptc._squash((b - 16) * 128)
        self.i = 0
        self.frac = 0

    def refine(self, p, ctx):
        s = ptc._STRETCH[p] + 2048       # 1..4095
        self.i = ctx * self.BUCKETS + (s >> 7)
        self.frac = s & 127
        return (self.t[self.i] * (128 - self.frac)
                + self.t[self.i + 1] * self.frac) >> 7

    def update(self, bit):
        target = ptc.ONE - 1 if bit else 1
        for j in (self.i, self.i + 1):
            v = self.t[j] + ((target - self.t[j]) >> 6)
            self.t[j] = 1 if v < 1 else ptc.ONE - 1 if v > ptc.ONE - 1 else v


def _build_cdf(cdf, prob, vocab):
    """Integer CDF over the model's distribution. Every token gets at least one
    count so that any token remains codable."""
    freq = np.maximum((prob * SCALE).astype(np.int64), 1)
    np.cumsum(freq, out=cdf[1:vocab + 1])
    cdf[vocab + 1:] = cdf[vocab]


def _split(cdf, lo, hi):
    """P(token lands in the upper half of [lo, hi)), as a 12-bit probability."""
    mid = (lo + hi) >> 1
    low = int(cdf[mid]) - int(cdf[lo])
    high = int(cdf[hi]) - int(cdf[mid])
    total = low + high
    if not total:
        return ptc.ONE >> 1, mid                 # unreachable, but identical both sides
    p = (high * ptc.ONE) // total
    return (1 if p < 1 else ptc.ONE - 1 if p > ptc.ONE - 1 else p), mid


def _code_token(coder, cdf, bank, mixer, apm, tid=None):
    """Walk the binarised id space. Encodes when tid is given, otherwise decodes
    and returns the token. One body for both directions, so they cannot drift."""
    lo, hi = 0, 1 << VBITS
    # Match state as of before this token - identical on both sides.
    mlen = min(bank.best_len(), 7) if bank is not None else 0
    for depth in range(VBITS):
        p_llm, mid = _split(cdf, lo, hi)
        if bank is not None:
            p = mixer.mix([p_llm] + bank.opinions(lo, hi))
            if apm is not None:
                # Blend rather than replace: hedges while the APM is still cold.
                p = (APM_W * apm.refine(p, mlen * VBITS + depth)
                     + (4 - APM_W) * p) >> 2
                p = 1 if p < 1 else ptc.ONE - 1 if p > ptc.ONE - 1 else p
        else:
            p = p_llm
        if tid is None:
            bit = coder.decode(p)
        else:
            bit = 1 if tid >= mid else 0
            coder.encode(bit, p)
        if bank is not None:
            mixer.update(bit)
            bank.learn(bit)
            if apm is not None:
                apm.update(bit)
        lo, hi = (mid, hi) if bit else (lo, mid)
    return lo


def _schedule(n):
    """For each token index i, where its context starts in the virtual sequence
    V = [eos, t0, t1, ...]. Replays feed()'s window bookkeeping with no model
    calls, so the batched encoder reproduces the decoder's contexts exactly."""
    s, out = 0, []
    for i in range(n):
        if i - s + 1 > LIMIT:
            s = i - WINDOW + 1
        assert s >= 0, f'negative context start: WINDOW={WINDOW} > LIMIT={LIMIT}'
        out.append(s)
    return out


def _progress(done, total, bits, data_len):
    seen = data_len * done / total
    print(f'  ... {done:,}/{total:,} tokens, {bits / seen:.3f} bpb so far',
          file=sys.stderr, flush=True)


def _encode_ids(tok, data):
    """Tokenise, and prove the tokeniser round-trips these exact bytes first.

    add_special_tokens would prepend a BOS for Gemma/Llama-lineage tokenisers,
    and decompress() renders every id back as text, so the output would gain the
    BOS's literal spelling. gpt2/SmolLM2/Qwen3 add nothing, which is why this was
    latent - it arms itself the moment the model is swapped, i.e. exactly when a
    long run is about to be paid for. The assert costs milliseconds.
    """
    text = data.decode('utf-8')
    ids = tok(text, add_special_tokens=False).input_ids
    assert tok.decode(ids, clean_up_tokenization_spaces=False) == text, (
        f'{MODEL} does not detokenise to its own input - not usable as a codec')
    return ids


def compress(data):
    if not data:
        return b''
    tok, _ = _load()
    ids = _encode_ids(tok, data)
    pred, enc = Predictor(), ptc.Encoder()
    bank = MatchBank(ORDERS) if ORDERS else None
    mixer = Mixer(1 + len(ORDERS)) if ORDERS else None
    apm = APM(8 * VBITS) if ORDERS and APM_ON else None
    prev = tok.eos_token_id
    chatty = len(ids) > 10000
    for i, tid in enumerate(ids):
        if chatty and i and not i % 2000:
            _progress(i, len(ids), 8 * len(enc.out), len(data))
        pred.feed(prev)
        _build_cdf(pred.cdf, pred.prob, pred.V)
        _code_token(enc, pred.cdf, bank, mixer, apm, tid)
        if bank is not None:
            bank.push(tid)
        prev = tid
    return len(ids).to_bytes(4, 'big') + enc.finish()


def compress_batched(data):
    """A RATIO MEASUREMENT, not a codec. One forward per WINDOW, 15.9x faster.

    Legal only when encoding: the whole token sequence is already known, and
    causal masking means position k's distribution depends only on 0..k. So one
    pass yields every distribution, reading the weights once per window rather
    than once per token.

    The stream it returns does NOT decode. Batched and single-token GEMMs reduce
    floats in a different order, the 12-bit quantisation does not absorb it, and
    a sequential decode diverges at byte 253. Measured against the verified
    sequential figure it is faithful to 0.15% - a good measurement and an invalid
    codec, until inference is integer. The header is flagged so decompress()
    refuses it rather than returning plausible garbage.
    """
    if not data:
        return b''
    tok, model = _load()
    # Opposite thread optimum to the sequential path: one forward per WINDOW is
    # a big GEMM that parallelises properly, so it wants every core. Measured on
    # 16 KB: 20 threads 967 B/s, 8: 830, 6: 828. _load() defaults to 6 for the
    # sequential codec, which would cost this path ~17%.
    if 'LLM_PTC_THREADS' not in os.environ:
        torch.set_num_threads(os.cpu_count() or 4)
    ids = _encode_ids(tok, data)
    virt = [tok.eos_token_id] + ids               # virt[k] predicts ids[k]
    starts = _schedule(len(ids))
    enc = ptc.Encoder()
    bank = MatchBank(ORDERS) if ORDERS else None
    mixer = Mixer(1 + len(ORDERS)) if ORDERS else None
    apm = APM(8 * VBITS) if ORDERS and APM_ON else None
    vocab = model.config.vocab_size
    cdf = np.zeros((1 << VBITS) + 1, dtype=np.int64)
    chatty = len(ids) > 10000

    i = 0
    while i < len(ids):
        s = starts[i]
        j = i
        while j + 1 < len(ids) and starts[j + 1] == s:
            j += 1                               # widest run sharing this context base
        logits = model(input_ids=torch.tensor([virt[s:j + 1]])).logits[0]
        for k in range(i, j + 1):
            _build_cdf(cdf, torch.softmax(logits[k - s].double(), -1).numpy(), vocab)
            _code_token(enc, cdf, bank, mixer, apm, ids[k])
            if bank is not None:
                bank.push(ids[k])
        i = j + 1
        if chatty:
            _progress(i, len(ids), 8 * len(enc.out), len(data))
    # Top bit of the count marks "measured, not decodable". Without it this blob
    # is indistinguishable from compress()'s and decompress() would happily
    # return wrong text for it.
    return (BATCHED_FLAG | len(ids)).to_bytes(4, 'big') + enc.finish()


class _LockPredictor:
    """S independent segments advanced together, one [S,1] forward per step.

    The batch dimension carries unrelated segments, not one sequence. Rows do not
    attend to each other, so a row's distribution is exactly what it would be
    alone AT THE SAME BATCH SHAPE - and shape is the thing that must not move,
    since it decides the GEMM's float reduction order. That is why a finished
    segment keeps being fed a filler token rather than being dropped from the
    batch: dropping it would reshape the GEMM and change the numbers for every
    surviving row.

    All S segments are fed one token per step from the same starting point, so
    they cross LIMIT on the same step and slide together, which keeps the KV
    cache rectangular. Segment lengths differ by at most 1 for the same reason.
    """

    def __init__(self, S):
        self.tok, self.model = _load()
        self.S = S
        self.past = None
        self.ids = [[] for _ in range(S)]

    def feed(self, toks):
        """Advance every segment one token; return an [S, vocab] distribution."""
        for s, t in enumerate(toks):
            self.ids[s].append(t)
        if len(self.ids[0]) > LIMIT:
            # See Predictor.feed: without logits_to_keep this materialises
            # [S, WINDOW, vocab] float32 - 12.0 GiB at S=16, on a box with ~2.4 GB
            # free. Every lockstep run so far was 8-32 KB, where a segment at S>1
            # is 140-2,221 tokens and never reaches LIMIT, so this branch had only
            # ever executed at S=1. Verified 2026-08-31 at LIMIT=128 on 4,096 B,
            # which slides 16 times per segment: round-trip ok at S=1 and S=4.
            self.ids = [x[-WINDOW:] for x in self.ids]
            out = self.model(input_ids=torch.tensor(self.ids), use_cache=True,
                             logits_to_keep=1)
        else:
            out = self.model(input_ids=torch.tensor([[t] for t in toks]),
                             past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return torch.softmax(out.logits[:, -1].double(), -1).numpy()


def _segments(n, S):
    """Start offset and length per segment, as equal as possible.

    Equal lengths are not cosmetic: they keep every segment sliding on the same
    step, and they bound the filler steps at the tail to at most one per segment.
    """
    base, rem = divmod(n, S)
    lens = [base + (1 if i < rem else 0) for i in range(S)]
    starts, acc = [], 0
    for L in lens:
        starts.append(acc)
        acc += L
    return starts, lens


def _lockstep(data_or_blob, S=None, decode=False):
    """One step-major loop, used for BOTH directions.

    That shared loop is the whole point. compress_batched encoded with a [1, W]
    forward and decoded with [1, 1], so the two sides reduced floats in different
    orders and diverged at byte 253. Here both sides run [S, 1] forwards, the same
    number of them, in the same order - so they see bit-identical probabilities
    and the stream actually decodes.
    """
    tok, model = _load()
    vocab = model.config.vocab_size
    if decode:
        blob = data_or_blob
        n = int.from_bytes(blob[:4], 'big') & ~LOCKSTEP_FLAG
        S = blob[4]
        sizes = [int.from_bytes(blob[5 + 4 * i:9 + 4 * i], 'big') for i in range(S)]
        off, coders = 5 + 4 * S, []
        for sz in sizes:
            coders.append(ptc.Decoder(blob[off:off + sz]))
            off += sz
        ids = [0] * n
    else:
        ids = _encode_ids(tok, data_or_blob)
        n = len(ids)
        coders = [ptc.Encoder() for _ in range(S)]
    if not n:
        return b''
    S = min(S, n)                                  # never more segments than tokens
    starts, lens = _segments(n, S)
    pred = _LockPredictor(S)
    banks = [MatchBank(ORDERS) if ORDERS else None for _ in range(S)]
    mixers = [Mixer(1 + len(ORDERS)) if ORDERS else None for _ in range(S)]
    apms = [APM(8 * VBITS) if ORDERS and APM_ON else None for _ in range(S)]
    cdf = np.zeros((1 << VBITS) + 1, dtype=np.int64)
    prev = [tok.eos_token_id] * S
    chatty = n > 10000
    for t in range(max(lens)):
        probs = pred.feed(prev)
        for s in range(S):
            if t >= lens[s]:
                continue                           # done: prev[s] stays, as filler
            _build_cdf(cdf, probs[s], vocab)
            tid = _code_token(coders[s], cdf, banks[s], mixers[s], apms[s],
                              None if decode else ids[starts[s] + t])
            if decode:
                ids[starts[s] + t] = tid
            if banks[s] is not None:
                banks[s].push(tid)
            prev[s] = tid
        if chatty and t and not t % 2000:
            done = sum(min(t, L) for L in lens)
            print(f'  ... {done:,}/{n:,} tokens (S={S})', file=sys.stderr, flush=True)
    if decode:
        return tok.decode(ids, clean_up_tokenization_spaces=False).encode('utf-8')
    blobs = [c.finish() for c in coders]
    return ((LOCKSTEP_FLAG | n).to_bytes(4, 'big') + bytes([S])
            + b''.join(len(b).to_bytes(4, 'big') for b in blobs) + b''.join(blobs))


def compress_lockstep(data, S=4):
    """S segments coded in lockstep. Unlike compress_batched, this DOES decode.

    Costs ratio, because each segment starts from zero context. The cost is a
    function of segment LENGTH, not of S: on alice29 at LIMIT=8192, S=4 measured
    +9.08% at 560-token segments and +3.97% at 2,221. Quote 3.97% - the scratch
    pilot claimed 2.6% for that step and it did NOT replicate.

    Both figures are still measured where a segment is SHORTER than LIMIT, so they
    mix one cold start per segment with reduced context per segment. On a file
    where segments exceed LIMIT only the cold start applies, which predicts a
    smaller penalty - a prediction, not a measurement, and nobody has run it.
    """
    return _lockstep(data, S=max(1, S))


def decompress_lockstep(blob):
    return _lockstep(blob, decode=True)


def decompress(blob):
    if not blob:
        return b''
    tok, _ = _load()
    n = int.from_bytes(blob[:4], 'big')
    if n & LOCKSTEP_FLAG:
        return decompress_lockstep(blob)
    if n & BATCHED_FLAG:
        raise ValueError(
            'this stream came from compress_batched, which is a ratio measurement '
            'and not a decodable codec - its GEMMs reduce floats in a different '
            'order than the sequential decoder. Re-encode with compress().')
    pred, dec = Predictor(), ptc.Decoder(blob[4:])
    bank = MatchBank(ORDERS) if ORDERS else None
    mixer = Mixer(1 + len(ORDERS)) if ORDERS else None
    apm = APM(8 * VBITS) if ORDERS and APM_ON else None
    prev = tok.eos_token_id
    ids = []
    for _ in range(n):
        pred.feed(prev)
        _build_cdf(pred.cdf, pred.prob, pred.V)
        tid = _code_token(dec, pred.cdf, bank, mixer, apm)
        ids.append(tid)
        if bank is not None:
            bank.push(tid)
        prev = tid
    # Explicit: the cleanup default flipped inside our supported transformers
    # range and is destructive for BPE (it strips spaces before punctuation).
    return tok.decode(ids, clean_up_tokenization_spaces=False).encode('utf-8')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'corpus/alice29.txt'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    data = open(path, 'rb').read()[:cap]
    _load()                                      # keep load time out of the timings

    # Ratio needs compress() only. Decoding proves correctness, and that is
    # already verified on prefixes through this same code path - so a long
    # headline run can skip it and cost half as much.
    encode_only = 'enc' in sys.argv[3:]
    batched = 'batch' in sys.argv[3:]
    # lockstep: S segments in one [S,1] forward per step. Decodes, unlike batch.
    lock = next((int(a[4:] or 4) for a in sys.argv[3:] if a.startswith('lock')), 0)

    t = time.perf_counter()
    if lock:
        packed = compress_lockstep(data, lock)
    else:
        packed = (compress_batched if batched else compress)(data)
    tc = time.perf_counter() - t
    td = None
    if batched and not encode_only:
        print('  (batched encode, sequential decode - this is the validity test)')

    if encode_only:
        print(f'{os.path.basename(path)} first {len(data):,} B   '
              f'round-trip: NOT RUN (encode-only)')
    else:
        t = time.perf_counter()
        back = decompress(packed)
        td = time.perf_counter() - t
        ok = back == data
        print(f'{os.path.basename(path)} first {len(data):,} B   round-trip: '
              f'{"ok" if ok else "FAILED"}')
        if not ok:
            sys.exit('  diverged at byte '
                     f'{next(i for i, (a, b) in enumerate(zip(back, data)) if a != b)}')
    import bz2
    import lzma
    rate = f'{len(data) / tc:.0f} B/s enc' + (f', {len(data) / td:.0f} B/s dec' if td else '')
    print(f'  llm_ptc  {len(packed):>8,} B  {8 * len(packed) / len(data):.3f} bpb   {rate}')
    for name, fn in (('ptc', ptc.compress), ('xz -9', lambda d: lzma.compress(d, preset=9)),
                     ('bz2 -9', lambda d: bz2.compress(d, 9))):
        c = fn(data)
        print(f'  {name:<8} {len(c):>8,} B  {8 * len(c) / len(data):.3f} bpb')
    mb = sum(os.path.getsize(os.path.join(r, f))
             for r, _, fs in os.walk(os.path.expanduser('~/.cache/huggingface/hub'))
             for f in fs if MODEL.replace('/', '--') in r) / 1e6
    print(f'  (model on disk: {mb:,.0f} MB - free only where both ends already have it)')
