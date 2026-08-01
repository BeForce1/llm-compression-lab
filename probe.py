"""Why are we at ~1.81 bpb when ts_zip reports 1.142 on the same file?

Two suspects, and this separates them instead of guessing:

  A. OUR CODER wastes bits. Measured by comparing the bits we actually emit
     against the model's own uncertainty, -log2 p(token) summed. That sum is
     the floor no coder can beat, so the difference is purely our overhead
     (binarisation + 12-bit probability quantisation).

  B. CONTEXT RESETS. GPT-2 caps at 1024 tokens so we rebuild from the last
     WINDOW; RWKV (ts_zip) never resets. Measured by sweeping the context
     budget - if bpb barely moves from 256 to 1024 tokens of context, then
     unbounded context would not have saved us either and the gap is model
     quality, not our windowing.

Sizes only. The headline job owns the CPU, so ignore any timing here.
"""
import math
import os
import sys

import ptc
import llm_ptc

SWEEP = [(256, 128), (512, 256), (1024, 512)]      # (LIMIT, WINDOW)


def analyse(data, limit, window):
    tok, _ = llm_ptc._load()
    llm_ptc.LIMIT, llm_ptc.WINDOW = limit, window
    ids = tok(data.decode('utf-8')).input_ids
    pred, enc = llm_ptc.Predictor(), ptc.Encoder()
    prev = tok.eos_token_id
    ideal = 0.0        # -log2 p under the model's RAW distribution
    smoothed = 0.0     # -log2 p under the 12-bit-quantised probs we actually code with
    resets = 0
    since = []                                      # tokens since the last slide
    cost = []                                       # raw-model bits for that token
    age = 0                                         # 0 = first token after a slide

    for tid in ids:
        before = len(pred.ids)
        pred.feed(prev)
        if len(pred.ids) <= before:                 # window slid
            resets += 1
            age = 0
        cdf = pred.cdf
        mass = int(cdf[tid + 1]) - int(cdf[tid])
        bits = -math.log2(mass / int(cdf[-1]))
        ideal += bits
        since.append(age)
        cost.append(bits)
        age += 1

        lo, hi = 0, 1 << llm_ptc.VBITS
        for _ in range(llm_ptc.VBITS):
            p, mid = llm_ptc._split(cdf, lo, hi)
            bit = 1 if tid >= mid else 0
            # what the coder is actually charged for this decision
            q = p / ptc.ONE if bit else 1 - p / ptc.ONE
            smoothed += -math.log2(q)
            enc.encode(bit, p)
            lo, hi = (mid, hi) if bit else (lo, mid)
        prev = tid

    actual_bits = 8 * (len(enc.finish()) + 4)       # + the 4-byte length header
    return {'limit': limit, 'window': window, 'tokens': len(ids), 'resets': resets,
            'ideal_bpb': ideal / len(data), 'smooth_bpb': smoothed / len(data),
            'actual_bpb': actual_bits / len(data),
            'coder': actual_bits / smoothed - 1,    # pure coder inefficiency
            'smoothing': smoothed / ideal - 1,      # what 12-bit clamping buys/costs
            'since': since, 'cost': cost}


def cold_vs_warm(r, edge=64):
    """Do tokens just after a slide cost more than tokens deep into a window?

    Both buckets must come from AFTER the first slide, or this measures
    memorisation instead of context depth - GPT-2 has Alice's opening lines by
    heart, and the first version of this probe reported nonsense for exactly
    that reason. The second version still did, less obviously: it skipped only
    the first WINDOW and defined warm as age >= LIMIT-64, but a token's age can
    never exceed LIMIT-WINDOW after a slide, so *no* post-slide token could ever
    land in warm. The warm bucket was 100% pre-slide file opening. Age is now
    measured against what post-slide tokens can actually reach.
    """
    # Needs a sample big enough to slide a few times: at the 4096-byte default
    # the LIMIT=1024 row only reaches age 73, so its warm bucket is empty and
    # this returns None. Raise the cap to measure that row.
    top = r['limit'] - r['window']                 # oldest age reachable post-slide
    pairs = list(zip(r['since'], r['cost']))[r['limit']:]      # drop everything pre-slide
    cold = [c for s, c in pairs if s < edge]
    warm = [c for s, c in pairs if s >= top - edge]
    if not cold or not warm:
        return None
    return sum(cold) / len(cold), len(cold), sum(warm) / len(warm), len(warm)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'corpus/alice29.txt'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    data = open(path, 'rb').read()[:cap]
    llm_ptc._load()

    print(f'{os.path.basename(path)} first {len(data):,} B\n')
    print(f"  {'context':>9} {'slides':>7} {'raw model':>10} {'quantised':>10} "
          f"{'emitted':>9} {'coder':>8} {'smoothing':>10}")
    results = []
    for limit, window in SWEEP:
        r = analyse(data, limit, window)
        results.append(r)
        print(f'  {limit:>9} {r["resets"]:>7} {r["ideal_bpb"]:>10.3f} '
              f'{r["smooth_bpb"]:>10.3f} {r["actual_bpb"]:>9.3f} '
              f'{r["coder"]:>7.2%} {r["smoothing"]:>9.2%}')

    print('\n  A: is our coder the problem?')
    for r in results:
        print(f'     context {r["limit"]:>4}: coding against our quantised probs costs '
              f'{r["smooth_bpb"]:.3f} bpb, we emit {r["actual_bpb"]:.3f} '
              f'-> coder wastes {r["coder"]:.2%}')
    print(f'     12-bit clamping vs the raw model: {results[-1]["smoothing"]:+.2%} '
          f'(negative = the clamp SMOOTHS an overconfident model and wins)')

    print('\n  B: is context length the problem?')
    best, worst = results[-1], results[0]
    print(f'     {worst["limit"]} -> {best["limit"]} tokens of context: '
          f'{worst["actual_bpb"]:.3f} -> {best["actual_bpb"]:.3f} bpb  '
          f'({1 - best["actual_bpb"] / worst["actual_bpb"]:+.1%})')
    for r in results:
        cw = cold_vs_warm(r)
        if cw:
            print(f'     context {r["limit"]:>4}: first {64} tokens after a slide '
                  f'{cw[0]:.2f} bits (n={cw[1]}), deep-in-window {cw[2]:.2f} bits '
                  f'(n={cw[3]})')
    print(f'\n  target: ts_zip 1.142 bpb (RWKV 169M, unbounded context) on full alice29')
