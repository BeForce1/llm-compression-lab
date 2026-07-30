"""Scoreboard. Every compression idea gets judged here before it gets built.

Four rules, because each one is a way people fool themselves:

  * round-trip is verified per file - a codec that cannot decompress scores
    nothing, no matter how small its output
  * bits per byte, because that is the unit every rival and every paper quotes
  * corpus files are byte-identical to the ones xz and ts_zip publish on, so
    our number lands in the same table as theirs (see REFERENCE)
  * the decoder counts - printed as a self-contained total, Hutter-style

Usage:  python bench.py [byte-cap-for-ptc]     (0 or omitted = whole files)
"""
import bz2
import lzma
import math
import os
import sys
import time
import zlib

import brotli

import ptc

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, 'corpus')
FILES = ['alice29.txt', 'book1', 'kennedy.xls', 'ptt5']
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 0        # ptc is the slow one

# Quoted from bellard.org/ts_zip, NOT measured here. Bits per byte. These are
# the targets: 'xz -9' should match our own measurement (same file, same
# setting) which is how we know the harness is not lying to us.
REFERENCE = {
    'alice29.txt': {'xz -9': 2.551, 'ts_zip': 1.142},
    'book1': {'xz -9': 2.717, 'ts_zip': 1.431},
}

CODECS = {
    'zlib -9': (lambda d: zlib.compress(d, 9), zlib.decompress),
    'bz2 -9': (lambda d: bz2.compress(d, 9), bz2.decompress),
    'xz -9': (lambda d: lzma.compress(d, preset=9), lzma.decompress),
    'brotli 11': (lambda d: brotli.compress(d, quality=11), brotli.decompress),
    'ptc': (ptc.compress, ptc.decompress),
}
BASELINE = 'xz -9'
SELF_CONTAINED = {'ptc': os.path.join(HERE, 'ptc.py')}   # ours ships its decoder


def measure(name, data):
    """Compress, decompress, verify. Returns None if it does not round-trip."""
    comp, decomp = CODECS[name]
    t = time.perf_counter()
    packed = comp(data)
    tc = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    restored = decomp(packed)
    td = (time.perf_counter() - t) * 1000
    if restored != data:
        return None
    return {'size': len(packed), 'bpb': 8 * len(packed) / len(data),
            'ratio': len(data) / len(packed), 'tc': tc, 'td': td}


def weissman(row, base, alpha=1.0):
    """W = alpha * (r/r_base) * (log T_base / log T).  Baseline scores 1.0.

    ponytail: the log makes this unit-sensitive, so the unit is part of the
            metric - fixed here at milliseconds, floored at 2ms so log T > 0.
            Compare our W only against other W from this harness.
    """
    t, tb = max(row['tc'], 2.0), max(base['tc'], 2.0)
    return alpha * (row['ratio'] / base['ratio']) * (math.log(tb) / math.log(t))


def run(path):
    data = open(path, 'rb').read()
    name = os.path.basename(path)
    print(f'\n{name}  -  {len(data):,} bytes')
    print(f"  {'codec':<10} {'bytes':>9} {'bpb':>6} {'ratio':>6} "
          f"{'comp ms':>9} {'dec ms':>8} {'weissman':>9}")

    rows, capped = {}, False
    for codec in CODECS:
        subset = data
        if codec == 'ptc' and CAP and len(data) > CAP:
            subset, capped = data[:CAP], True
        r = measure(codec, subset)
        if r is None:
            print(f'  {codec:<10} {"ROUND-TRIP FAILED":>50}')
            continue
        rows[codec] = r

    base = rows.get(BASELINE)
    for codec, r in rows.items():
        w = f'{weissman(r, base):>9.3f}' if base else f'{"-":>9}'
        mark = ' *' if codec == 'ptc' and capped else ''
        print(f'  {codec:<10} {r["size"]:>9,} {r["bpb"]:>6.3f} {r["ratio"]:>6.2f} '
              f'{r["tc"]:>9.1f} {r["td"]:>8.1f} {w}{mark}')

    for codec, src in SELF_CONTAINED.items():
        if codec in rows:
            dec = os.path.getsize(src)
            total = rows[codec]['size'] + dec
            print(f'  {"":<10} {codec} + its own decoder ({dec:,} B) = {total:,} B '
                  f'-> {8 * total / len(data):.3f} bpb'
                  f'{"  (beats xz)" if base and total < base["size"] else ""}')

    for codec, bpb in REFERENCE.get(name, {}).items():
        ours = rows.get(codec, {}).get('bpb')
        check = f'   [ours: {ours:.3f}]' if ours else ''
        print(f'  {"published":<10} {codec:<12} {bpb:>6.3f} bpb{check}')
    if capped:
        print(f'  * ptc measured on first {CAP:,} bytes only - not comparable '
              f'to the full-file numbers above')


if __name__ == '__main__':
    missing = [f for f in FILES if not os.path.exists(os.path.join(CORPUS, f))]
    if missing:
        sys.exit(f'missing corpus files in {CORPUS}: {missing}')
    for f in FILES:
        run(os.path.join(CORPUS, f))
    print('\nround-trip verified for every row shown. baseline = xz -9 (W = 1.000)')
