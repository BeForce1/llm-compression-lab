"""PTC - PonyTail Coder. Ours. Not a repackaged DEFLATE.

Idea: never store data, store *surprise*. Five predictors guess the next
single bit; a learned mixer blends their opinions; an arithmetic coder spends
-log2(p) bits on the outcome. Nothing is copied, no dictionary, no code table -
a byte that was fully expected costs almost nothing, so the file size is
literally the model's total ignorance.

The predictors, in order of how much they earn their keep:
  order 1/2/4  - what usually follows these 1, 2 or 4 bytes
  match        - we saw this exact 6-byte context before, and this byte
                 followed it; confidence learned from match length
  stride       - what sits one record-width back, with the width voted on
                 continuously (fixed-size binary records hide from byte
                 contexts entirely, which is what bench.py caught)

Everything adapts as it reads, so encoder and decoder stay in lockstep with
zero side information: the header is 4 bytes of original length, period.

ponytail: pure Python, measured ~24 KB/s. Ratio is the point here, not
        throughput - but do not read that as "speed does not matter". Under a
        fixed time budget (which is exactly how the Hutter Prize scores: a time
        and memory LIMIT, then size only) speed is the currency you buy ratio
        with. A faster inner loop affords a wider mixer or a bigger model in the
        same wall clock. Every frontier compressor is hand-tuned C++ with SIMD
        for this reason, not for its own sake.
        Port order when the format is pinned (optimising a moving target is the
        waste, so not before):
          1. int8 GEMM kernels, AVX-512 VNNI / VPDPBUSD   -> 4-10x on the model
          2. mixer dot product + weight update, FMA       -> scales with NINP
          3. counter table updates, software prefetch     -> memory-bound
          4. arithmetic coder, branch-free                -> ~5%
"""
import math
from array import array

BITS = 12
ONE = 1 << BITS                       # probabilities live in [1, 4095]
NCTX = 4                              # hashed contexts: order 1, 2, 4, stride
NINP = NCTX + 1                       # ... plus the match model
TABLE = 1 << 20                       # counters per hashed predictor
MASK = TABLE - 1
MTMASK = (1 << 20) - 1                # match index: context hash -> position
MINLEN = 6                            # context bytes before we trust a match
MAXLEN = 31                           # match length is also a confidence bucket
STRIDES = (2, 3, 4, 8, 16)            # candidate record widths
LR = 12                               # mixer learning rate (right-shift)
RATE = 5                              # counter adaptation (right-shift)

# stretch(p) = ln(p/(1-p)), squash = its inverse. Mixing in the stretched
# (log-odds) domain is what makes a weighted blend behave sanely near 0 and 1.
_STRETCH = array('h', (max(-2047, min(2047, round(256 * math.log((p + 0.5) / (ONE - p - 0.5)))))
                       for p in range(ONE)))
_SQUASH = array('H', (max(1, min(ONE - 1, int(ONE / (1 + math.exp(-x / 256)))))
                      for x in range(-2047, 2048)))


def _squash(x):
    return _SQUASH[(-2047 if x < -2047 else 2047 if x > 2047 else x) + 2047]


def _clamp(v):
    return 1 if v < 1 else ONE - 1 if v > ONE - 1 else v


class Model:
    """Five adaptive bit predictors plus the mixer that learns to trust them.

    Call order per byte is begin() -> 8x (predict, update) -> push(byte), and
    it must be identical on both sides or the two diverge silently.
    """

    def __init__(self):
        self.tabs = [array('H', [ONE >> 1]) * TABLE for _ in range(NCTX)]
        self.mt = array('i', [0]) * (MTMASK + 1)     # where a context last occurred
        self.mtab = array('H', [ONE >> 1]) * (2 * (MAXLEN + 1))
        self.w = [1 << 15] * NINP
        self.st = [0] * NINP
        self.idx = [0] * NCTX
        self.p = ONE >> 1
        self.buf = bytearray()
        self.hist = 0
        self.mptr = self.mlen = 0
        self.exp = None                              # byte the match model expects
        self.mi = -1
        self.votes = [0] * len(STRIDES)
        self.stride = STRIDES[0]
        self.hs = (0,) * NCTX

    def begin(self):
        """Freeze the contexts that every bit of this byte will share."""
        h, s, b = self.hist, self.stride, self.buf
        far = (b[-s] << 8 | b[-2 * s]) if len(b) >= 2 * s else 0
        self.hs = ((h & 0xFF) * 0x2545F491 & 0xFFFFFFFF,
                   (h & 0xFFFF) * 0x9E3779B1 & 0xFFFFFFFF,
                   (h & 0xFFFFFFFF) * 0x85EBCA6B & 0xFFFFFFFF,
                   (far * 0x27220A95 ^ s * 0x165667B1) & 0xFFFFFFFF)
        self.exp = b[self.mptr] if self.mlen and self.mptr < len(b) else None

    def predict(self, node):
        total = 0
        nk = node * 0x9E3779B1
        for i in range(NCTX):
            j = (self.hs[i] ^ nk) & MASK
            self.idx[i] = j
            s = _STRETCH[self.tabs[i][j]]
            self.st[i] = s
            total += self.w[i] * s
        # Match model: only speaks while the expected byte still agrees with the
        # bits already coded. Its confidence is looked up, not hand-tuned.
        self.mi, s = -1, 0
        if self.exp is not None:
            j = node.bit_length() - 1
            if (0x100 | self.exp) >> (8 - j) == node:
                self.mi = self.mlen * 2 + (self.exp >> (7 - j) & 1)
                s = _STRETCH[self.mtab[self.mi]]
            else:
                self.exp = None                      # wrong, stay quiet this byte
        self.st[NCTX] = s
        total += self.w[NCTX] * s
        self.p = _squash(total >> 16)
        return self.p

    def update(self, bit):
        target = bit << BITS
        err = target - self.p
        for i in range(NINP):
            self.w[i] += (self.st[i] * err) >> LR   # st = 0 means no vote, no update
        for i in range(NCTX):
            t, j = self.tabs[i], self.idx[i]
            t[j] = _clamp(t[j] + ((target - t[j]) >> RATE))
        if self.mi >= 0:
            self.mtab[self.mi] = _clamp(
                self.mtab[self.mi] + ((target - self.mtab[self.mi]) >> RATE))

    def push(self, byte):
        """Advance history, match pointer and the stride vote."""
        b = self.buf
        b.append(byte)
        pos = len(b)
        if self.mlen and self.mptr < pos - 1 and b[self.mptr] == byte:
            self.mptr += 1
            if self.mlen < MAXLEN:
                self.mlen += 1
        else:
            self.mlen = 0
        self.hist = (self.hist << 8 | byte) & 0xFFFFFFFFFFFF
        if pos >= MINLEN:
            h = (self.hist * 0x2545F4914F6CDD1D >> 26) & MTMASK
            if not self.mlen:
                cand = self.mt[h]
                if cand:
                    self.mptr, self.mlen = cand, 1
            self.mt[h] = pos
        # Which record width would have predicted the byte we just saw? Decay so
        # the answer can change mid-file (headers, then rows, then trailer).
        for i, s in enumerate(STRIDES):
            v = self.votes[i]
            self.votes[i] = v - (v >> 6) + (16 if pos > s and b[-1 - s] == byte else 0)
        self.stride = STRIDES[self.votes.index(max(self.votes))]


class _Coder:
    """Carry-less binary arithmetic coder. Encoder and decoder run the exact
    same interval arithmetic, so they cannot drift apart."""

    def __init__(self):
        self.lo = 0
        self.hi = 0xFFFFFFFF

    def _mid(self, p):
        span = self.hi - self.lo
        m = self.lo + (span >> BITS) * p
        return self.lo if m < self.lo else self.hi - 1 if m >= self.hi else m


class Encoder(_Coder):
    def __init__(self):
        super().__init__()
        self.out = bytearray()

    def encode(self, bit, p):
        m = self._mid(p)
        if bit:
            self.hi = m
        else:
            self.lo = m + 1
        while not (self.lo ^ self.hi) & 0xFF000000:
            self.out.append(self.lo >> 24)
            self.lo = self.lo << 8 & 0xFFFFFFFF
            self.hi = (self.hi << 8 | 0xFF) & 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.out.append(self.lo >> 24)
            self.lo = self.lo << 8 & 0xFFFFFFFF
        return bytes(self.out)


class Decoder(_Coder):
    def __init__(self, buf):
        super().__init__()
        self.buf = buf
        self.i = 4
        self.x = int.from_bytes(buf[:4].ljust(4, b'\0'), 'big')

    def decode(self, p):
        m = self._mid(p)
        bit = 1 if self.x <= m else 0
        if bit:
            self.hi = m
        else:
            self.lo = m + 1
        while not (self.lo ^ self.hi) & 0xFF000000:
            self.lo = self.lo << 8 & 0xFFFFFFFF
            self.hi = (self.hi << 8 | 0xFF) & 0xFFFFFFFF
            nxt = self.buf[self.i] if self.i < len(self.buf) else 0
            self.i += 1
            self.x = (self.x << 8 | nxt) & 0xFFFFFFFF
        return bit


def compress(data):
    if not data:
        return b''
    m, enc = Model(), Encoder()
    for byte in data:
        m.begin()
        node = 1
        for k in (7, 6, 5, 4, 3, 2, 1, 0):
            bit = byte >> k & 1
            enc.encode(bit, m.predict(node))
            m.update(bit)
            node = node << 1 | bit
        m.push(byte)
    return len(data).to_bytes(4, 'big') + enc.finish()


def decompress(blob):
    if not blob:
        return b''
    n = int.from_bytes(blob[:4], 'big')
    m, dec = Model(), Decoder(blob[4:])
    out = bytearray()
    while len(out) < n:
        m.begin()
        node = 1
        for _ in range(8):
            bit = dec.decode(m.predict(node))
            m.update(bit)
            node = node << 1 | bit
        byte = node & 0xFF
        out.append(byte)
        m.push(byte)
    return bytes(out)


if __name__ == '__main__':
    import zlib, lzma

    cases = [
        ('empty', b''),
        ('one byte', b'a'),
        ('runs', b'a' * 10),
        ('english', (b'the quick brown fox jumps over the lazy dog. ' * 40)),
        ('counter', bytes(range(256)) * 12),
        ('random', bytes((i * 2654435761 >> 13) & 0xFF for i in range(3000))),
        ('source', open(__file__, 'rb').read()),
    ]
    for name, c in cases:
        assert decompress(compress(c)) == c, f'round-trip failed: {name}'
    print('round-trip: ok\n')

    print(f"{'input':>10} {'raw':>7} {'ptc':>7} {'zlib':>7} {'lzma':>7}   vs zlib")
    for name, c in cases[2:]:
        p, z, x = len(compress(c)), len(zlib.compress(c, 9)), len(lzma.compress(c))
        print(f'{name:>10} {len(c):>7} {p:>7} {z:>7} {x:>7}   {z / p:>5.2f}x')
