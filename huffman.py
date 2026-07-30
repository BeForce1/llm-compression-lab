# Order-0 Huffman codec, round-trips arbitrary bytes.
# ponytail: real compression is zlib/zstd (already in stdlib). This is the learn-it version.
import heapq, zlib
from collections import Counter


def _lengths(data):
    """Code length per symbol, from a Huffman tree."""
    freq = Counter(data)
    if len(freq) == 1:
        return {s: 1 for s in freq}          # one symbol still needs one bit
    heap = [(f, i, {s: 0}) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        for d in (a, b):
            for s in d:
                d[s] += 1                    # ponytail: O(n^2) depth bump, fine under 256 symbols
        a.update(b)
        heapq.heappush(heap, (f1 + f2, tie, a))
        tie += 1
    return heap[0][2]


def _codes(lengths):
    """Canonical Huffman: lengths alone rebuild the exact same codes, so the
    header only needs to store lengths."""
    code = prev = 0
    out = {}
    for s, L in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= L - prev
        prev = L
        out[s] = (code, L)
        code += 1
    return out


def compress(data):
    if not data:
        return b""
    lengths = _lengths(data)
    codes = _codes(lengths)
    bits = nbits = 0
    out = bytearray()
    for byte in data:
        c, L = codes[byte]
        bits = (bits << L) | c
        nbits += L
        while nbits >= 8:
            nbits -= 8
            out.append((bits >> nbits) & 0xFF)
        bits &= (1 << nbits) - 1             # drop consumed high bits, keeps the int small
    if nbits:
        out.append((bits << (8 - nbits)) & 0xFF)
    header = bytes(lengths.get(i, 0) for i in range(256))
    return len(data).to_bytes(4, "big") + header + bytes(out)


def decompress(blob):
    if not blob:
        return b""
    n = int.from_bytes(blob[:4], "big")
    lengths = {i: L for i, L in enumerate(blob[4:260]) if L}
    lookup = {(L, c): s for s, (c, L) in _codes(lengths).items()}
    out = bytearray()
    code = L = 0
    for byte in blob[260:]:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            L += 1
            if (L, code) in lookup:
                out.append(lookup[(L, code)])
                code = L = 0
                if len(out) == n:
                    return bytes(out)
    return bytes(out)


if __name__ == "__main__":
    cases = [
        b"",
        b"a",
        b"aaaaaaaaaa",
        b"the quick brown fox jumps over the lazy dog " * 50,
        bytes(range(256)) * 20,
        open(__file__, "rb").read(),
    ]
    for c in cases:
        assert decompress(compress(c)) == c, f"round-trip failed on {c[:20]!r}"
    print("round-trip: ok\n")

    print(f"{'input':>28} {'raw':>8} {'huffman':>8} {'zlib':>8}")
    for c in cases[2:]:
        label = repr(c[:24])[:26]
        print(f"{label:>28} {len(c):>8} {len(compress(c)):>8} {len(zlib.compress(c)):>8}")
