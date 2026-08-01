"""Fetch the benchmark corpora. Not committed - some are large, all are public.

    python scripts/fetch_corpus.py

Files land in corpus/. Sizes are asserted against the published ones, because a
truncated download would silently produce wrong benchmark numbers.
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(HERE), 'corpus')

# zip url -> {member: expected size}. Sizes are the published ones; every number
# in the README depends on getting byte-identical files.
ZIPS = {
    'https://corpus.canterbury.ac.nz/resources/cantrbry.zip':
        {'alice29.txt': 152089, 'ptt5': 513216, 'kennedy.xls': 1029744},
    'https://corpus.canterbury.ac.nz/resources/calgary.zip':
        {'book1': 768771},
    'https://mattmahoney.net/dc/enwik8.zip':
        {'enwik8': 100000000},
}


def _ok(member, size):
    """Present AND the right size. Existence alone is not integrity: a file that
    was truncated on its first write is accepted forever by an exists() check,
    and only alice29/book1 have a published figure that would expose it."""
    p = os.path.join(CORPUS, member)
    return os.path.exists(p) and os.path.getsize(p) == size


def fetch(url, members):
    if all(_ok(m, s) for m, s in members.items()):
        print(f'have {", ".join(members)}')
        return
    print(f'fetching {url} ...')
    with urllib.request.urlopen(url, timeout=300) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for member, size in members.items():
            with z.open(names[member]) as f:
                data = f.read()
            if len(data) != size:
                sys.exit(f'  {member}: got {len(data):,} bytes, expected {size:,}')
            open(os.path.join(CORPUS, member), 'wb').write(data)
            print(f'  {member}: {len(data):,} bytes')


def slice_enwik8():
    """A mid-file slice, trimmed to a valid UTF-8 boundary.

    The FIRST 256 KB of enwik8 is the XML preamble and siteinfo block, which
    compresses ~13% better than real article text and makes any codec look good.
    Benchmark from the middle.
    """
    src = os.path.join(CORPUS, 'enwik8')
    dst = os.path.join(CORPUS, 'enwik8_mid')
    if not os.path.exists(src) or os.path.exists(dst):
        return
    with open(src, 'rb') as f:
        f.seek(50_000_000)
        buf = f.read(262_144)
    # Trim to character boundaries directly rather than by trial decoding. The
    # old loop raised on the END being mid-character and "fixed" it by advancing
    # START, silently dropping up to 4 valid bytes - or, if byte 4 was itself a
    # continuation byte, walking end down to start and writing an EMPTY file.
    start, end = 0, len(buf)
    while start < end and buf[start] & 0xC0 == 0x80:      # skip continuation bytes
        start += 1
    while end > start:
        try:
            buf[start:end].decode('utf-8')
            break
        except UnicodeDecodeError:
            end -= 1
    out = buf[start:end]
    assert out, 'enwik8_mid trimmed to nothing - check the source file'
    open(dst, 'wb').write(out)
    print(f'  enwik8_mid: {len(out):,} bytes from offset 50,000,000, '
          f'sha256 {hashlib.sha256(out).hexdigest()[:16]}...')


POST2026_URL = 'https://arxiv.org/html/2602.19626'
POST2026_SHA = '20b192d7639c0fdadaa6c0737c3c68577785062e655c3e2efb108aea03de4839'


def fetch_post2026():
    """The contamination control: prose published long after the model's cutoff.

    Fetched, never committed. Two reasons: it is someone else's paper under
    their licence, not ours to redistribute under MIT, and it carries the
    author's contact details, which do not belong in a public repo.

    The sha256 pins the exact bytes the README's 1.338 bpb was measured on.
    arXiv pages do change, so a mismatch is a warning rather than an error -
    it means your number is not comparable to the recorded one, not that
    anything is broken.
    """
    import re
    dst = os.path.join(CORPUS, 'post2026.txt')
    if os.path.exists(dst):
        # Re-hash: the pin is worthless if it is only checked on the fetch that
        # creates the file. A copy grabbed from a since-changed arXiv page would
        # otherwise report 'have post2026.txt' forever.
        got = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
        print(f'have post2026.txt{"" if got == POST2026_SHA else "  <- SHA MISMATCH"}')
        if got != POST2026_SHA:
            print(f'  NOTE: sha256 {got[:16]}... != pinned {POST2026_SHA[:16]}...\n'
                  '        Your contamination bpb is not comparable to the recorded one.')
        return
    print(f'fetching {POST2026_URL} ...')
    req = urllib.request.Request(POST2026_URL, headers={'User-Agent': 'llm-compression-lab'})
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode('utf-8', errors='replace')
    for block in ('script', 'style', 'head'):
        html = re.sub(rf'<{block}.*?</{block}>', '', html, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', html)
    for a, b in (('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>')):
        text = text.replace(a, b)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(\r?\n\s*){2,}', '\n\n', text).strip()
    text = ''.join(c for c in text if ord(c) < 128)   # ascii, to match alice29
    data = text.encode('utf-8')
    open(dst, 'wb').write(data)
    got = hashlib.sha256(data).hexdigest()
    print(f'  post2026.txt: {len(data):,} bytes')
    if got != POST2026_SHA:
        print(f'  NOTE: sha256 {got[:16]}... != pinned {POST2026_SHA[:16]}...\n'
              '        The arXiv page has changed since 2026-07-31. Your bpb will\n'
              '        not be directly comparable to the figure in the README.')


if __name__ == '__main__':
    os.makedirs(CORPUS, exist_ok=True)
    for url, members in ZIPS.items():
        fetch(url, members)
    slice_enwik8()
    fetch_post2026()
    print('\ncorpus ready. Nothing here is committed to the repo - these are other '
          'people\'s texts under their own licences.')
