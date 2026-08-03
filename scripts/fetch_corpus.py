"""Fetch the benchmark corpora. Not committed - some are large, all are public.

    python scripts/fetch_corpus.py

Files land in corpus/. Sizes are asserted against the published ones, because a
truncated download would silently produce wrong benchmark numbers.
"""
import hashlib
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
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


NEWS_SHA = '51f5badb611f9a1a7222fa5ffe96ce545d5c26e3e28ba161e1652a622484f2f4'
NEWS_API = 'https://en.wikinews.org/w/api.php'


def fetch_post2026_news():
    """The GENRE control: post-cutoff prose that is narrative, not technical.

    post2026.txt is arXiv. Qwen3 trains on far more technical text than SmolLM2,
    so on that file alone "generalises better" and "likes this genre" predict the
    same result - and they imply opposite advice. This file separates them, and
    it did: Qwen3-Base's 12.2% win on arXiv is a 0.7% loss here.

    Wikinews, CC BY. Sized to exactly 50,757 B to match post2026.txt so the two
    unseen corpora differ in genre and nothing else (trap 2). Not committed: CC BY
    needs attribution this repo would have to carry, and corpus/ is gitignored.

    The sha is unlikely to reproduce - it depends on which articles are newest at
    fetch time - so a mismatch means your bpb is not comparable to the recorded
    one, not that anything is broken.
    """
    import time
    dst = os.path.join(CORPUS, 'post2026_news.txt')
    if os.path.exists(dst):
        got = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
        print(f'have post2026_news.txt{"" if got == NEWS_SHA else "  <- differs from recorded"}')
        return

    def api(**kw):
        kw.setdefault('format', 'json')
        kw.setdefault('action', 'query')
        url = NEWS_API + '?' + urllib.parse.urlencode(kw)
        delay = 2.0
        for attempt in range(6):
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(
                    url, headers={'User-Agent': 'llm-compression-lab'}), timeout=90))
                time.sleep(0.6)           # explaintext is 1 article per request
                return r
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2

    print('fetching Wikinews 2026 articles ...')
    members, cont = [], {}
    while len(members) < 300:
        r = api(list='categorymembers', cmtitle='Category:Published', cmsort='timestamp',
                cmdir='desc', cmlimit=100, cmprop='title|timestamp', **cont)
        members += r['query']['categorymembers']
        if 'continue' not in r:
            break
        cont = r['continue']
    parts = []
    for m in (x for x in members if x['timestamp'] >= '2026-01-01'):
        r = api(prop='extracts', explaintext=1, titles=m['title'], redirects=1)
        t = next(iter(r['query']['pages'].values())).get('extract', '')
        # Drop == Sources == and friends: they were 20.6% of the first attempt,
        # and a citation list is structurally closer to arXiv's reference section
        # than to narrative - which would blunt the very contrast this file exists
        # to create.
        t = re.sub(r'\n=+ ?(Sources|Related news|External links|References|See also)'
                   r' ?=+.*?(?=\n=+ ?[A-Z]|\Z)', '', t, flags=re.S)
        t = ''.join(c for c in t if ord(c) < 128)
        t = re.sub(r'[ \t]+', ' ', t)
        t = re.sub(r'(\r?\n\s*){2,}', '\n\n', t).strip()
        if len(t) > 400:                                  # skip stubs and redirects
            parts.append(t)
        if sum(len(x) + 2 for x in parts) > 50_757 * 1.15:
            break
    data = '\n\n'.join(parts).encode('ascii')
    if len(data) < 50_757:
        sys.exit(f'  only {len(data):,} B of 2026 Wikinews available, need 50,757')
    data = data[:50_757]
    open(dst, 'wb').write(data)
    got = hashlib.sha256(data).hexdigest()
    print(f'  post2026_news.txt: {len(data):,} bytes from {len(parts)} articles')
    if got != NEWS_SHA:
        print(f'  NOTE: sha256 {got[:16]}... != recorded {NEWS_SHA[:16]}...\n'
              '        Expected - the newest articles change. Your bpb is not\n'
              '        directly comparable to the recorded genre-control figures.')


if __name__ == '__main__':
    os.makedirs(CORPUS, exist_ok=True)
    for url, members in ZIPS.items():
        fetch(url, members)
    slice_enwik8()
    fetch_post2026()
    fetch_post2026_news()
    print('\ncorpus ready. Nothing here is committed to the repo - these are other '
          'people\'s texts under their own licences.')
