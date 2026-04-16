"""Bulk-download perfume images via Bing image search.

Progress is printed to stdout as we go (per CLAUDE.md: long-running programs
should emit intermediate results).

Strategy:
  1. For each perfume, hit Bing image search with "<name> perfume bottle".
  2. Parse the inline 'murl' (media URL) fields from the HTML.
  3. Try candidate URLs in order — take the first that downloads as a real
     image >= MIN_BYTES and isn't an HTML error page.
  4. Save to images/{slug}.{ext}.  A manifest maps slug -> filename.

Usage:
  python3 download_images.py                # download all remaining
  python3 download_images.py --only slug1   # one-off
  python3 download_images.py --limit 20     # first N
  python3 download_images.py --force        # re-download even if present
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / 'images'
MANIFEST = IMAGES / 'manifest.json'
PERFUMES_CACHE = ROOT / 'batches' / 'perfumes.json'

UA = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)
HEADERS = {
    'User-Agent': UA,
    'Accept': 'text/html,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

MIN_BYTES = 5_000
TIMEOUT = 15

IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'webp', 'gif'}


def http_get(url: str, *, binary: bool = False, timeout: int = TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data if binary else data


def ext_from_url_or_bytes(url: str, data: bytes) -> str | None:
    # Prefer extension from URL
    lower = url.lower().split('?')[0]
    for e in ('jpeg', 'jpg', 'png', 'webp', 'gif'):
        if lower.endswith('.' + e):
            return 'jpg' if e == 'jpeg' else e
    # Sniff magic bytes
    if data.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    return None


def is_real_image(data: bytes) -> bool:
    if len(data) < MIN_BYTES:
        return False
    # Reject HTML error pages
    head = data[:200].lower()
    if b'<html' in head or b'<!doctype' in head:
        return False
    return ext_from_url_or_bytes('', data) is not None


def find_image_urls(query: str) -> list[str]:
    """Scrape Bing Images for candidate URLs."""
    q = urllib.parse.quote_plus(query)
    search_url = (
        f'https://www.bing.com/images/search?q={q}&form=HDRSC2&first=1'
    )
    try:
        html = http_get(search_url).decode('utf-8', errors='ignore')
    except Exception as e:
        print(f'    ! bing fetch failed: {e}')
        return []

    urls: list[str] = []
    for raw in re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', html):
        url = raw.encode().decode('unicode_escape')
        urls.append(url)

    # Filter: drop obviously non-image URLs, drop tiny social/CDN thumbs we
    # can detect, keep order (Bing ranks roughly by relevance).
    filtered: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        lower = u.lower().split('?')[0]
        if not any(lower.endswith('.' + e) for e in IMAGE_EXTS):
            # Still include — server may return correct content-type.
            pass
        filtered.append(u)
    return filtered


def already_have(slug: str, manifest: dict) -> bool:
    if slug not in manifest:
        return False
    f = IMAGES / manifest[slug]
    return f.exists() and f.stat().st_size >= MIN_BYTES


def download_one(slug: str, name: str, *, force: bool, manifest: dict) -> tuple[str, str, str]:
    """Returns (slug, status, detail). Mutates manifest on success."""
    if not force and already_have(slug, manifest):
        return slug, 'cached', manifest[slug]

    query = f'{name} perfume bottle'
    candidates = find_image_urls(query)
    if not candidates:
        return slug, 'no_candidates', ''

    tried = 0
    for url in candidates[:8]:  # try up to 8 candidates
        tried += 1
        try:
            data = http_get(url, binary=True)
        except Exception as e:
            continue
        if not is_real_image(data):
            continue
        ext = ext_from_url_or_bytes(url, data) or 'jpg'
        filename = f'{slug}.{ext}'
        path = IMAGES / filename
        path.write_bytes(data)
        manifest[slug] = filename
        return slug, 'ok', f'{filename} ({len(data)//1024}KB, tried {tried})'

    return slug, 'all_failed', f'tried {tried} candidates'


def load_perfumes() -> list[dict]:
    if PERFUMES_CACHE.exists():
        return json.loads(PERFUMES_CACHE.read_text())
    # Rebuild from batches/batch_*.json
    batches = sorted((ROOT / 'batches').glob('batch_*.json'))
    merged: list[dict] = []
    for b in batches:
        merged.extend(json.loads(b.read_text()))
    PERFUMES_CACHE.parent.mkdir(exist_ok=True)
    PERFUMES_CACHE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=[], help='slug(s) to fetch')
    ap.add_argument('--limit', type=int, default=0, help='process at most N')
    ap.add_argument('--force', action='store_true', help='redownload cached')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    IMAGES.mkdir(exist_ok=True)
    perfumes = load_perfumes()
    manifest = load_manifest()

    if args.only:
        perfumes = [p for p in perfumes if p['slug'] in set(args.only)]
    elif args.limit:
        # Prefer ones we don't have yet.
        missing = [p for p in perfumes if not already_have(p['slug'], manifest)]
        perfumes = missing[: args.limit]

    # Skip cached unless --force
    todo = [p for p in perfumes if args.force or not already_have(p['slug'], manifest)]
    cached = len(perfumes) - len(todo)
    print(f'▶ {len(perfumes)} perfumes selected · {cached} already cached · {len(todo)} to fetch')

    if not todo:
        print('Nothing to do.')
        return 0

    ok = failed = 0
    start = time.time()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, p['slug'], p['name'],
                      force=args.force, manifest=manifest): p
            for p in todo
        }
        for i, fut in enumerate(cf.as_completed(futures), 1):
            p = futures[fut]
            try:
                slug, status, detail = fut.result()
            except Exception as e:
                slug, status, detail = p['slug'], 'exception', repr(e)
            if status == 'ok':
                ok += 1
                print(f'[{i:>3}/{len(todo)}] ✓ {p["name"]}  →  {detail}')
            elif status == 'cached':
                ok += 1
                print(f'[{i:>3}/{len(todo)}] (cached) {p["name"]}')
            else:
                failed += 1
                print(f'[{i:>3}/{len(todo)}] ✗ {p["name"]}  [{status}] {detail}')
            # Flush manifest periodically so we don't lose progress.
            if i % 20 == 0:
                save_manifest(manifest)

    save_manifest(manifest)
    dur = time.time() - start
    print(f'\nDone in {dur:.0f}s. ok={ok} failed={failed} total_manifest={len(manifest)}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
