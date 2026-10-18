"""Stage 2 — download (optional, not on the critical path).

Reads data/manifest.csv. For each award/report URL: live first, then the
Wayback raw copy (`id_` = original bytes, no toolbar). Files already on disk
are skipped. Files capped at exactly 1,048,576 bytes are retried from the
largest other capture of the same URL — different crawls hit different caps.

Local names are f"{i:04d}_{filename}" with i the 1-based row index of the
award/report subset of the manifest, matching the Kaggle batch.
"""

import argparse
import csv
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
MANIFEST = os.path.join(DATA, "manifest.csv")
PDF_DIR = os.path.join(DATA, "pdfs")

CDX = "http://web.archive.org/cdx/search/cdx"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; dataset construction)"}
CAP = 1048576
DELAY = 2.0          # the Archive throttles hard and returns blocks of failures


def all_captures(url):
    """Every capture of one URL (no collapse), largest first."""
    try:
        r = requests.get(CDX, params={
            "url": url, "output": "json",
            "fl": "timestamp,length,statuscode", "limit": "200"},
            headers=HEADERS, timeout=60)
        rows = r.json()
    except Exception:
        return []
    if len(rows) < 2:
        return []
    caps = [dict(zip(rows[0], x)) for x in rows[1:]]
    return sorted(caps, key=lambda d: int(d.get("length") or 0), reverse=True)


def grab(url, ts, dest):
    for target in (url, f"https://web.archive.org/web/{ts}id_/{url}"):
        try:
            r = requests.get(target, headers=HEADERS, timeout=180)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(dest, "wb") as f:
                    f.write(r.content)
                return len(r.content)
        except requests.RequestException:
            pass
        time.sleep(DELAY)
    return 0


def download(dest_dir=PDF_DIR, delay=DELAY):
    with open(MANIFEST, encoding="utf-8") as f:
        man = [r for r in csv.DictReader(f)
               if r["looks_like_award"] == "True" or r["looks_like_report"] == "True"]
    os.makedirs(dest_dir, exist_ok=True)

    # skip anything already in the corpus, whichever directory we write to
    have = {n: os.path.getsize(os.path.join(PDF_DIR, n))
            for n in os.listdir(PDF_DIR)} if os.path.isdir(PDF_DIR) else {}

    log = []
    for i, r in enumerate(man, 1):
        name = f"{i:04d}_{r['filename']}"
        dest = os.path.join(dest_dir, name)
        size = have.get(name) or (os.path.getsize(dest) if os.path.exists(dest) else 0)
        if size and size != CAP:
            continue

        reason = "truncated" if size == CAP else "missing"
        got = 0
        if reason == "truncated":
            for cap in all_captures(r["url"])[:4]:
                if int(cap.get("length") or 0) <= CAP:
                    continue
                got = grab(r["url"], cap["timestamp"], dest)
                if got > CAP:
                    break
                time.sleep(delay)
        if not got:
            got = grab(r["url"], r["wayback_ts"], dest)

        log.append((name, reason, got))
        print(f"[{i}/{len(man)}] {reason:9} {got:>10,}  {name[:60]}", flush=True)
        time.sleep(delay)

    ok = sum(1 for _, reason, got in log
             if got > CAP or (reason == "missing" and got > 0))
    print(f"\n{len(log)} attempted, {ok} recovered -> {dest_dir}")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=PDF_DIR,
                    help="where to write; use a staging dir to avoid changing "
                         "the corpus under a running pipeline")
    download(ap.parse_args().dest)
