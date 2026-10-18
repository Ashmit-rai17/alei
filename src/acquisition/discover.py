"""Stage 1 — discovery: every PDF URL the Wayback Machine has seen on cioins.co.in.

The live site no longer links its award compilations (cioins.co.in/Download
now points at a per-complaint OTP tracker), so the CDX API is the only
systematic way to enumerate them.

Writes data/manifest.csv.
"""

import csv
import os
import re
from urllib.parse import unquote, urlparse

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
MANIFEST = os.path.join(DATA, "manifest.csv")

CDX = "http://web.archive.org/cdx/search/cdx"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; dataset construction)"}


def cdx_pdfs(domain="cioins.co.in"):
    params = {
        "url": domain,
        "matchType": "domain",
        "output": "json",
        "fl": "original,timestamp,statuscode,mimetype",
        "collapse": "urlkey",      # one row per distinct URL
        "filter": "mimetype:application/pdf",
        "limit": "20000",
    }
    r = requests.get(CDX, params=params, headers=HEADERS, timeout=180)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    return [dict(zip(header, row)) for row in data]


def classify(url):
    """Best-effort guess at centre / type / year from the URL itself.

    Nothing here is authoritative — it exists so the manifest is sortable.
    """
    low = unquote(url).lower()

    centres = [
        "ahmedabad", "bengaluru", "bangalore", "bhopal", "bhubaneswar",
        "chandigarh", "chennai", "delhi", "guwahati", "hyderabad",
        "jaipur", "ernakulam", "kochi", "kolkata", "lucknow", "mumbai",
        "noida", "patna", "pune", "thane",
    ]
    centre = next((c for c in centres if c in low), "")

    if "mediclaim" in low or "health" in low:
        ins_type = "health"
    elif "life" in low:
        ins_type = "life"
    elif "general" in low or "gi" in low:
        ins_type = "general"
    else:
        ins_type = ""

    year = re.search(r"((?:19|20)\d{2})", low)
    year = year.group(1) if year else ""

    looks_like_award = any(
        k in low for k in ("award", "book", "mediclaim", "judgement", "judgment")
    )
    looks_like_report = "annual" in low or "report" in low

    return centre, ins_type, year, looks_like_award, looks_like_report


def discover():
    print("querying wayback cdx ...")
    rows = cdx_pdfs()
    print(f"  {len(rows)} distinct pdf urls seen on the domain")

    seen, out = set(), []
    for row in rows:
        url = row["original"]
        if url in seen:
            continue
        seen.add(url)

        centre, ins_type, year, is_award, is_report = classify(url)
        out.append({
            "url": url,
            "wayback_ts": row["timestamp"],
            "archived_status": row["statuscode"],
            "centre": centre,
            "ins_type": ins_type,
            "year_guess": year,
            "looks_like_award": is_award,
            "looks_like_report": is_report,
            "filename": os.path.basename(urlparse(url).path) or "unnamed.pdf",
        })

    # awards first, then reports, then everything else
    out.sort(key=lambda d: (not d["looks_like_award"], not d["looks_like_report"], d["url"]))

    os.makedirs(DATA, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"  {sum(d['looks_like_award'] for d in out)} look like award compilations")
    print(f"  {sum(d['looks_like_report'] for d in out)} look like annual reports")
    print(f"  manifest -> {MANIFEST}")
    return out


if __name__ == "__main__":
    discover()
