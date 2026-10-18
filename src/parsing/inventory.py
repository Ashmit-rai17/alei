"""Stage 3 — inventory: which PDFs in data/pdfs are usable.

Reads only data/pdfs/. Never downloads. Writes data/inventory.parquet.
"""

import os
import re
import warnings
from collections import Counter

import pandas as pd
import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)   # the truncated files are noisy
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
PDF_DIR = os.path.join(DATA, "pdfs")
INVENTORY = os.path.join(DATA, "inventory.parquet")

CAP = 1048576          # the Internet Archive's size cap
YEAR_MIN, YEAR_MAX = 1995, 2026


# ------------------------------------------------------------
# claim type, straight off the filename
# ------------------------------------------------------------

CLAIM_TYPE = [
    ("health", "health"), ("mediclaim", "health"), ("hospital", "health"),
    ("deathclaim", "life_death"), ("death", "life_death"), ("dealth", "life_death"),
    ("maturity", "life_maturity"),
    ("misselling", "life_misselling"), ("miselling", "life_misselling"),
    ("mis-selling", "life_misselling"), ("mis_selling", "life_misselling"),
    ("mis selling", "life_misselling"), ("mis%20selling", "life_misselling"),
    # spellings seen in the corpus and the Wayback downloads
    ("miss-selling", "life_misselling"), ("miss selling", "life_misselling"),
    ("miss%20selling", "life_misselling"), ("missselling", "life_misselling"),
    ("missell", "life_misselling"),
    ("survival", "life_survival"), ("survivalbenefit", "life_survival"),
    ("sb_", "life_survival"), ("sb%20", "life_survival"), ("%20sb.", "life_survival"),
    ("misclife", "life_misc"), ("lifemisc", "life_misc"),
    ("miscelleanous", "life_misc"), ("miscellaneous", "life_misc"),
    ("miscllaneous", "life_misc"), ("misc", "life_misc"),
]


def claim_type(fn):
    low = fn.lower()
    for key, val in CLAIM_TYPE:
        if key in low:
            return val
    return "unknown"


# ------------------------------------------------------------
# safe text extraction — truncated PDFs open fine then die mid-tree
# ------------------------------------------------------------

def page_text(path, max_pages=None):
    doc = pymupdf.open(path)
    out = []
    try:
        n = min(doc.page_count, max_pages) if max_pages else doc.page_count
        for p in range(n):
            try:
                out.append(doc[p].get_text())
            except Exception:
                break          # stop at the first dead page, keep what we have
    finally:
        try: doc.close()
        except Exception: pass
    return "\n".join(out)


# ------------------------------------------------------------
# dates -> years (never a bare four-digit number: rupee amounts match that)
# ------------------------------------------------------------

DATE_NUM = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
DATE_ORD = re.compile(r"\b(\d{1,2})\s*(?:ST|ND|RD|TH)?\s+"
                      r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+(\d{4})\b", re.I)


def date_years(text):
    yrs = []
    for m in DATE_NUM.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and YEAR_MIN <= y <= YEAR_MAX:
            yrs.append(y)
    for m in DATE_ORD.finditer(text):
        d, y = int(m.group(1)), int(m.group(3))
        if 1 <= d <= 31 and YEAR_MIN <= y <= YEAR_MAX:
            yrs.append(y)
    return yrs


def filename_year(fn):
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", fn)
    if m and YEAR_MIN <= int(m.group(1)) <= YEAR_MAX:
        return int(m.group(1))
    return None


# ------------------------------------------------------------

def inspect(path):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    rec = dict(file=name, claim_type=claim_type(name), size=size, size_cap=size == CAP,
               no_eof=False, repaired=False,
               pages=0, chars=0, has_text=False, last_page_ok=False, truncated=False,
               broken=False, year=None, year_source="", note="")
    with open(path, "rb") as f:
        f.seek(max(0, size - 2048))
        rec["no_eof"] = b"%%EOF" not in f.read()

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        rec.update(broken=True, truncated=rec["no_eof"] or rec["size_cap"],
                   note=f"BROKEN: {str(e)[:60]}")
        return rec

    try:
        rec["repaired"] = bool(doc.is_repaired)
        rec["pages"] = doc.page_count
        if rec["pages"] == 0:
            rec.update(broken=True, truncated=rec["no_eof"] or rec["repaired"] or rec["size_cap"],
                       note="BROKEN: no pages")
            return rec

        text = ""
        for p in range(min(rec["pages"], 8)):
            try:
                text += doc[p].get_text()
            except Exception:
                break
        rec["chars"] = len(text.strip())
        rec["has_text"] = rec["chars"] > 500

        # real truncation check: the last page must actually load
        try:
            doc[rec["pages"] - 1].get_text()
            rec["last_page_ok"] = True
        except Exception:
            rec["last_page_ok"] = False
    except Exception as e:
        rec.update(broken=True, truncated=True, note=f"BROKEN: {str(e)[:60]}")
        return rec
    finally:
        try: doc.close()
        except Exception: pass

    # pymupdf rebuilds the xref of a cut-off file on open, so the last page
    # it reports usually loads fine. The structural signals (xref had to be
    # repaired, no %%EOF trailer) catch what the last-page read cannot; the
    # 1,048,576-byte size cap is kept as a secondary signal.
    rec["truncated"] = (not rec["last_page_ok"]) or rec["repaired"] or rec["no_eof"] or rec["size_cap"]

    yrs = date_years(text)
    if yrs:
        rec["year"] = Counter(yrs).most_common(1)[0][0]
        rec["year_source"] = "dates"
    elif filename_year(name):
        rec["year"] = filename_year(name)
        rec["year_source"] = "filename"

    if not rec["has_text"]:
        rec["note"] = "SCANNED"
    elif rec["truncated"]:
        rec["note"] = "TRUNCATED"
    return rec


def inventory():
    names = sorted(n for n in os.listdir(PDF_DIR) if n.lower().endswith(".pdf"))
    df = pd.DataFrame([inspect(os.path.join(PDF_DIR, n)) for n in names])
    df["year"] = df["year"].astype("Int64")
    df["clean"] = df.has_text & ~df.truncated & ~df.broken
    df.to_parquet(INVENTORY, index=False)

    print(f"pdfs on disk        {len(df)}")
    print(f"readable text       {df.has_text.sum()}")
    print(f"scanned             {(df.note == 'SCANNED').sum()}")
    print(f"broken              {df.broken.sum()}")
    print(f"truncated           {df.truncated.sum()}  (of which broken: {(df.truncated & df.broken).sum()}, "
          f"partially readable: {(df.truncated & ~df.broken).sum()})")
    print(f"  signals: last page fails {(~df.last_page_ok & ~df.broken).sum()}, xref repaired {df.repaired.sum()}, "
          f"no %%EOF {df.no_eof.sum()}, size == cap {df.size_cap.sum()}")
    print(f"CLEAN + READABLE    {df.clean.sum()}")
    print(f"pages (all)         {df.pages.sum():,}")
    print(f"pages (readable)    {df[df.has_text].pages.sum():,}")
    print(f"pages (clean)       {df[df.clean].pages.sum():,}")
    yrs = df[df.has_text].year.dropna()
    print(f"years (readable)    {yrs.min()} - {yrs.max()}   "
          f"(from dates: {(df.year_source == 'dates').sum()}, filename: {(df.year_source == 'filename').sum()})")
    print("\nby claim type (readable):")
    print(df[df.has_text].claim_type.value_counts().to_string())
    return df


if __name__ == "__main__":
    inventory()
