"""Corpus statistics for the deck: data/corpus_stats.json + data/corpus_stats.md.

Reads manifest.csv, inventory.parquet and awards_safe.parquet. No labels needed.
Outcome and centre are text heuristics; their coverage is reported next to every figure.
"""

import json
import os
import re
import sys
import time

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from build_explorer import insurer_group, outcome  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo

CODES = {"AHD": "Ahmedabad", "BNG": "Bengaluru", "BHP": "Bhopal", "BPL": "Bhopal", "BHU": "Bhubaneswar",
         "CHD": "Chandigarh", "CHN": "Chennai", "DEL": "Delhi", "GUW": "Guwahati", "GHY": "Guwahati",
         "HYD": "Hyderabad", "JPR": "Jaipur", "KOC": "Kochi", "KCH": "Kochi", "ERN": "Kochi", "KOL": "Kolkata",
         "LCK": "Lucknow", "LKO": "Lucknow", "MUM": "Mumbai", "NOI": "Noida", "PAT": "Patna", "PUN": "Pune",
         "PUNE": "Pune", "THN": "Thane", "IND": "Bhopal"}
CITIES = {"ahmedabad": "Ahmedabad", "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bhopal": "Bhopal",
          "bhubaneswar": "Bhubaneswar", "bhubaneshwar": "Bhubaneswar", "chandigarh": "Chandigarh",
          "chennai": "Chennai", "delhi": "Delhi", "guwahati": "Guwahati", "hyderabad": "Hyderabad",
          "jaipur": "Jaipur", "kochi": "Kochi", "ernakulam": "Kochi", "kolkata": "Kolkata", "lucknow": "Lucknow",
          "mumbai": "Mumbai", "noida": "Noida", "patna": "Patna", "pune": "Pune", "thane": "Thane"}
CITY_RX = "|".join(CITIES)
CENTRE_TEXT = re.compile(
    rf"(?i)ombudsman[ ,\-–]*(?:centre|center|office)?[ ,\-–:]*\(?({CITY_RX})\b|\b({CITY_RX})\s+ombudsman\s+(?:centre|center|office)|"
    rf"\bIO\s*\(\s*([A-Z]{{3,4}})\s*\)|ombudsman\s*\(\s*({CITY_RX})\s*\)")


def centre(award_id, text):
    m = re.match(r"\s*([A-Z]{3,4})[-/ ]", award_id or "")
    if m and m.group(1) in CODES:
        return CODES[m.group(1)], "award_id"
    m = re.search(r"/(KCH|LKO|GHY|IND|MUM|HYD|KOL|CHN|BNG|AHD|JPR|PAT|PUN)\b", award_id or "")
    if m and m.group(1) in CODES:
        return CODES[m.group(1)], "award_id"
    for m in CENTRE_TEXT.finditer(text or ""):
        city = next(g for g in m.groups() if g)
        if city.upper() in CODES:
            return CODES[city.upper()], "text"
        if city.lower() in CITIES:
            return CITIES[city.lower()], "text"
    return None, None


def table(s, total, top=None):
    s = s.value_counts()
    if top:
        s = s.head(top)
    return [{"value": str(k), "awards": int(v), "of": int(total), "share": round(v / total, 4)} for k, v in s.items()]


def main():
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    inv = pd.read_parquet(os.path.join(DATA, "inventory.parquet"))
    d = pd.read_parquet(os.path.join(DATA, "awards_safe.parquet"),
                        columns=["award_id", "source_pdf", "claim_type", "insurer", "award_date", "is_repudiation",
                                 "ground_span", "text"])
    n = len(d)
    cen = [centre(a, t) for a, t in zip(d.award_id, d.text)]
    d["centre"] = [c for c, _ in cen]
    d["centre_src"] = [s for _, s in cen]
    d["outcome"] = d.text.map(outcome)
    d["ins"] = d.insurer.map(insurer_group)
    d["year"] = d.award_date.fillna("").str[:4].where(lambda s: s.str.fullmatch(r"(19|20)\d\d"))
    rep = d[d.is_repudiation]

    readable = int((~inv.broken & inv.has_text).sum()) if "has_text" in inv else None
    stats = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "caveat": ("Contested denials that reached an Insurance Ombudsman — a small fraction of all repudiations. "
                   "These figures describe challenged grounds, not typical ones."),
        "provenance": {
            "archived_urls": int(len(man)),
            "pdfs": int(len(inv)),
            "pdfs_readable_text": readable,
            "pdfs_scanned_no_text": int((~inv.broken & ~inv.has_text).sum()),
            "pdfs_broken": int(inv.broken.sum()),
            "pdfs_truncated": int(inv.truncated.sum()),
            "pages": int(inv.pages.fillna(0).sum()),
            "pages_in_clean_pdfs": int(inv[inv.clean].pages.fillna(0).sum()),
            "clean_pdfs": int(inv.clean.sum()),
            "awards": n,
            "repudiation_awards": int(d.is_repudiation.sum()),
            "awards_with_ground_sentence": int((d.ground_span.fillna("").str.strip() != "").sum()),
            "source_books_with_awards": int(d.source_pdf.nunique()),
            "unreadable_zero_pages": int((inv.broken & inv.note.str.contains("no pages")).sum()),
            "unreadable_fail_to_open": int((inv.broken & ~inv.note.str.contains("no pages")).sum()),
            "unreadable_all_at_archive_cap": bool((inv[inv.broken]["size"] == 1_048_576).all()),
        },
        "provenance_definitions": {
            "pdfs_readable_text": "opens, and yields more than 500 characters of text (text layer present)",
            "pdfs_scanned_no_text": "opens and has pages, but yields <= 500 characters: image-only scans (no OCR run)",
            "pdfs_broken": ("cannot be read at all: either opens with zero pages or fails to open. All are exactly "
                            "1,048,576 bytes, the Internet Archive's size cap, i.e. truncated downloads, not scans"),
            "pdfs_truncated": ("archive cap hit, missing %%EOF, or xref repaired on open; includes the broken ones; "
                               "the rest are partly readable and their readable pages are used"),
            "clean_pdfs": "readable text and not truncated",
            "pages": "page count over all PDFs that open (broken ones contribute 0)",
            "reconciliation": ("an earlier count of 8 scanned / 5 broken put the 5 zero-page truncated files under "
                               "'scanned'; they are truncated downloads, so the authoritative split is 3 scanned / "
                               "10 broken"),
        },
        "claim_type": table(d.claim_type, n),
        "claim_type_repudiation": table(rep.claim_type, len(rep)),
        "is_repudiation": {"true": int(d.is_repudiation.sum()), "false": int((~d.is_repudiation).sum()), "of": n},
        "insurer_top15": table(d.ins, n, 15),
        "insurer_top15_repudiation": table(rep.ins, len(rep), 15),
        "insurer_not_extracted": int((d.ins == "(not extracted)").sum()),
        "year": sorted(table(d.year.dropna(), n), key=lambda r: r["value"]),
        "year_not_extracted": int(d.year.isna().sum()),
        "centre": table(d.centre.dropna(), n),
        "centre_coverage": {"from_award_id": int((d.centre_src == "award_id").sum()),
                            "from_text": int((d.centre_src == "text").sum()),
                            "not_extracted": int(d.centre.isna().sum()), "of": n},
        "outcome_all": table(d.outcome, n),
        "outcome_repudiation": table(rep.outcome, len(rep)),
        "outcome_method": ("heuristic on the last 2,500 chars: latest of dismissed / allowed / partly allowed / "
                           "ex gratia ('not allowed' counts as dismissed; partly allowed and ex gratia win when "
                           "present); 'not extracted' when none appears. Spot-checked: 29 of 32 sampled awards "
                           "correct after fixing 'not allowed' / 'allowed partially'; 1 ambiguous. Splits by "
                           "claim type, era and insurer: data/outcome_analysis.json"),
    }
    lic_rep = float((rep.ins == "LIC of India").mean())
    stats["caveats"] = [
        {"applies_to": "year", "text": f"{d.year.isna().mean():.0%} of awards ({int(d.year.isna().sum()):,} of {n:,}) "
         "have no extracted award year. No award is dated 2017 and one is dated 2016: the archived compilations are "
         "older 'Books' running to 2015 and monthly lists from 2018, and none of them covers awards decided in "
         "2016-17 (complaints filed then were decided in 2018-19 and are in the corpus). Do not draw a time trend "
         "across the gap."},
        {"applies_to": "insurer", "text": f"{(d.ins == '(not extracted)').mean():.0%} of awards "
         f"({int((d.ins == '(not extracted)').sum()):,}) have no extracted insurer."},
        {"applies_to": "outcome", "text": f"{(rep.outcome == 'not extracted').mean():.0%} of repudiation awards "
         f"({int((rep.outcome == 'not extracted').sum()):,} of {len(rep):,}) have no extracted outcome; outcome "
         "is a text heuristic, not a field."},
        {"applies_to": "insurer", "text": f"LIC of India is {lic_rep:.0%} of repudiation awards "
         f"({int((rep.ins == 'LIC of India').sum()):,} of {len(rep):,}); any 'all insurers' figure is mostly LIC."},
        {"applies_to": "claim_type", "text": "health is 4% of repudiation awards and comes from a single book "
         "(Apr 2021); life_death is 83%."},
        {"applies_to": "centre", "text": f"{d.centre.isna().mean():.0%} of awards have no extracted Ombudsman centre."},
        {"applies_to": "all", "text": stats["caveat"]},
    ]
    json.dump(stats, open(os.path.join(DATA, "corpus_stats.json"), "w"), indent=1)

    def md_table(rows, label):
        out = [f"| {label} | awards | of | share |", "|---|---:|---:|---:|"]
        out += [f"| {r['value']} | {r['awards']:,} | {r['of']:,} | {r['share']:.1%} |" for r in rows]
        return "\n".join(out)

    p = stats["provenance"]
    md = f"""# Alei corpus statistics

Generated {stats['generated']}. Source: `data/corpus_stats.json`.

> {stats['caveat']}

## Provenance

{p['archived_urls']:,} archived URLs → {p['pdfs']:,} PDFs ({p['pdfs_broken']} broken, {p['pdfs_scanned_no_text']} scanned with no text,
{p['pdfs_truncated']} truncated by the archive) → {p['pages']:,} pages ({p['pages_in_clean_pdfs']:,} of them in the
{p['clean_pdfs']} clean, untruncated PDFs; awards are also read from the readable part of truncated ones) → **{p['awards']:,} awards** from {p['source_books_with_awards']} books →
**{p['repudiation_awards']:,} repudiation awards** ({p['awards_with_ground_sentence']:,} with an extracted ground sentence).

## Repudiation

{stats['is_repudiation']['true']:,} of {n:,} awards ({stats['is_repudiation']['true'] / n:.1%}) concern a repudiated or rejected claim.

## Claim type (all awards)

{md_table(stats['claim_type'], 'claim type')}

## Claim type (repudiation awards)

{md_table(stats['claim_type_repudiation'], 'claim type')}

## Insurer, top 15 (all awards)

{md_table(stats['insurer_top15'], 'insurer')}

Spelling variants merged, plus known renames (HDFC Standard Life → HDFC Life, Tata AIG Life → Tata AIA Life,
Kotak Life → Kotak Mahindra Life). "Bajaj Allianz" and "Bajaj Allianz Life" are left as written (the header does not say which
company). Insurer not extracted from the header: {stats['insurer_not_extracted']:,} awards.

## Year of award

{md_table(stats['year'], 'year')}

No award date extracted: {stats['year_not_extracted']:,} awards.

## Ombudsman centre

{md_table(stats['centre'], 'centre')}

Coverage: {stats['centre_coverage']['from_award_id']:,} from the complaint number, {stats['centre_coverage']['from_text']:,} from the
award text, {stats['centre_coverage']['not_extracted']:,} not extracted.

## Outcome (heuristic)

All awards:

{md_table(stats['outcome_all'], 'outcome')}

Repudiation awards:

{md_table(stats['outcome_repudiation'], 'outcome')}

Method: {stats['outcome_method']}.
"""
    open(os.path.join(DATA, "corpus_stats.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
