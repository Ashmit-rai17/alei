"""Stage 6 — the 80-record labeling sheet.

Reads ONLY data/awards_safe.parquet. Never opens awards.parquet.
Writes data/labeling_sample.xlsx with sheets: labeling, instructions,
candidate_grounds.
"""

import os
import random
import re
import sys
from collections import Counter

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from inventory import DATA  # noqa: E402

SAFE = os.path.join(DATA, "awards_safe.parquet")
RAW = os.path.join(DATA, "awards.parquet")          # only stat()ed, never read
OUT = os.path.join(DATA, "labeling_sample.xlsx")

SEED = 20261017
N = 79
# life_maturity and life_survival are 0 by decision (2026-10-17): one maturity
# row is noise, and survival-benefit disputes are not repudiations. They stay
# in the dict (at 0) so the seeded draw order — and the other rows — is
# unchanged from the 80-row version.
TARGETS = {"life_death": 25, "health": 20, "life_maturity": 0, "life_misc": 10, "life_survival": 0}
# where a shortfall goes, in order: scarce strata first, life_death last
REFILL_ORDER = ["life_misc", "health", "life_death"]
MIN_CHARS, MAX_CHARS = 2000, 12000

COLUMNS = ["award_id", "claim_type", "insurer", "award_date", "ground_span", "award_text",
           "ground_1", "ground_2", "ground_3", "ground_4", "ground_5", "corrected", "notes"]

INSTRUCTIONS = (
    "Read the award. Identify every ground the insurer gave for rejecting the claim — not the "
    "Ombudsman's opinion of it, not who won. More than one ground is normal. ground_span is an "
    "automatic guess; check it against the text and correct it. Mark corrected yes if you changed "
    "anything. Fill notes whenever you're unsure."
)

TRIGGER = re.compile(r"\b(?:on the grounds?(?: of| that)?|stating that|on account of)\b\s*", re.I)


def load_safe():
    """Gate 3: the sample must come from the redacted file, and a fresh one."""
    path = os.path.realpath(SAFE)
    assert os.path.basename(path) == "awards_safe.parquet", path
    if not os.path.exists(path):
        sys.exit(f"{path} missing — run redact first")
    if os.path.exists(RAW) and os.path.getmtime(path) < os.path.getmtime(RAW):
        sys.exit("awards_safe.parquet is OLDER than awards.parquet — it is stale. Re-run redact.")
    df = pd.read_parquet(path)
    redacted = df.text.str.contains(r"\[(?:NAME|NUM|POLICY_NO|PIN|NAME_ADDRESS|ADDRESS)\]").mean()
    if redacted < 0.5:
        sys.exit(f"only {redacted:.0%} of rows carry redaction placeholders — is this really the safe file?")
    print(f"GATE 3: sampling from {path}")
    print(f"        {len(df):,} rows, {redacted:.1%} contain redaction placeholders; awards.parquet not opened")
    return df


def allocate(pool):
    avail = pool.claim_type.value_counts().to_dict()
    alloc = {k: min(v, avail.get(k, 0)) for k, v in TARGETS.items()}
    short = N - sum(alloc.values())
    for k in REFILL_ORDER:
        if short <= 0:
            break
        add = min(short, avail.get(k, 0) - alloc[k])
        alloc[k] += add
        short -= add
    return alloc, avail


def spread_pick(stratum, n, rng):
    """Greedy: next row is the one whose source_pdf and insurer are least used so far."""
    rows = stratum.sample(frac=1, random_state=rng.randrange(2**31)).to_dict("records")
    by_pdf, by_ins, picked = Counter(), Counter(), []
    for _ in range(min(n, len(rows))):
        best = min(range(len(rows)),
                   key=lambda i: (by_pdf[rows[i]["source_pdf"]], by_ins[rows[i]["insurer"] or "?"], i))
        r = rows.pop(best)
        by_pdf[r["source_pdf"]] += 1
        by_ins[r["insurer"] or "?"] += 1
        picked.append(r)
    return picked


def ground_phrase(span):
    m = TRIGGER.search(span)
    tail = span[m.end():] if m else span
    return re.sub(r"\s+", " ", tail).strip(" .").lower()


def clean(v):
    return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v


def sample():
    df = load_safe()
    pool = df[(df.ground_span != "")
              & (df.n_chars > MIN_CHARS) & (df.n_chars < MAX_CHARS)
              & df.claim_type.isin(TARGETS)
              & (df.n_headers < 2)]          # suspected welds: two awards in one row
    alloc, avail = allocate(pool)

    rng = random.Random(SEED)
    picked = []
    for k in TARGETS:
        picked += spread_pick(pool[pool.claim_type == k], alloc[k], rng)
    out = pd.DataFrame(picked).rename(columns={"text": "award_text"})
    for c in COLUMNS[6:]:
        out[c] = ""
    out = out[COLUMNS].map(clean)

    print(f"\npool: ground_span present, {MIN_CHARS} < n_chars < {MAX_CHARS}, misselling excluded, "
          f"suspected welds excluded -> {len(pool)} rows")
    print(f"{'claim_type':15} {'target':>6} {'avail':>6} {'drawn':>6} {'pdfs':>5} {'insurers':>8}")
    for k in TARGETS:
        sub = out[out.claim_type == k]
        print(f"{k:15} {TARGETS[k]:>6} {avail.get(k, 0):>6} {len(sub):>6} "
              f"{pd.DataFrame(picked)[lambda d: d.claim_type == k].source_pdf.nunique() if len(sub) else 0:>5} "
              f"{sub.insurer.nunique():>8}")
    print(f"total drawn {len(out)}   seed {SEED}")

    # candidate grounds from the full (redacted) dataset
    g = df[df.ground_span != ""].ground_span.map(lambda s: re.sub(r"\s+", " ", s).strip())
    phr = g.map(ground_phrase)
    pf = phr.value_counts()
    cand = (pd.DataFrame({"ground_span": g, "ground_phrase": phr})
              .groupby(["ground_span", "ground_phrase"]).size().rename("frequency").reset_index())
    cand["phrase_frequency"] = cand.ground_phrase.map(pf)
    cand = cand.sort_values(["phrase_frequency", "frequency", "ground_phrase"], ascending=[False, False, True])
    cand = cand[["ground_span", "frequency", "ground_phrase", "phrase_frequency"]].map(clean)

    instr = pd.DataFrame({"instructions": [
        INSTRUCTIONS, "",
        "Columns ground_1 … ground_5, corrected (yes/no) and notes are yours to fill. Do not reorder columns.",
        "Text is redacted: [NAME], [ADDRESS], [NAME_ADDRESS], [HOSPITAL], [POLICY_NO], [NUM], [PIN], "
        "[PHONE], [EMAIL] replace personal details. If you still see a real person's name, address or "
        "policy number, note it in notes.",
        "", f"Sample: {len(out)} awards, stratified by claim_type, seed {SEED}, drawn from awards_safe.parquet.",
        "Drawn per claim_type: " + ", ".join(f"{k} {int((out.claim_type == k).sum())}" for k in TARGETS),
    ]})

    assert len(out) == N, f"drew {len(out)}, expected {N}"
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="labeling", index=False)
        instr.to_excel(xw, sheet_name="instructions", index=False)
        cand.to_excel(xw, sheet_name="candidate_grounds", index=False)

        ws = xw.sheets["labeling"]
        widths = dict(award_id=24, claim_type=14, insurer=18, award_date=12, ground_span=60,
                      award_text=100, ground_1=30, ground_2=30, ground_3=30, ground_4=30,
                      ground_5=30, corrected=11, notes=40)
        for j, c in enumerate(COLUMNS, 1):
            col = ws.cell(row=1, column=j).column_letter
            ws.column_dimensions[col].width = widths[c]
            ws.cell(row=1, column=j).font = Font(bold=True)
            if j >= 7:
                ws.cell(row=1, column=j).fill = PatternFill("solid", fgColor="FFF2CC")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2"
        dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"L2:L{len(out) + 1}")

        wi = xw.sheets["instructions"]
        wi.column_dimensions["A"].width = 120
        for row in wi.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        wc = xw.sheets["candidate_grounds"]
        for col, w in zip("ABCD", (90, 11, 70, 16)):
            wc.column_dimensions[col].width = w
        wc.freeze_panes = "A2"

    print(f"\ncandidate_grounds: {len(cand)} distinct ground_span values, "
          f"{cand.ground_phrase.nunique()} distinct phrases")
    print(f"-> {OUT}")
    return out


if __name__ == "__main__":
    sample()
