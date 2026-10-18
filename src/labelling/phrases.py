"""Task 5 — taxonomy input: data/ground_phrases.csv.

From awards_safe.parquet (redacted) where is_repudiation is True:
  1. take the clause after "on the ground(s) of / on the ground(s) that /
     stating that", and the value of a "Reason for repudiation" form field;
  2. normalise: lowercase, drop redaction tags, dates, amounts, numbers, punctuation;
  3. count 2–6 word phrases, once per award (document frequency), so one
     verbose award cannot dominate;
  4. drop phrases that start/end on a function word, and a phrase whose count
     equals that of a longer phrase containing it (it adds nothing);
  5. write the top 200 with counts and two example award_ids.
"""

import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from inventory import DATA  # noqa: E402
from segment import _mask_soft_dots  # noqa: E402

SAFE = os.path.join(DATA, "awards_safe.parquet")
OUT = os.path.join(DATA, "ground_phrases.csv")
TOP = 200

TRIGGER = re.compile(r"(?i)\bon\s+the\s+grounds?\s+(?:of|that)\b|\bstating\s+that\b")
REASON_FIELD = re.compile(r"(?i)reason\s+for\s+(?:the\s+)?repudiation(?:\s*/\s*rejection)?[ \t:\-]*\n?\s*")
NEXT_ITEM = re.compile(r"\n[ \t]*\d{1,2}[ \t]*[.)]?[ \t]*[A-Z]|\n\s*\n\s*\n")
SENT_END = re.compile(r"\.(?!\d)|\n\s*\n")
FORM_LABEL = re.compile(
    r"(?i)^\d{0,2}\s*[.)]?\s*(?:date\s+of|nature\s+of|amount\s+of|name\s+of|type\s+of|policy\b|"
    r"duration|sum\s+insured|complaint\s+(?:registered|how)|representation|reason\s+for|"
    r"disposal|place\s+of|period\s+of)")
NA = re.compile(r"(?i)^(?:na|n\.?a\.?|nil|not applicable|-+)$")

STOP = set("""a an the of and or to in on at by for from with as that which who whom this these those
is was were be been being has had have he she his her him it its they their them there here
any all such so not no nor but if then than also into upon under over per vide i e viz
""".split())


def clauses(text):
    masked = _mask_soft_dots(text)
    out = []
    for m in TRIGGER.finditer(masked):
        end = SENT_END.search(masked, m.end())
        out.append(text[m.end():min(end.start() if end else len(text), m.end() + 300)])
    for m in REASON_FIELD.finditer(text):
        # the value is the line(s) right after the label, up to the next form
        # label or numbered item; forms often leave items unnumbered
        val = []
        for line in text[m.end():m.end() + 400].split("\n"):
            line = line.strip()
            if not line:
                if val:
                    break
                continue
            if FORM_LABEL.match(line) or re.match(r"^\d{1,2}\s*[.)]?\s*$", line):
                break
            val.append(line)
            if len(val) == 3:
                break
        v = " ".join(val)
        if v and not NA.match(v):
            out.append(v)
    return out


def normalise(s):
    s = re.sub(r"\[[A-Z_]+\]", " ", s)                                   # redaction tags
    s = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", s)            # dates
    s = re.sub(r"(?i)\b(?:rs|inr)\.?\s*[\d,]+(?:\.\d+)?(?:/-)?", " ", s)  # amounts
    s = s.lower().replace("\u2024", ".")
    s = re.sub(r"[^a-z\s]", " ", s)                                       # punctuation, digits
    return re.sub(r"\s+", " ", s).strip()


def ngrams(words, lo=2, hi=6):
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            g = words[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            yield " ".join(g)


def phrases():
    df = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "is_repudiation", "text"])
    rep = df[df.is_repudiation]
    df_count, examples = Counter(), defaultdict(list)
    n_with = 0
    for aid, text in zip(rep.award_id, rep.text):
        cl = clauses(text)
        if not cl:
            continue
        n_with += 1
        grams = set()
        for c in cl:
            grams.update(ngrams(normalise(c).split()))
        for g in grams:
            df_count[g] += 1
            if len(examples[g]) < 2 and aid not in examples[g]:
                examples[g].append(aid)

    # closed phrases only: drop g if a longer phrase containing it has the same count
    keep = {}
    by_len = sorted(df_count, key=lambda g: -len(g.split()))
    longer = defaultdict(int)          # best count of any longer phrase containing g
    for g in by_len:
        if df_count[g] < 3:
            continue
        if longer[g] != df_count[g]:
            keep[g] = df_count[g]
        w = g.split()
        for n in range(2, len(w)):
            for i in range(len(w) - n + 1):
                sub = " ".join(w[i:i + n])
                longer[sub] = max(longer[sub], df_count[g])

    top = sorted(keep.items(), key=lambda kv: (-kv[1], -len(kv[0].split()), kv[0]))[:TOP]
    out = pd.DataFrame([{"phrase": g, "n_words": len(g.split()), "awards": c,
                         "example_1": examples[g][0], "example_2": examples[g][1] if len(examples[g]) > 1 else ""}
                        for g, c in top])
    out.to_csv(OUT, index=False)
    print(f"repudiation awards: {len(rep):,}; with at least one extracted clause: {n_with:,} ({n_with / len(rep):.1%})")
    print(f"distinct 2-6 word phrases: {len(df_count):,}; closed, in 3+ awards: {len(keep):,}; written: {len(out)} -> {OUT}")
    print(out.head(40).to_string(index=False))
    return out


if __name__ == "__main__":
    phrases()
