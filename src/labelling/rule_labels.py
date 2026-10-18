"""Rule labels for every award with an extracted ground_span (no LLM).

  python scripts/rule_labels.py  ->  data/labels_rule.parquet

Same schema as data/labels.parquet, plus label_source = "rule". quote = the ground_span sentence
(checked against the award text like the LLM quotes); src = "rule" (ground_span is an extracted
sentence, its speaker is not classified).
"""

import os
import sys

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from rules import rule_grounds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
SAFE = os.path.join(DATA, "awards_safe.parquet")
OUT = os.path.join(DATA, "labels_rule.parquet")
AWARDS = os.path.join(DATA, "labeled_awards_rule.parquet")


def norm_ws(s):
    return " ".join(str(s).split())


def main():
    d = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "ground_span", "text"])
    d = d[d.ground_span.fillna("").str.strip() != ""].copy()
    d["uid"] = d.award_id + "|" + d.source_pdf
    rows, n_hit = [], 0
    for r in d.itertuples():
        gs = rule_grounds(r.ground_span)
        n_hit += bool(gs)
        ok = norm_ws(r.ground_span) in norm_ws(r.text)
        for g in gs:
            rows.append({"uid": r.uid, "award_id": r.award_id, "source_pdf": r.source_pdf, "ground_id": g,
                         "quote": r.ground_span, "quote_verified": ok, "src": "rule", "id_valid": True,
                         "description": "", "batch_id": "rule-v1", "label_source": "rule"})
    lab = pd.DataFrame(rows)
    lab.to_parquet(OUT, index=False)
    pd.DataFrame({"uid": d.uid}).to_parquet(AWARDS, index=False)
    per = lab.groupby("uid").size().reindex(d.uid, fill_value=0)
    print(f"ground_span awards {len(d):,}; mapped to >=1 ground {n_hit:,} ({n_hit / len(d):.1%}); "
          f"fall through {len(d) - n_hit:,} ({1 - n_hit / len(d):.1%}) -> {OUT}")
    print("grounds per award:", per.value_counts().sort_index().to_dict())
    print("per category (awards):", lab.ground_id.value_counts().to_dict())
    print("by claim_type, mapped share:",
          d.assign(hit=d.uid.isin(lab.uid)).groupby("claim_type").hit.agg(["sum", "count"]).to_dict("index"))
    return d, lab


if __name__ == "__main__":
    main()
