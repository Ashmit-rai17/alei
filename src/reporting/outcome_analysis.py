"""Outcome splits for repudiation awards: overall, by claim_type, by era, by top-5 insurer.

  python scripts/outcome_analysis.py   ->  data/outcome_analysis.json (+ printed tables)

Outcome is the heuristic in build_explorer.outcome (closing 2,500 chars). Two "in favour" readings:
  strict  = allowed + partly allowed
  broad   = allowed + partly allowed + ex gratia   (ex gratia: a goodwill payment, usually with the
            repudiation itself upheld; whether it counts as "in favour" is a definitional choice)
Shares are over awards with an extracted outcome; "not extracted" is reported beside every row.
"""

import json
import os
import sys

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from build_explorer import insurer_group, outcome  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
OC = ["dismissed", "allowed", "partly allowed", "ex gratia"]


def split(df, by):
    rows = []
    for key, g in df.groupby(by):
        x = g[g.oc.isin(OC)]
        n = len(x)
        c = x.oc.value_counts()
        row = {by: key, "n_with_outcome": n, "not_extracted": int((g.oc == "not extracted").sum())}
        for o in OC:
            row[o + "_%"] = round(100 * c.get(o, 0) / n, 1) if n else None
        row["in_favour_strict_%"] = round(100 * (c.get("allowed", 0) + c.get("partly allowed", 0)) / n, 1) if n else None
        row["in_favour_broad_%"] = round(100 * (n - c.get("dismissed", 0)) / n, 1) if n else None
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n_with_outcome"])


def main():
    d = pd.read_parquet(os.path.join(DATA, "awards_safe.parquet"),
                        columns=["award_id", "claim_type", "insurer", "award_date", "is_repudiation", "text"])
    d["oc"] = d.text.map(outcome)
    d["ins"] = d.insurer.map(insurer_group)
    y = pd.to_numeric(d.award_date.fillna("").str[:4], errors="coerce")
    d["era"] = pd.cut(y, [0, 2014, 2016, 2100], labels=["2003-2014", "2015-2016", "2018-2022"]).astype(str).replace(
        "nan", "no year")
    r = d[d.is_repudiation]
    top5 = r.ins[r.ins != "(not extracted)"].value_counts().head(5).index
    out = {
        "method": __doc__.strip(),
        "repudiation_overall": split(r.assign(all="all"), "all"),
        "all_awards_overall": split(d.assign(all="all"), "all"),
        "by_claim_type": split(r, "claim_type"),
        "by_era": split(r, "era"),
        "by_era_life_death_only": split(r[r.claim_type == "life_death"], "era"),
        "by_era_excluding_health": split(r[r.claim_type != "health"], "era"),
        "by_insurer_top5": split(r[r.ins.isin(top5)], "ins"),
    }
    json.dump(out, open(os.path.join(DATA, "outcome_analysis.json"), "w"), indent=1)
    cols = ["n_with_outcome", "not_extracted", "dismissed_%", "allowed_%", "partly allowed_%", "ex gratia_%",
            "in_favour_strict_%", "in_favour_broad_%"]
    for k, v in out.items():
        if k == "method":
            continue
        t = pd.DataFrame(v)
        print(f"\n== {k}")
        print(t[[t.columns[0]] + cols].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
