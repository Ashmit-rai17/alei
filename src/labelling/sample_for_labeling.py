"""Pick the 1,200 awards Qwen labels (Groq free tier: 8k tokens/min rules out all 3,831).

  python scripts/sample_for_labeling.py  ->  data/label_sample.csv

From the is_repudiation awards in awards_safe.parquet:
  1. prashant    every award in Prashant's sheets (phase1_read + phase2_label, 79 rows), matched on
                 award_id, falling back to the start of the award text when an id is missing or
                 ambiguous. Matched awards that are not is_repudiation are still included, and listed.
  2. health      every health award (the only health source: one book)
  3. stratified  the rest, proportional to the full repudiation set's claim_type shares
                 (excluding health, which is already taken whole), allocated across source_pdf in
                 proportion to each book's size within the claim type (largest remainder), then
                 drawn at random inside each (claim_type, book) cell.
"""

import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
SAFE = os.path.join(DATA, "awards_safe.parquet")
SHEETS = [os.path.join(DATA, "prashant", "phase1_read.xlsx"), os.path.join(DATA, "prashant", "phase2_label.xlsx")]
OUT = os.path.join(DATA, "label_sample.csv")
N = 1200
SEED = 20261018


def norm_ws(s):
    return " ".join(str(s).split())


def largest_remainder(weights, total):
    w = np.asarray(weights, float)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base


def main():
    safe = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "is_repudiation", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    rep = safe[safe.is_repudiation]

    # 1. Prashant's 79
    sheet = pd.concat([pd.read_excel(p, sheet_name=0)[["award_id", "award_text"]] for p in SHEETS], ignore_index=True)
    by_id = safe.groupby("award_id").uid.apply(list).to_dict()
    by_text = {norm_ws(t)[:400]: u for u, t in zip(safe.uid, safe.text)}
    found, missing, how = [], [], {"id": 0, "text": 0}
    for a, t in zip(sheet.award_id.astype(str), sheet.award_text):
        cand = by_id.get(a, [])
        key = norm_ws(t)[:400]
        if len(cand) == 1:
            found.append(cand[0]); how["id"] += 1
        elif key in by_text:
            found.append(by_text[key]); how["text"] += 1
        else:
            missing.append(a)
    found = list(dict.fromkeys(found))
    not_rep = [u for u in found if not safe.set_index("uid").at[u, "is_repudiation"]]
    print(f"Prashant's sheets: {len(sheet)} rows; found {len(found)} in corpus (by id {how['id']}, by text {how['text']}); "
          f"NOT FOUND {len(missing)}: {missing}")
    if not_rep:
        print(f"  {len(not_rep)} found but not is_repudiation (included anyway): {not_rep}")

    # 2. all health
    health = [u for u in rep.uid[rep.claim_type == "health"] if u not in found]

    # 3. stratified fill
    taken = set(found) | set(health)
    rest = N - len(taken)
    pool = rep[~rep.uid.isin(taken) & (rep.claim_type != "health")]
    shares = rep[rep.claim_type != "health"].claim_type.value_counts()
    forced_by_type = safe.set_index("uid").loc[list(found)].claim_type.value_counts()
    # targets per claim type for the whole sample's non-health part, minus what Prashant's rows already supply
    nonhealth_total = rest + int(forced_by_type.drop("health", errors="ignore").sum())
    tgt = pd.Series(largest_remainder(shares.values, nonhealth_total), index=shares.index)
    need = (tgt - forced_by_type.reindex(tgt.index, fill_value=0)).clip(lower=0)
    need = pd.Series(largest_remainder(need.values, rest), index=need.index) if need.sum() != rest else need
    rng = np.random.default_rng(SEED)
    strat = []
    for ct, k in need.items():
        sub = pool[pool.claim_type == ct]
        books = sub.source_pdf.value_counts()
        alloc = largest_remainder(books.values, min(int(k), len(sub)))
        for book, m in zip(books.index, alloc):
            cell = sub.uid[sub.source_pdf == book].to_numpy()
            strat += list(rng.choice(cell, size=m, replace=False))

    rows = ([(u, "prashant") for u in found] + [(u, "health") for u in health] + [(u, "stratified") for u in strat])
    out = pd.DataFrame(rows, columns=["uid", "inclusion_reason"]).merge(
        safe[["uid", "claim_type", "source_pdf"]], on="uid")[["uid", "claim_type", "source_pdf", "inclusion_reason"]]
    assert out.uid.is_unique
    out["seed"] = SEED
    out.to_csv(OUT, index=False)

    full = rep.claim_type.value_counts()
    samp = out.claim_type.value_counts()
    nh = lambda s: s.drop("health", errors="ignore")
    t = pd.DataFrame({"full_n": full, "full_%": (full / full.sum() * 100).round(1),
                      "sample_n": samp.reindex(full.index, fill_value=0),
                      "sample_%": (samp.reindex(full.index, fill_value=0) / len(out) * 100).round(1),
                      "full_%_excl_health": (nh(full) / nh(full).sum() * 100).round(1),
                      "sample_%_excl_health": (nh(samp) / nh(samp).sum() * 100).round(1)})
    print(f"\nsample {len(out)} -> {OUT} (seed {SEED}); by reason {out.inclusion_reason.value_counts().to_dict()}")
    print(t.fillna("").to_string())
    bk = out.source_pdf.value_counts()
    print(f"books: {out.source_pdf.nunique()} of {rep.source_pdf.nunique()}; largest book share "
          f"{bk.iloc[0] / len(out):.1%} ({bk.index[0]}) vs {rep.source_pdf.value_counts().iloc[0] / len(rep):.1%} in full set; "
          f"largest excluding the health book: {bk.drop(rep[rep.claim_type == 'health'].source_pdf.unique(), errors='ignore').iloc[0] / len(out):.1%}")


if __name__ == "__main__":
    main()
