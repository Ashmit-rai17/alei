"""Task B — multi-label grounds classifier.

  python scripts/train.py                      # real labels: data/labels.parquet
  python scripts/train.py --stub               # plumbing test on keyword stub labels (never reported)

Features  TF-IDF word 1-3 grams + char_wb 3-5 grams, min_df=3, sublinear_tf
Model     OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000))
Split     StratifiedGroupKFold: stratified by claim_type, grouped by source_pdf (one fold = test)

Evaluation, reported separately:
  1a. HEADLINE, book-grouped: held-out Qwen labels on a test fold of books never seen in training.
      All health awards come from one book, so health is kept in training and cannot be scored here.
  1b. SECONDARY, within-book (health only): the health book's awards split 80/20 at random; the
      model trains on the headline train set + 80% of health, scored on the other 20%. Optimistic:
      the same book is on both sides.
  2. Prashant's human-verified rows (data/prashant/phase2_label.xlsx; partly filled sheets are
     scored on the filled rows, n reported), model trained on every labeled award except those
     rows. Qwen-vs-human agreement on the same rows is reported alongside.
Every class is modelled; a class with < 10 training positives is flagged low support, never dropped.
A class with 0 positives gets scikit-learn's constant (never) predictor.

Writes models/alei_clf.joblib (plain dict of scikit-learn objects), data/feature_weights.json,
data/eval_report.json.
"""

import argparse
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from rules import rule_grounds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
SAFE = os.path.join(DATA, "awards_safe.parquet")
LABELS = os.path.join(DATA, "labels.parquet")
AWARDS = os.path.join(DATA, "labeled_awards.parquet")        # every labeled uid, incl. zero-ground awards
STUB_LABELS = os.path.join(DATA, "labels_STUB.parquet")
STUB_AWARDS = os.path.join(DATA, "labeled_awards_STUB.parquet")
HUMAN = os.path.join(DATA, "prashant", "phase2_label.xlsx")
MODEL = os.path.join(ROOT, "models", "alei_clf.joblib")
WEIGHTS = os.path.join(DATA, "feature_weights.json")
REPORT = os.path.join(DATA, "eval_report.json")
SEED = 20261018
THRESHOLD = 0.5
# Award separators ("-----", "*****", "ooooo") became char n-grams among the top contributing terms.
# Stripped before vectorising; the pattern travels inside the .joblib so predict.py applies the same.
SEPARATOR_RUNS = r"([-*=_#oO])\1{2,}"


def clean(t):
    return re.sub(SEPARATOR_RUNS, " ", t or "")
LOW_SUPPORT = 10     # classes with fewer training positives are modelled but flagged "low support"
HEALTH_TEST_FRAC = 0.2

NAMES = {"G1": "non_disclosure_proposal", "G2": "non_disclosure_revival", "G3": "fraud_misrepresentation",
         "G4": "policy_not_in_force", "G5": "documents_not_submitted", "G6": "excluded_by_clause",
         "G7": "pre_existing_disease", "G8": "waiting_period", "G9": "hospitalisation_not_justified",
         "G10": "treatment_not_covered", "G11": "amount_limited", "G12": "cause_outside_cover", "G99": "other"}
ORDER = list(NAMES)


def norm_ws(s):
    return " ".join(str(s).split())


def pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=3, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b", lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 lowercase=True)),
    ])
    # n_jobs=1: joblib 1.3.2 (the s390x pin) leaves loky resource-tracker tracebacks at exit on Python 3.12
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=1000), n_jobs=1)
    return Pipeline([("features", feats), ("clf", clf)])


# ---------------------------------------------------------------- stub (plumbing only)

STUB_RULES = {
    "G1": r"(?i)(suppress|conceal|non[- ]?disclos|withh[eo]ld).{0,200}proposal|proposal.{0,200}(suppress|conceal|non[- ]?disclos)",
    "G2": r"(?i)reviv\w*.{0,200}(suppress|conceal|good health|personal statement)",
    "G3": r"(?i)fraud|deliberate(ly)? (mis|wrong|false)|manipulat|fabricat",
    "G4": r"(?i)(policy|policies) (was|were|had) (in )?lapsed|lapsed condition",
    "G5": r"(?i)(not|non)[- ](submi|furnish|produc)\w* .{0,60}(document|record|fir|post[- ]mortem)",
    "G6": r"(?i)exclusion clause|clause \d+(\.\d+)* .{0,40}exclu",
    "G7": r"(?i)pre[- ]?existing",
    "G8": r"(?i)waiting period",
    "G9": r"(?i)hospitali[sz]ation (was )?not (justified|necessary|required)|could have been (done|treated) .{0,30}(opd|out ?patient)",
    "G10": r"(?i)not (covered|payable) under .{0,40}(day ?care|surgical)|not an? (approved )?day ?care",
    "G11": r"(?i)sub[- ]?limit|room rent|non[- ]?medical (items|expenses)|deduct",
    "G12": r"(?i)suicide|accident(al)? (death )?benefit .{0,80}(not|rejected|denied)|murder",
}


def make_stub():
    df = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "is_repudiation", "text"])
    df = df[df.is_repudiation]
    rows = []
    for a, s, t in zip(df.award_id, df.source_pdf, df.text):
        for g, rx in STUB_RULES.items():
            m = re.search(rx, t)
            if m:
                rows.append({"uid": f"{a}|{s}", "award_id": a, "source_pdf": s, "ground_id": g,
                             "quote": m.group(0), "quote_verified": True, "batch_id": "STUB"})
    pd.DataFrame(rows).to_parquet(STUB_LABELS, index=False)
    pd.DataFrame({"uid": [f"{a}|{s}" for a, s in zip(df.award_id, df.source_pdf)]}).to_parquet(STUB_AWARDS, index=False)
    print(f"STUB labels: {len(rows):,} rows over {df.shape[0]:,} awards -> {STUB_LABELS}")


# ---------------------------------------------------------------- data

def load(labels_path, awards_path, verified_only):
    safe = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "ground_span", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    lab = pd.read_parquet(labels_path)
    lab = lab[lab.ground_id.isin(NAMES)]
    if verified_only:
        lab = lab[lab.quote_verified]
    awards = pd.read_parquet(awards_path).uid
    y = lab.groupby("uid").ground_id.apply(lambda s: sorted(set(s))).reindex(awards).apply(
        lambda v: v if isinstance(v, list) else [])
    d = safe.set_index("uid").loc[awards].reset_index()
    d["y"] = y.values
    d["text"] = d.text.map(clean)
    return safe, d


PRASHANT_SHEETS = [os.path.join(DATA, "prashant", "phase1_read.xlsx"), HUMAN]


def match_rows(rows, safe, safe_norm=None):
    """award_id/award_text rows -> uids (None when unmatched). Join on award_id; when the id is
    missing or ambiguous, match the start of the text; ids can drift with re-segmentation
    (LI/BK-10/05 -> LI/BK-10/05-06), so last, probe 3 text windows."""
    safe_norm = safe_norm or [norm_ws(t) for t in safe.text]
    safe_key = {t[:400]: u for u, t in zip(safe.uid, safe_norm)}
    by_id = safe.groupby("award_id").uid.apply(list).to_dict()
    uids, how = [], {"id": 0, "text": 0, "probe": 0, "unmatched": 0}
    for aid, text in zip(rows.award_id.astype(str), rows.award_text):
        cand = by_id.get(aid, [])
        full = norm_ws(text)
        if len(cand) == 1:
            uids.append(cand[0]); how["id"] += 1
        elif full[:400] in safe_key:
            uids.append(safe_key[full[:400]]); how["text"] += 1
        else:
            probes = [full[o:o + 60] for o in (500, 1500, len(full) // 2) if len(full) > o + 60]
            hits = [u for u, t in zip(safe.uid, safe_norm) if probes and all(p in t for p in probes)]
            if len(hits) == 1:
                uids.append(hits[0]); how["probe"] += 1
            else:
                uids.append(None); how["unmatched"] += 1
    return uids, how


def prashant_uids(safe):
    """Every award on Prashant's sheets, filled or not. These never train, so the human evaluation
    is always out-of-sample whenever his rows arrive."""
    rows = pd.concat([pd.read_excel(p, sheet_name=0)[["award_id", "award_text"]] for p in PRASHANT_SHEETS
                      if os.path.exists(p)], ignore_index=True)
    uids, how = match_rows(rows, safe)
    return {u for u in uids if u}, f"{len(rows)} sheet rows, matched {how}"


def load_human(safe, paths=None):
    """Filled rows of the phase-2 sheet(s) -> (DataFrame uid, y, row source) and a status line.
    Several files (Prashant's batches) are merged; a later file wins for the same award."""
    paths = [p for p in (paths or [HUMAN]) if os.path.exists(p)]
    if not paths:
        return None, "sheet missing"
    frames = []
    for p in paths:
        h = pd.read_excel(p, sheet_name="labeling" if "labeling" in pd.ExcelFile(p).sheet_names else 0).fillna("")
        gcols = [c for c in h.columns if re.fullmatch(r"ground_\d", c)]
        h["y"] = [sorted({m.upper() for c in gcols for m in re.findall(r"(?i)\bG\d{1,2}\b", str(r[c]))
                          if m.upper() in NAMES}) for _, r in h.iterrows()]
        filled = h[h[gcols].astype(str).apply(lambda col: col.str.strip() != "").any(axis=1)].copy()
        filled["file"] = os.path.basename(p)
        frames.append(filled)
    filled = pd.concat(frames, ignore_index=True)
    n_rows = sum(len(f) for f in frames)
    if filled.empty:
        return None, f"{len(paths)} sheet(s), 0 rows filled"
    uids, how = match_rows(filled, safe)
    filled = filled.assign(uid=uids)
    filled = filled[filled.uid.notna()].drop_duplicates("uid", keep="last")
    return filled[["uid", "y", "file"]], f"{n_rows} filled rows in {len(paths)} sheet(s); matched {how}; scored {len(filled)}"


# ---------------------------------------------------------------- metrics

def per_label(Y_true, Y_pred, classes, train_pos=None):
    p, r, f, s = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0, labels=range(len(classes)))
    t = pd.DataFrame({"ground": classes, "name": [NAMES[c] for c in classes], "precision": p, "recall": r,
                      "f1": f, "support": s}).round(3)
    if train_pos is not None:
        t["train_pos"] = [int(train_pos.get(c, 0)) for c in classes]
        t["flag"] = ["LOW SUPPORT" if k < LOW_SUPPORT else "" for k in t.train_pos]
    present = t.support > 0
    macro = float(t.loc[present, "f1"].mean()) if present.any() else float("nan")
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    return t, macro, micro


def g1_g7(Y_true, Y_pred, classes):
    if "G1" not in classes or "G7" not in classes:
        return {}
    i1, i7 = classes.index("G1"), classes.index("G7")
    out = {}
    for name, mask in [("true G1 only", (Y_true[:, i1] == 1) & (Y_true[:, i7] == 0)),
                       ("true G7 only", (Y_true[:, i7] == 1) & (Y_true[:, i1] == 0)),
                       ("true both", (Y_true[:, i1] == 1) & (Y_true[:, i7] == 1))]:
        p1, p7 = Y_pred[mask, i1], Y_pred[mask, i7]
        out[name] = {"n": int(mask.sum()), "pred G1 only": int(((p1 == 1) & (p7 == 0)).sum()),
                     "pred G7 only": int(((p7 == 1) & (p1 == 0)).sum()),
                     "pred both": int(((p1 == 1) & (p7 == 1)).sum()),
                     "pred neither": int(((p1 == 0) & (p7 == 0)).sum())}
    return out


def show(title, table, macro, micro, conf):
    print(f"\n== {title}: macro-F1 {macro:.3f} (over labels with test support > 0), micro-F1 {micro:.3f}")
    print(table.sort_values(["support", "f1"], ascending=[False, True]).to_string(index=False))
    zero = table[table.support == 0].ground.tolist()
    if zero:
        print("no test support (not scored, not in macro):", zero)
    print("worst 3:", ", ".join(f"{r.ground} F1 {r.f1:.2f} (n={r.support})"
                                for r in table[table.support > 0].nsmallest(3, "f1").itertuples()))
    print("G1 vs G7:", json.dumps(conf))


def baseline(rows, classes, field):
    """Keyword baseline (scripts/rules.py) as a classifier: on ground_span (empty -> nothing) or full text."""
    return MultiLabelBinarizer(classes=classes).fit_transform(
        [[g for g in rule_grounds(t if isinstance(t, str) else "") if g in classes] for t in rows[field]])


def versus(title, Y_true, P_model, rows, classes, train_pos=None):
    """Classifier vs the two keyword baselines on the same rows; per-label F1 side by side."""
    tm, mm, um = per_label(Y_true, P_model, classes)
    tb, mb, ub = per_label(Y_true, baseline(rows, classes, "ground_span"), classes)
    tf, mf, uf = per_label(Y_true, baseline(rows, classes, "text"), classes)
    t = pd.DataFrame({"ground": classes, "support": tm.support, "model_f1": tm.f1, "kw_span_f1": tb.f1,
                      "kw_text_f1": tf.f1})
    def win(r):
        if r.support == 0:
            return "-"
        sc = {"model": r.model_f1, "kw_span": r.kw_span_f1, "kw_text": r.kw_text_f1}
        best = [k for k, v in sc.items() if v == max(sc.values())]
        return best[0] if len(best) == 1 else "tie(" + "=".join(best) + ")"
    t["winner"] = [win(r) for r in t.itertuples()]
    print(f"\n-- {title}: classifier vs keyword baseline (same rows)")
    print(f"   macro-F1  model {mm:.3f} | keywords on ground_span {mb:.3f} | keywords on full text {mf:.3f}")
    print(f"   micro-F1  model {um:.3f} | keywords on ground_span {ub:.3f} | keywords on full text {uf:.3f}")
    print(t[t.support > 0].to_string(index=False))
    beaten = t[(t.support > 0) & t.winner.str.startswith("kw")].ground.tolist()
    print("   baseline beats the model on:", beaten or "none")
    return {"model": {"macro_f1": mm, "micro_f1": um}, "kw_ground_span": {"macro_f1": mb, "micro_f1": ub},
            "kw_full_text": {"macro_f1": mf, "micro_f1": uf}, "per_label": t.to_dict("records"),
            "baseline_wins": beaten}


def predict(model, X):
    return (model.predict_proba(X) >= THRESHOLD).astype(int)


# ---------------------------------------------------------------- main

def main(stub, verified_only):
    if stub:
        make_stub()
    labels, awards = (STUB_LABELS, STUB_AWARDS) if stub else (LABELS, AWARDS)
    safe, d = load(labels, awards, verified_only)
    human, human_msg = load_human(safe)
    print(f"{'STUB ' if stub else ''}labeled awards {len(d):,}; human set: {human_msg}")
    held, held_msg = prashant_uids(safe)
    hum_uids = held | (set(human.uid) if human is not None else set())
    n0 = len(d)
    d = d[~d.uid.isin(hum_uids)].reset_index(drop=True)          # Prashant's awards never train
    print(f"held out for human evaluation: {n0 - len(d)} labeled awards ({held_msg})")

    counts = pd.Series([g for v in d.y for g in v]).value_counts()
    classes = list(ORDER)                                   # every class modelled, none dropped
    print("training positives:", {c: int(counts.get(c, 0)) for c in ORDER},
          "| low support (<%d):" % LOW_SUPPORT, [c for c in ORDER if counts.get(c, 0) < LOW_SUPPORT])
    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(d.y)
    is_h = (d.claim_type == "health").to_numpy()

    # 1a. headline: book-grouped split over the non-health awards; health stays in training
    nh = np.where(~is_h)[0]
    dn = d.iloc[nh]
    overall = dn.claim_type.value_counts(normalize=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(dn.text, dn.claim_type, groups=dn.source_pdf))
    ftr, fte = min(folds, key=lambda f: (overall - dn.claim_type.iloc[f[1]].value_counts(normalize=True)
                                         .reindex(overall.index, fill_value=0)).abs().sum())
    te = nh[fte]
    tr = np.concatenate([nh[ftr], np.where(is_h)[0]])
    assert not set(d.source_pdf.iloc[tr]) & set(d.source_pdf.iloc[te])
    print(f"headline split: train {len(tr):,} awards / {d.source_pdf.iloc[tr].nunique()} books "
          f"(incl. all {is_h.sum()} health), test {len(te):,} / {d.source_pdf.iloc[te].nunique()} books; "
          f"test claim_type {d.claim_type.iloc[te].value_counts().to_dict()}")
    tr_pos = pd.Series(Y[tr].sum(0), index=classes)
    m = pipeline().fit(d.text.iloc[tr], Y[tr])
    P = predict(m, d.text.iloc[te])
    t_ho, mac_ho, mic_ho = per_label(Y[te], P, classes, tr_pos)
    conf_ho = g1_g7(Y[te], P, classes)
    base_ho = versus("1a headline", Y[te], P, d.iloc[te], classes)
    show("1a. HEADLINE held-out Qwen labels, book-grouped (no health: one book only)" + (" [STUB]" if stub else ""),
         t_ho, mac_ho, mic_ho, conf_ho)

    # 1b. secondary: within-book health
    h_idx = np.where(is_h)[0]
    rng = np.random.default_rng(SEED)
    h_te = rng.choice(h_idx, size=max(1, int(round(len(h_idx) * HEALTH_TEST_FRAC))), replace=False) if len(h_idx) else h_idx
    sec = {"status": "no health awards labeled"}
    if len(h_te):
        tr2 = np.setdiff1d(np.concatenate([nh[ftr], h_idx]), h_te)
        m2 = pipeline().fit(d.text.iloc[tr2], Y[tr2])
        P2 = predict(m2, d.text.iloc[h_te])
        t_h2, mac_h2, mic_h2 = per_label(Y[h_te], P2, classes, pd.Series(Y[tr2].sum(0), index=classes))
        conf_h2 = g1_g7(Y[h_te], P2, classes)
        base_h2 = versus("1b within-book health", Y[h_te], P2, d.iloc[h_te], classes)
        show(f"1b. SECONDARY within-book health (OPTIMISTIC: same book in train and test), n={len(h_te)}"
             + (" [STUB]" if stub else ""), t_h2, mac_h2, mic_h2, conf_h2)
        sec = {"baseline": base_h2, "macro_f1": mac_h2, "micro_f1": mic_h2, "per_label": t_h2.to_dict("records"), "g1_vs_g7": conf_h2,
               "n_test": int(len(h_te)), "note": "within-book: same source_pdf on both sides; optimistic"}

    final = pipeline().fit(d.text, Y)
    report = {"stub": stub, "verified_only": verified_only, "n_train_awards": len(d), "classes": classes,
              "low_support": [c for c in ORDER if counts.get(c, 0) < LOW_SUPPORT], "threshold": THRESHOLD,
              "heldout_book_grouped": {"baseline": base_ho, "macro_f1": mac_ho, "micro_f1": mic_ho, "per_label": t_ho.to_dict("records"),
                                       "g1_vs_g7": conf_ho, "n_test": len(te)},
              "health_within_book": sec, "human": {"status": human_msg}}
    if human is not None and len(human):
        H = safe.set_index("uid").loc[human.uid]
        H = H.assign(text=H.text.map(clean))
        Yh = MultiLabelBinarizer(classes=classes).fit_transform(human.y)
        Ph = predict(final, H.text)
        t_h, mac_h, mic_h = per_label(Yh, Ph, classes, pd.Series(Y.sum(0), index=classes))
        conf_h = g1_g7(Yh, Ph, classes)
        report["human_baseline"] = versus("2 human set", Yh, Ph, H.reset_index(), classes)
        show(f"2. HUMAN-verified (Prashant), classifier vs human, n={len(human)} filled rows", t_h, mac_h, mic_h, conf_h)
        if mac_h < 0.5:
            print(f"!! FLAG: human-set macro-F1 {mac_h:.3f} < 0.5 (n={len(human)})")
        report["human"].update({"macro_f1": mac_h, "micro_f1": mic_h, "per_label": t_h.to_dict("records"),
                                "g1_vs_g7": conf_h, "n": len(human)})
        # Qwen vs human on the same rows (rows Qwen labeled)
        lab_all = pd.read_parquet(labels)
        lab_all = lab_all[lab_all.ground_id.isin(NAMES)]
        qy = lab_all.groupby("uid").ground_id.apply(lambda s: sorted(set(s))).to_dict()
        labeled = set(pd.read_parquet(awards).uid)
        hq = human[human.uid.isin(labeled)]
        if len(hq):
            Yq = MultiLabelBinarizer(classes=classes).fit_transform([qy.get(u, []) for u in hq.uid])
            Yhq = MultiLabelBinarizer(classes=classes).fit_transform(hq.y)
            t_q, mac_q, mic_q = per_label(Yhq, Yq, classes)
            exact = float(np.mean([set(qy.get(u, [])) == set(y) for u, y in zip(hq.uid, hq.y)]))
            show(f"2b. QWEN vs human, n={len(hq)} rows (exact set match {exact:.1%})", t_q, mac_q, mic_q,
                 g1_g7(Yhq, Yq, classes))
            report["qwen_vs_human"] = {"macro_f1": mac_q, "micro_f1": mic_q, "exact_match": exact,
                                       "per_label": t_q.to_dict("records"), "n": len(hq)}
    else:
        print(f"\n== 2. HUMAN-verified: not computed ({human_msg})")

    # explainability: top 15 n-grams per class
    names = final.named_steps["features"].get_feature_names_out()
    est = final.named_steps["clf"].estimators_
    weights = {}
    for c, e in zip(classes, est):
        if not hasattr(e, "coef_"):                      # constant predictor: class had no positives
            weights[c] = {"name": NAMES[c], "top": [], "note": "no training positives"}
            continue
        coef = e.coef_.ravel()
        top = np.argsort(coef)[::-1][:15]
        weights[c] = {"name": NAMES[c], "top": [[names[i].split("__", 1)[1], round(float(coef[i]), 4),
                                                 names[i].split("__", 1)[0]] for i in top]}
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    suffix = "_STUB" if stub else ""
    model_path = MODEL.replace(".joblib", f"{suffix}.joblib")
    joblib.dump({"pipeline": final, "classes": classes, "names": {c: NAMES[c] for c in classes},
                 "threshold": THRESHOLD, "stub": stub, "n_train": len(d),
                 "train_uids": sorted(d.uid), "held_out_uids": sorted(hum_uids),
                 "preprocess_regex": SEPARATOR_RUNS, "preprocess_replace": " "}, model_path, compress=3)
    json.dump(weights, open(WEIGHTS.replace(".json", f"{suffix}.json"), "w"), indent=1)
    json.dump(report, open(REPORT.replace(".json", f"{suffix}.json"), "w"), indent=1, default=float)
    print(f"\nmodel -> {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB); weights, report written")
    for c in classes:
        print(f"  {c}: {[w[0] for w in weights[c]['top'][:8]] or weights[c].get('note')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--verified-only", action="store_true", help="train only on quote_verified labels")
    a = ap.parse_args()
    main(a.stub, a.verified_only)
