"""Score Prashant's human labels against the SAVED model — no retraining.

  python scripts/human_eval.py                                  # data/prashant/phase2_label.xlsx
  python scripts/human_eval.py batch1.xlsx batch2.xlsx ...      # his batches; a later file wins per award
  python scripts/human_eval.py data/prashant/phase2_label_completed.xlsx data/prashant/phase1_read_completed.xlsx
                                                                # phase 1 + 2 merged (n=79)
Phase 1 sheets need the category ids in ground_1..ground_5 columns, like phase 2.
  python scripts/human_eval.py --model data/demo_package/alei_clf.joblib

Scores whatever rows are filled (ground_1..ground_5 holding G-ids) and reports n. Three comparisons
on the same rows:
  1. classifier (saved .joblib) vs human           — the headline number
  2. Qwen labels (data/labels.parquet) vs human     — how good the training labels are
  3. keyword rules (scripts/rules.py) vs human      — the baseline the classifier must beat
Refuses to score an award the model trained on (train.py holds all of Prashant's awards out).
Writes data/human_eval.json.
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from train import (DATA, HUMAN, LABELS, MODEL, NAMES, SAFE, g1_g7, load_human, per_label,  # noqa: E402
                   versus)
import re  # noqa: E402

OUT = os.path.join(DATA, "human_eval.json")


def main(paths, model_path):
    safe = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "ground_span", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    human, msg = load_human(safe, paths)
    print(f"human set: {msg}")
    if human is None or human.empty:
        return
    bundle = joblib.load(model_path)
    classes = bundle["classes"]
    leak = set(human.uid) & set(bundle.get("train_uids", []))
    if leak:
        sys.exit(f"!! {len(leak)} human-labelled awards were in this model's training set — retrain with "
                 f"train.py (it holds Prashant's awards out) before scoring. e.g. {sorted(leak)[:3]}")
    if "train_uids" not in bundle:
        print("!! model predates the hold-out bookkeeping; overlap with training cannot be checked")
    else:
        held = set(bundle.get("held_out_uids", []))
        print(f"training exclusion: {len(set(human.uid) & set(bundle['train_uids']))} of {len(human)} human-labelled "
              f"awards are in the model's training set ({bundle['n_train']} training awards); "
              f"{len(set(human.uid) & held)} of {len(human)} are on the model's held-out list")
    H = safe.set_index("uid").loc[human.uid].reset_index()
    mlb = MultiLabelBinarizer(classes=classes)
    Yh = mlb.fit_transform([[g for g in y if g in classes] for y in human.y])
    n = len(human)
    report = {"n": n, "status": msg, "model": os.path.abspath(model_path), "per_file": human.file.value_counts().to_dict()}

    rx = bundle.get("preprocess_regex")                    # same cleaning the model was trained with
    X = H.text.map(lambda t: re.sub(rx, bundle.get("preprocess_replace", " "), t)) if rx else H.text
    print(f"preprocessing: {'strip ' + rx if rx else 'none (model predates it)'}")
    P = (bundle["pipeline"].predict_proba(X) >= bundle["threshold"]).astype(int)
    t, mac, mic = per_label(Yh, P, classes)
    print(f"\n== 1. CLASSIFIER vs human, n={n}: macro-F1 {mac:.3f} (labels with support > 0), micro-F1 {mic:.3f}")
    print(t[t.support > 0].sort_values("support", ascending=False).to_string(index=False))
    print("G1 vs G7:", json.dumps(g1_g7(Yh, P, classes)))
    if mac < 0.5:
        print(f"!! FLAG: classifier macro-F1 on the human set is {mac:.3f} < 0.5 (n={n})")
    report["classifier"] = {"macro_f1": mac, "micro_f1": mic, "per_label": t.to_dict("records"),
                            "g1_vs_g7": g1_g7(Yh, P, classes)}

    if os.path.exists(LABELS):
        lab = pd.read_parquet(LABELS)
        q = lab[lab.ground_id.isin(NAMES)].groupby("uid").ground_id.apply(lambda s: sorted(set(s))).to_dict()
        labeled = set(pd.read_parquet(os.path.join(DATA, "labeled_awards.parquet")).uid)
        m = human.uid.isin(labeled).to_numpy()
        if m.any():
            Yq = mlb.transform([[g for g in q.get(u, []) if g in classes] for u in human.uid[m]])
            tq, macq, micq = per_label(Yh[m], Yq, classes)
            exact = float((Yq == Yh[m]).all(axis=1).mean())
            print(f"\n== 2. QWEN labels vs human, n={int(m.sum())}: macro-F1 {macq:.3f}, micro-F1 {micq:.3f}, "
                  f"exact ground-set match {exact:.1%}")
            print(tq[tq.support > 0].sort_values("support", ascending=False).to_string(index=False))
            report["qwen"] = {"n": int(m.sum()), "macro_f1": macq, "micro_f1": micq, "exact_match": exact,
                              "per_label": tq.to_dict("records")}

    report["keyword_baseline"] = versus(f"3. classifier vs keyword baseline on the human set, n={n}", Yh, P, H,
                                        classes)
    json.dump(report, open(OUT, "w"), indent=1, default=float)
    by_n = OUT.replace(".json", f"_n{n}.json")             # kept per n, so n=64 stays beside n=79
    json.dump(report, open(by_n, "w"), indent=1, default=float)
    print(f"\n-> {OUT} and {by_n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="*", default=[HUMAN])
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    main(a.sheets, a.model)
