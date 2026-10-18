"""Read award text on stdin; print predicted grounds and the n-grams that pushed each one up.

  cat award.txt | python predict.py [--model alei_clf.joblib] [--top 8] [--all]
  python predict.py < sample_awards.txt        # several awards, each starting "===== AWARD … ====="

The model defaults to alei_clf.joblib next to this script, else <repo>/models/alei_clf.joblib.

Needs only scikit-learn (and its joblib / numpy dependencies).
"""

import argparse
import os
import re
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = next((p for p in (os.path.join(HERE, "alei_clf.joblib"),
                                  os.path.join(os.path.dirname(os.path.dirname(HERE)), "models", "alei_clf.joblib"))
                      if os.path.exists(p)), os.path.join(HERE, "alei_clf.joblib"))
SEP = re.compile(r"(?m)^=+ *(AWARD[^\n=]*?) *=+ *$")


def explain(bundle, text, top=8, show_all=False):
    pipe = bundle["pipeline"]
    feats, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    if bundle.get("preprocess_regex"):                     # same cleaning the model was trained with
        text = re.sub(bundle["preprocess_regex"], bundle.get("preprocess_replace", " "), text)
    x = feats.transform([text])
    names = feats.get_feature_names_out()
    probs = clf.predict_proba(x)[0]
    out = []
    for c, p, est in zip(bundle["classes"], probs, clf.estimators_):
        if p < bundle["threshold"] and not show_all:
            continue
        if not hasattr(est, "coef_"):                      # class had no training positives
            out.append((c, bundle["names"][c], float(p), []))
            continue
        contrib = x.multiply(est.coef_.ravel()).tocsr()       # tf-idf value × weight, per present n-gram
        idx, vals = contrib.indices, contrib.data
        order = np.argsort(vals)[::-1][:top]
        out.append((c, bundle["names"][c], float(p),
                    [(names[idx[i]].split("__", 1)[1], float(vals[i])) for i in order if vals[i] > 0]))
    return sorted(out, key=lambda r: -r[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="show every class, not just predicted ones")
    a = ap.parse_args()
    text = sys.stdin.read()
    if not text.strip():
        sys.exit("no text on stdin")
    if not os.path.exists(a.model):
        sys.exit(f"model not found: {a.model}")
    bundle = joblib.load(a.model)
    if bundle.get("stub"):
        print("!! STUB MODEL — trained on keyword stub labels, plumbing test only\n")
    parts = SEP.split(text)
    awards = [("input", text)] if len(parts) == 1 else list(zip(parts[1::2], parts[2::2]))
    for title, body in awards:
        if len(awards) > 1:
            print(f"\n=== {title.strip()} ({len(body):,} chars) ===")
        rows = explain(bundle, body, a.top, a.all)
        if not rows:
            print(f"no ground above threshold {bundle['threshold']}")
        for c, name, p, grams in rows:
            mark = "*" if p >= bundle["threshold"] else " "
            print(f"{mark} {c:<4} {name:<30} p={p:.2f}")
            for g, v in grams:
                print(f"      {v:+.3f}  {g!r}")


if __name__ == "__main__":
    main()
