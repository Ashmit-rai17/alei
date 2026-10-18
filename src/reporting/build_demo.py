"""Build the s390x demo package and zip it.

  python scripts/build_demo.py            # interim: data/alei_demo_v1/ + data/alei_demo_v1.zip
  python scripts/build_demo.py --final    # final:   data/alei_demo_final/ + data/alei_demo_final.zip

Run in .venv (needs pandas). The model is copied as-is from models/alei_clf.joblib, which must have
been trained in .venv-ship (scikit-learn 1.4.1.post1, the s390x pin); this script refuses otherwise.
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from display_mask import mask_policy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
TPL = os.path.join(ROOT, "demo")
DEMO_UIDS = [  # held out of training (Prashant's set), LLM-labelled, < 3,000 chars, different grounds
    "LI/BK-10/05-06|0056_DeathClaim-Book6.pdf",                      # G1 + G7
    "IO/KCH/LI/21-001-675/2011-12|0049_DeathClaim-Book18.pdf",       # G2
    "21.08.2234.|0046_DeathClaim-Book14.pdf",                        # G12
]
NAMES = {"G1": "non_disclosure_proposal", "G2": "non_disclosure_revival", "G3": "fraud_misrepresentation",
         "G4": "policy_not_in_force", "G5": "documents_not_submitted", "G6": "excluded_by_clause",
         "G7": "pre_existing_disease", "G8": "waiting_period", "G9": "hospitalisation_not_justified",
         "G10": "treatment_not_covered", "G11": "amount_limited", "G12": "cause_outside_cover", "G99": "other"}

PREDICT_RULES_HEAD = '''"""ALEI keyword baseline — standalone. Pure Python 3 standard library: no scikit-learn, no numpy.

  python3 predict_rules.py < award.txt
  python3 predict_rules.py < sample_awards.txt     # several awards, each starting "===== AWARD … ====="

Prints each matched ground and the phrase that triggered it. Same rules as the project's
scripts/rules.py (copied verbatim below); taxonomy v1.0, 12 grounds + G99.
"""
'''

PREDICT_RULES_TAIL = '''

NAMES = {names}
_SEP = re.compile(r"(?m)^=+ *(AWARD[^\\n=]*?) *=+ *$")


def _main():
    import sys
    text = sys.stdin.read()
    if not text.strip():
        sys.exit("no text on stdin")
    parts = _SEP.split(text)
    awards = [("input", text)] if len(parts) == 1 else list(zip(parts[1::2], parts[2::2]))
    for title, body in awards:
        if len(awards) > 1:
            print(f"\\n=== {{title.strip()}} ({{len(body):,}} chars) ===")
        hits = rule_explain(body)
        if not hits:
            print("no rule fired")
        for g, phrase in hits:
            print(f"* {{g:<4}} {{NAMES[g]:<30}} \\"{{phrase}}\\"")


if __name__ == "__main__":
    _main()
'''


def readme(final, n_train, built, ids):
    status = ("FINAL MODEL" if final else
              "INTERIM MODEL — alei_clf.joblib and explorer.html are placeholders.\n"
              "A final version (more labelled awards) arrives ~20:30 IST. Only those two files change;\n"
              "the scripts, sample awards and the way you run it stay the same.")
    return f"""ALEI demo package  ({'final' if final else 'interim'}, built {built})
{'=' * 78}
{status}
{'=' * 78}

WHAT THIS IS
Grounds on which Indian insurers repudiate claims, read from Insurance Ombudsman awards.
Two predictors on 3 redacted awards, in this order:
  1. keyword rules  (predict_rules.py) — deterministic extractor, pure Python, no dependencies
  2. classifier     (alei_clf.joblib)  — TF-IDF + one-vs-rest logistic regression,
                                         trained on {n_train} LLM-labelled awards
Each is shown with its measured macro-F1 against awards labelled blind by a human (METRICS.txt).
The 3 sample awards were NOT used to train the classifier.

RUN
  ./run_demo.sh                         # uses python3 on PATH
  PYTHON=/usr/bin/python3.12 ./run_demo.sh
Expected: one screen; header (uname -m, OS, Python, CPU/RAM, model load time), then the rules on
all three awards, then the classifier on the same three, then total runtime. No warnings or tracebacks. If anything else
appears, send the full output.

  python3 predict_rules.py < sample_awards.txt   # rules only; works with no packages at all
  python3 predict.py < sample_awards.txt         # classifier only, with contributing n-grams

REQUIREMENTS (classifier only; predict_rules.py needs nothing)
  python 3.12, numpy 1.26.4, scipy 1.11.4, scikit-learn 1.4.1.post1, joblib 1.3.2
  (requirements-s390x.txt). The model was saved under exactly these versions.

FILES
  run_demo.sh, demo.py        the screenshot demo
  predict_rules.py            keyword baseline, standalone
  predict.py                  classifier CLI
  alei_clf.joblib             classifier ({'final' if final else 'INTERIM'})
  sample_awards.txt           3 redacted awards: {', '.join(ids)}
  explorer.html               evidence explorer; open in any browser, works offline ({'final' if final else 'INTERIM'})
  requirements-s390x.txt, VERSIONS.txt, MODEL_STATUS.txt

PRIVACY
Text is redacted ([NAME], [POLICY_NO], [HOSPITAL] …) to a measured residual rate — identifiers
remained in 22 of 64 audited awards; policy numbers are additionally masked here. Do not paste it into any AI tool or
share the files outside the team.

SEND BACK
A screenshot of ./run_demo.sh, plus the full text output.
"""


V1_PREMASK_SHA = "b1e7a414c85b6afe"   # first v1 build (17:35 IST), before the policy-number mask
WHY = {
    "alei_clf.joblib": "retrained: award separators (----, ****, ====, oooo, ####, ____) stripped before "
                       "vectorising so they no longer show as contributing terms; more training awards",
    "explorer.html": "policy numbers below the 'Policy No:' label masked at display time; latest labels",
    "demo.py": "rules first (deterministic extractor), classifier second, each with its measured macro-F1 "
               "against human labels; applies the model's own text preprocessing",
    "predict.py": "applies the text preprocessing stored in the model (separator stripping)",
    "METRICS.txt": "scores printed by the demo, from the human evaluation",
    "README.txt": "describes the new order",
    "MODEL_STATUS.txt": "model status line",
    "VERSIONS.txt": "version note",
    "sample_awards.txt": "policy numbers masked",
}


def metrics_text():
    """Scores the demo prints, from the latest human evaluation (scripts/human_eval.py)."""
    p = os.path.join(DATA, "human_eval.json")
    if not os.path.exists(p):
        return "rules=\nclassifier=\nn=?\n"
    h = json.load(open(p))
    kw = h["keyword_baseline"]["kw_full_text"]["macro_f1"]
    return (f"rules=macro-F1 {kw:.3f} vs human\nclassifier=macro-F1 {h['classifier']['macro_f1']:.3f} vs human\n"
            f"n={h['n']}\n")


def changes_text(name, out, base_zip):
    import hashlib
    old = {}
    if os.path.exists(base_zip):
        with zipfile.ZipFile(base_zip) as z:
            old = {i.filename.split("/", 1)[1]: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist()}
    lines = [f"{name}: differences from alei_demo_v1.zip — overwrite every file marked CHANGED or NEW", ""]
    for f in sorted(os.listdir(out)):
        if f == "CHANGES.txt":
            continue
        h = hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()
        st = "NEW" if f not in old else ("same" if old[f] == h else "CHANGED")
        lines.append(f"  {st:8} {f:24} {WHY.get(f, '') if st != 'same' else ''}".rstrip())
    lines += ["", f"If your alei_demo_v1.zip has sha256 starting {V1_PREMASK_SHA}, its explorer.html and sample_awards.txt are the",
              "PRE-mask versions (policy numbers visible): replace both with the v2 copies and delete the old ones.",
              "Check with:  sha256sum alei_demo_v1.zip"]
    return "\n".join(lines) + "\n"


def main(final, name=None):
    name = name or ("alei_demo_final" if final else "alei_demo_v1")
    out = os.path.join(DATA, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    built = time.strftime("%Y-%m-%d %H:%M")

    model = os.path.join(ROOT, "models", "alei_clf.joblib")
    import warnings
    import joblib as jl
    import sklearn
    # the saving version is only visible through scikit-learn's mismatch warning on load
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = jl.load(model)
    saved = {getattr(w.message, "original_sklearn_version", None) for w in caught} - {None}
    v = saved.pop() if saved else sklearn.__version__
    if v != "1.4.1.post1":
        sys.exit(f"models/alei_clf.joblib was saved by scikit-learn {v}; retrain in .venv-ship first")
    leaked = set(DEMO_UIDS) & set(bundle.get("train_uids", []))
    if leaked:
        sys.exit(f"demo awards are in the training set: {leaked}")
    shutil.copy(model, out)

    for f in ["demo.py", "run_demo.sh"]:
        shutil.copy(os.path.join(TPL, f), out)
    os.chmod(os.path.join(out, "run_demo.sh"), 0o755)
    shutil.copy(os.path.join(ROOT, "src", "modelling", "predict.py"), out)
    shutil.copy(os.path.join(ROOT, "requirements-s390x.txt"), out)
    shutil.copy(os.path.join(DATA, "explorer.html"), out)

    rules_src = open(os.path.join(ROOT, "src", "labelling", "rules.py")).read()
    body = rules_src.split('"""', 2)[2] if rules_src.startswith('"""') else rules_src
    open(os.path.join(out, "predict_rules.py"), "w").write(
        PREDICT_RULES_HEAD + body + PREDICT_RULES_TAIL.format(names=json.dumps(NAMES)))

    safe = pd.read_parquet(os.path.join(DATA, "awards_safe.parquet"), columns=["award_id", "source_pdf", "claim_type", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    lab = pd.read_parquet(os.path.join(DATA, "labels.parquet"))
    T = safe.set_index("uid")
    blocks = []
    for i, u in enumerate(DEMO_UIDS, 1):
        r = T.loc[u]
        g = "+".join(sorted(set(lab[lab.uid == u].ground_id), key=lambda x: int(x[1:]))) or "none"
        assert len(r.text) < 3000
        text, _ = mask_policy(r.text)                  # display-time policy-number mask; corpus frozen
        blocks.append(f"===== AWARD {i}: {r.award_id} ({r.claim_type}; LLM label {g}) =====\n{text.strip()}\n")
    open(os.path.join(out, "sample_awards.txt"), "w").write("\n".join(blocks))

    open(os.path.join(out, "MODEL_STATUS.txt"), "w").write(
        ("FINAL" if final else "INTERIM") + f" model, {bundle['n_train']} training awards, built {built}\n")
    open(os.path.join(out, "VERSIONS.txt"), "w").write(
        "alei_clf.joblib saved under: python 3.12, numpy 1.26.4, scipy 1.11.4, scikit-learn 1.4.1.post1, joblib 1.3.2\n")
    open(os.path.join(out, "METRICS.txt"), "w").write(metrics_text())
    open(os.path.join(out, "README.txt"), "w").write(
        readme(final, bundle["n_train"], built, [T.loc[u].award_id for u in DEMO_UIDS]))

    if name != "alei_demo_v1":
        open(os.path.join(out, "CHANGES.txt"), "w").write(changes_text(name, out, os.path.join(DATA, "alei_demo_v1.zip")))
    zpath = os.path.join(DATA, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            zi = zipfile.ZipInfo.from_file(os.path.join(out, f), arcname=f"{name}/{f}")
            zi.compress_type = zipfile.ZIP_DEFLATED
            with open(os.path.join(out, f), "rb") as fh:
                z.writestr(zi, fh.read())
    print(f"{zpath}  ({os.path.getsize(zpath) / 1e6:.1f} MB): {sorted(os.listdir(out))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--name", help="package name, e.g. alei_demo_v2")
    a = ap.parse_args()
    main(a.final, a.name)
