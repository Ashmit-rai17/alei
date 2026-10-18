"""Labels vs macro-F1: is the classifier data-limited or broken?

  .venv-ship/bin/python -I scripts/learning_curve.py   ->  data/learning_curve.json (+ table)

Same pool as train.py: every LLM-labelled award except Prashant's 79 (held out). For each size, a
nested random subsample of that pool (seed 20261018: the 90 are inside the 180, which are inside
the 270) trains the same pipeline with the same separator cleaning, and is scored two ways:
  human   — against Prashant's 64 blind labels (the number that matters)
  heldout — on train.py's book-grouped test fold, training only on that fold's train side
Extra seeds give the spread at each size, so a single lucky subsample cannot make the curve.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
import train as T  # noqa: E402

SIZES = [90, 180, 270, None]          # None = the full pool
SEEDS = [20261018, 1, 2, 3, 4, 5]
HUMAN_SHEET = os.path.join(T.DATA, "prashant", "phase2_label_completed.xlsx")
OUT = os.path.join(T.DATA, "learning_curve.json")


def main():
    safe, d = T.load(T.LABELS, T.AWARDS, False)
    held, _ = T.prashant_uids(safe)
    d = d[~d.uid.isin(held)].reset_index(drop=True)
    classes = list(T.ORDER)
    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(d.y)

    human, msg = T.load_human(safe, [HUMAN_SHEET])
    H = safe.set_index("uid").loc[human.uid]
    Xh = H.text.map(T.clean)
    Yh = MultiLabelBinarizer(classes=classes).fit_transform(human.y)

    # train.py's headline fold (non-health, book-grouped), reproduced exactly
    is_h = (d.claim_type == "health").to_numpy()
    nh = np.where(~is_h)[0]
    dn = d.iloc[nh]
    overall = dn.claim_type.value_counts(normalize=True)
    folds = list(T.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=T.SEED).split(
        dn.text, dn.claim_type, groups=dn.source_pdf))
    ftr, fte = min(folds, key=lambda f: (overall - dn.claim_type.iloc[f[1]].value_counts(normalize=True)
                                         .reindex(overall.index, fill_value=0)).abs().sum())
    te = nh[fte]
    tr_pool = np.concatenate([nh[ftr], np.where(is_h)[0]])

    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(len(d))         # nested: prefixes of one permutation
        perm_tr = [i for i in np.random.default_rng(seed).permutation(len(d)) if i in set(tr_pool)]
        for size in SIZES:
            n = len(d) if size is None else size
            idx = perm[:n]
            m = T.pipeline().fit(d.text.iloc[idx], Y[idx])
            Ph = T.predict(m, Xh)
            _, mac_h, mic_h = T.per_label(Yh, Ph, classes)
            k = min(len(perm_tr), round(n * len(tr_pool) / len(d)))
            m2 = T.pipeline().fit(d.text.iloc[perm_tr[:k]], Y[perm_tr[:k]])
            _, mac_o, _ = T.per_label(Y[te], T.predict(m2, d.text.iloc[te]), classes)
            rows.append({"seed": seed, "train_awards": int(n), "human_macro_f1": mac_h, "human_micro_f1": mic_h,
                         "heldout_train_awards": int(k), "heldout_macro_f1": mac_o})
            print(f"seed {seed:>8}  n={n:3}  human macro-F1 {mac_h:.3f} micro {mic_h:.3f}  |  "
                  f"held-out (train {k}) macro-F1 {mac_o:.3f}", flush=True)
    df = pd.DataFrame(rows)
    main_seed = df[df.seed == SEEDS[0]].set_index("train_awards")
    agg = df.groupby("train_awards").agg(human_mean=("human_macro_f1", "mean"), human_min=("human_macro_f1", "min"),
                                         human_max=("human_macro_f1", "max"), heldout_mean=("heldout_macro_f1", "mean"))
    agg["human_seed_20261018"] = main_seed.human_macro_f1
    agg["heldout_seed_20261018"] = main_seed.heldout_macro_f1
    print("\n" + agg.round(3).to_string())
    json.dump({"method": __doc__.strip(), "human_n": len(human), "pool": len(d), "rows": rows,
               "summary": agg.round(4).reset_index().to_dict("records")}, open(OUT, "w"), indent=1, default=float)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
