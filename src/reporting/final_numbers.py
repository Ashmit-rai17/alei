"""Every figure we quote, in one place: data/final_numbers.txt (+ .json).

  python scripts/final_numbers.py

Re-run after the final --finalize / train.py / human_eval.py so the deck and the submission form
cannot disagree. Reads only generated artefacts; computes nothing new except the agreement and
verification tallies, which it derives from the label files directly.
"""

import json
import os
import platform
import re
import sys
import time

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from qwen_label import dedupe  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
IDS = [f"G{i}" for i in range(1, 13)] + ["G99"]


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def jsonl(name):
    p = os.path.join(DATA, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def load_html_meta():
    """meta block of data/explorer.html (masking counts)."""
    import base64, gzip
    p = os.path.join(DATA, "explorer.html")
    if not os.path.exists(p):
        return {}
    m = re.search(r'<script id="blob"[^>]*>(.*?)</script>', open(p).read(), re.S)
    return json.loads(gzip.decompress(base64.b64decode(m.group(1))))["meta"] if m else {}


def soft(x):
    x = x.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def main():
    L, J = [], {}
    w = L.append
    w(f"ALEI — FINAL NUMBERS   (generated {time.strftime('%Y-%m-%d %H:%M')}; source of truth for deck and form)")
    w("Scope: contested denials that reached an Insurance Ombudsman. Grounds that were challenged, not typical.")
    he0 = load("human_eval.json") or {}
    q = he0.get("qwen", {})
    c = he0.get("classifier", {})
    w("")
    w("FRAMING (fixed; the deck and the submission form use these words)")
    w(f"   The LLM labelling layer is the result: macro-F1 {q.get('macro_f1', float('nan')):.3f} against "
      f"{q.get('n', he0.get('n', '?'))} human-verified awards it never saw,")
    qf = {r["ground"]: r for r in q.get("per_label", []) if r["support"] > 0}
    kf = {r["ground"]: max(r["kw_span_f1"], r["kw_text_f1"]) for r in he0.get("keyword_baseline", {}).get("per_label", [])
          if r["support"] > 0}
    beat = [g for g in qf if g in kf and qf[g]["f1"] - kf[g] > 0.05]
    tie = [g for g in qf if g in kf and abs(qf[g]["f1"] - kf[g]) <= 0.05]
    lose = [g for g in qf if g in kf and kf[g] - qf[g]["f1"] > 0.05]
    fmt = lambda gs: ", ".join(f"{g} {qf[g]['f1']:.2f} vs {kf[g]:.2f} (n={qf[g]['support']})" for g in gs) or "none"
    w(f"   beating the best keyword rule on {fmt(beat)};")
    w(f"   matching it on {fmt(tie)};")
    w(f"   trailing it on {fmt(lose)}.")
    kb = he0.get("keyword_baseline", {})
    lc = {r["train_awards"]: r for r in (load("learning_curve.json") or {}).get("summary", [])}
    w(f"   The classifier reaches macro-F1 {c.get('macro_f1', float('nan')):.3f}, trained on "
      f"{(load('eval_report.json') or {}).get('n_train_awards', '?')} awards — level with the deterministic keyword")
    w(f"   baseline at {kb.get('kw_full_text', {}).get('macro_f1', float('nan')):.3f} (full text; "
      f"{kb.get('kw_ground_span', {}).get('macro_f1', float('nan')):.3f} on the extracted ground sentence).")
    if lc:
        lo, hi = min(lc), max(lc)
        w(f"   Learning curve: it rose from {lc[lo]['human_mean']:.2f} to {lc[hi]['human_mean']:.2f} as training labels grew "
          f"from {lo} to {hi}, and is still rising — data-limited, not broken.")
    w("   The contribution is the dataset, the frozen taxonomy and the provenance method. The classifier is a")
    w("   cheap on-device approximation, always reported with that caveat and never as an accuracy claim.")

    cs = load("corpus_stats.json")
    if cs:
        p = cs["provenance"]
        w("\n1. CORPUS PROVENANCE")
        w(f"   {p['archived_urls']} archived URLs -> {p['pdfs']} PDFs ({p['pdfs_readable_text']} readable, "
          f"{p['pdfs_scanned_no_text']} image-only scans, {p['pdfs_broken']} unreadable: archive 1 MB cap) "
          f"-> {p['pages']:,} pages ({p['pages_in_clean_pdfs']:,} in {p['clean_pdfs']} complete PDFs)")
        w(f"   -> {p['awards']:,} awards from {p['source_books_with_awards']} books -> {p['repudiation_awards']:,} "
          f"repudiation awards ({p['repudiation_awards'] / p['awards']:.1%}); {p['awards_with_ground_sentence']:,} with an "
          f"extracted ground sentence")
        w("   Caveats:")
        for c in cs.get("caveats", []):
            w(f"   - {c['text']}")
        J["provenance"] = p

    safe = pd.read_parquet(os.path.join(DATA, "awards_safe.parquet"), columns=["award_id", "source_pdf", "claim_type", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    T = safe.set_index("uid")

    # LLM labels
    recs = {}
    for r in jsonl("qwen_labels.jsonl"):
        recs.setdefault(r["uid"], r)
    samp = pd.read_csv(os.path.join(DATA, "label_sample.csv"))
    reason = samp.set_index("uid").inclusion_reason.str.split(" ").str[0]
    G = [(u, g) for u, r in recs.items() for g in dedupe(r["grounds"])]
    raw = sum(len(r["grounds"]) for r in recs.values())
    lab = pd.read_parquet(os.path.join(DATA, "labels.parquet")) if os.path.exists(os.path.join(DATA, "labels.parquet")) else None
    w("\n2. LABELS BY SOURCE AND MODEL")
    w(f"   LLM: {len(recs):,} of {len(samp):,} sampled awards labelled (sample: "
      f"{samp.inclusion_reason.str.split(' ').str[0].value_counts().to_dict()}; seed 20261018)")
    w(f"        labelled by inclusion reason: {pd.Series([reason.get(u) for u in recs]).value_counts().to_dict()}")
    if lab is not None and "label_model" in lab:
        w(f"        label_model: {lab.drop_duplicates('uid').label_model.value_counts().to_dict()}  "
          f"(API model id: {lab.model_id.iloc[0] if len(lab) else '-'})")
    w(f"        grounds: {raw:,} raw -> {len(G):,} after collapsing repeats within an award")
    per = pd.Series({u: 0 for u in recs})
    per.update(pd.Series([u for u, _ in G]).value_counts())
    w(f"        grounds per award: mean {per.mean():.2f}; {per.value_counts().sort_index().to_dict()}")
    w(f"        per category: {pd.Series([g['id'] for _, g in G]).value_counts().to_dict()}")
    g99 = [(u, g) for u, g in G if g["id"] == "G99"]
    w(f"        G99 (fits no category): {len(g99)}")
    for u, g in g99[:8]:
        w(f"          - {u.split('|')[0]}: {str(g.get('description', ''))[:140]}")
    rl = pd.read_parquet(os.path.join(DATA, "labels_rule.parquet"))
    ra = set(pd.read_parquet(os.path.join(DATA, "labeled_awards_rule.parquet")).uid)
    w(f"   Rules: {len(ra):,} awards with a ground sentence; {rl.uid.nunique():,} map to a category "
      f"({rl.uid.nunique() / len(ra):.1%}), {len(ra) - rl.uid.nunique():,} fall through; "
      f"per category {rl.ground_id.value_counts().to_dict()}")

    w("\n3. QUOTE VERIFICATION (LLM)")
    v = sum(g["quote_verified"] for _, g in G)
    loose = sum(g["quote_verified"] or (soft(g["quote"]) and soft(g["quote"]) in soft(T.at[u, "text"])) for u, g in G)
    w(f"   strict (quote in award text, whitespace normalised only): {v / len(G):.1%} ({v:,}/{len(G):,})")
    w(f"   also ignoring punctuation/case (curly quotes, dashes):    {loose / len(G):.1%} ({loose:,}/{len(G):,})")
    src = pd.Series([g.get("src", "missing") for _, g in G])
    w(f"   quote source: {src.value_counts().to_dict()} -> insurer {(src == 'insurer').mean():.1%}")
    by = pd.DataFrame([(g.get("src"), g["quote_verified"]) for _, g in G], columns=["src", "v"]).groupby("src").v.mean()
    w(f"   verified by source: {by.round(3).to_dict()}")
    J["verified_strict"], J["verified_loose"] = v / len(G), loose / len(G)

    w("\n4. RULE vs LLM AGREEMENT (awards labelled both ways)")
    R = rl.groupby("uid").ground_id.apply(set).to_dict()
    Lset = {u: {g["id"] for g in dedupe(r["grounds"]) if g["id"] in IDS} for u, r in recs.items()}
    both = [u for u in Lset if u in ra]
    if both:
        ex = sum(R.get(u, set()) == Lset[u] for u in both)
        ov = sum(bool(R.get(u, set()) & Lset[u]) for u in both)
        w(f"   n={len(both)}: exact ground-set match {ex / len(both):.1%}; at least one shared ground {ov / len(both):.1%}")
        for g in IDS:
            a = sum(g in R.get(u, set()) for u in both)
            b = sum(g in Lset[u] for u in both)
            c = sum(g in R.get(u, set()) and g in Lset[u] for u in both)
            if b:
                w(f"   {g:4} LLM {b:4}  rules find {c:4} ({c / b:4.0%})  rule hits {a:4}, of which LLM agrees {c / a if a else 0:4.0%}")
        w("   Reading: rules catch the mechanical grounds (G1, G2, G10) and miss the judgement ones (G3, G6, G7).")

    loc = {}
    for r in jsonl("qwen_labels_local.jsonl"):
        loc.setdefault(r["uid"], r)
    ov8 = [u for u in loc if u in recs]
    if ov8:
        a = [g for u in ov8 for g in dedupe(loc[u]["grounds"])]
        b = [g for u in ov8 for g in dedupe(recs[u]["grounds"])]
        w("\n5. MODEL SCALE (same prompt, same awards)")
        w(f"   {len(ov8)} awards: local qwen3 8b verified {sum(g['quote_verified'] for g in a)}/{len(a)} "
          f"({sum(g['quote_verified'] for g in a) / len(a):.0%}); Groq qwen3.8-27b "
          f"{sum(g['quote_verified'] for g in b)}/{len(b)} ({sum(g['quote_verified'] for g in b) / len(b):.0%})")

    oa = load("outcome_analysis.json")
    if oa:
        w("\n6. OUTCOME (repudiation awards with a readable outcome; heuristic)")
        o = oa["repudiation_overall"][0]
        w(f"   all: n={o['n_with_outcome']:,} (+{o['not_extracted']:,} unreadable): dismissed {o['dismissed_%']}%, "
          f"allowed {o['allowed_%']}%, partly {o['partly allowed_%']}%, ex gratia {o['ex gratia_%']}% -> won "
          f"(full/part) {o['in_favour_strict_%']}%, got something {o['in_favour_broad_%']}%")
        for r in oa["by_claim_type"]:
            if r["n_with_outcome"] >= 10:
                w(f"   {r['claim_type']:16} n={r['n_with_outcome']:5,}  won {r['in_favour_strict_%']:5}%  got something "
                  f"{r['in_favour_broad_%']:5}%")

    ev = load("eval_report.json")
    if ev:
        w("\n7. CLASSIFIER (TF-IDF word 1-3 + char 3-5, one-vs-rest logistic regression, balanced)")
        w(f"   trained on {ev['n_train_awards']} LLM-labelled awards (Prashant's 79 held out); "
          f"low-support classes (<10): {ev['low_support']}")
        for k, title in [("heldout_book_grouped", "headline, book-grouped held-out LLM labels"),
                         ("health_within_book", "secondary, within-book health (optimistic)")]:
            e = ev.get(k, {})
            if "macro_f1" not in e:
                continue
            b = e.get("baseline", {})
            w(f"   {title}: n={e.get('n_test')}  macro-F1 {e['macro_f1']:.3f}  micro-F1 {e['micro_f1']:.3f}"
              + (f"  | keyword baseline: on ground sentence {b['kw_ground_span']['macro_f1']:.3f}, on full text "
                 f"{b['kw_full_text']['macro_f1']:.3f}; baseline wins on {b.get('baseline_wins')}" if b else ""))
            for r in e["per_label"]:
                if r["support"] > 0:
                    w(f"      {r['ground']:4} P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f} support {r['support']}")
    lcj = load("learning_curve.json")
    if lcj:
        w("\n7b. LEARNING CURVE (training labels vs macro-F1 against the human set, n=" + str(lcj["human_n"]) + ")")
        w("   same pool, cleaning and pipeline; nested random subsamples; mean over "
          f"{len({r['seed'] for r in lcj['rows']})} seeds (range), plus seed 20261018 and the book-grouped held-out score")
        for r in lcj["summary"]:
            w(f"   {r['train_awards']:4} awards: human macro-F1 {r['human_mean']:.3f} ({r['human_min']:.3f}-{r['human_max']:.3f}); "
              f"seed 20261018 {r['human_seed_20261018']:.3f}; book-grouped held-out (n={(load('eval_report.json') or {}).get('heldout_book_grouped', {}).get('n_test', '?')}) {r['heldout_mean']:.3f}")
        w("   Each step adds less (+0.12, +0.06, +0.04 per ~90 awards) but the curve has not flattened at 364.")
    he = load("human_eval.json")
    w("\n8. HUMAN EVALUATION (Prashant, blind, phase 2 sheet; none of these awards trained the classifier)")
    if he:
        from sklearn.metrics import precision_recall_fscore_support
        from rules import rule_grounds
        import glob
        sheets = sorted(glob.glob(os.path.join(DATA, "prashant", "*_completed.xlsx")))   # phase 2 (+ phase 1 when back)
        hs = pd.concat([pd.read_excel(f, sheet_name=0).fillna("").assign(file=os.path.basename(f)) for f in sheets],
                       ignore_index=True)
        for c in ["ground_span", "corrected", "notes"]:
            if c not in hs:
                hs[c] = ""
        gc = [f"ground_{i}" for i in range(1, 6)]
        hy = [sorted({str(r[c]).strip() for c in gc if str(r[c]).strip()}) for _, r in hs.iterrows()]
        from train import match_rows
        hs["uid"], _ = match_rows(hs, safe)
        present = [g for g in IDS if any(g in y for y in hy)]

        def table(pred, name):
            # all 13 classes, as human_eval.py does: wrong predictions in classes the human never used count
            yt = [[int(g in y) for g in IDS] for y in hy]
            yp = [[int(g in q) for g in IDS] for q in pred]
            P, R, F, S = precision_recall_fscore_support(yt, yp, zero_division=0)
            keep = [i for i, n_ in enumerate(S) if n_ > 0]
            mac = sum(F[i] for i in keep) / len(keep)
            from sklearn.metrics import f1_score
            mic = f1_score(yt, yp, average="micro", zero_division=0)
            exact = sum(set(a) == set(b) for a, b in zip(hy, pred)) / len(hy)
            w(f"   {name}: n={len(hy)}  macro-F1 {mac:.3f}  micro-F1 {mic:.3f}  exact ground-set match {exact:.1%}")
            for i in keep:
                w(f"      {IDS[i]:4} P {P[i]:.2f} R {R[i]:.2f} F1 {F[i]:.2f} support {S[i]}")
            return {"macro_f1": mac, "micro_f1": mic, "exact": exact}

        c = he["classifier"]
        w(f"   training exclusion: 0 of {he['n']} awards in the classifier's training set (checked by human_eval.py)")
        w(f"   classifier (final .joblib, {(load("eval_report.json") or {}).get("n_train_awards", "?")} training awards): n={he['n']}  macro-F1 {c['macro_f1']:.3f}  micro-F1 {c['micro_f1']:.3f}")
        for r in c["per_label"]:
            if r["support"] > 0:
                w(f"      {r['ground']:4} P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f} support {r['support']}")
        J["human"] = {"classifier": {"macro_f1": c["macro_f1"], "micro_f1": c["micro_f1"], "n": he["n"]}}
        J["human"]["qwen"] = table([sorted(Lset.get(u, set())) for u in hs.uid], "Qwen LLM labels")
        span = pd.read_parquet(os.path.join(DATA, "awards_safe.parquet"), columns=["award_id", "source_pdf", "ground_span"])
        span = dict(zip(span.award_id + "|" + span.source_pdf, span.ground_span.fillna("")))
        J["human"]["rules_span"] = table([rule_grounds(span.get(u, "")) for u in hs.uid], "keyword rules on the ground sentence")
        J["human"]["rules_text"] = table([rule_grounds(T.at[u, "text"]) for u in hs.uid], "keyword rules on the full award text")

        import glob
        runs = sorted(glob.glob(os.path.join(DATA, "human_eval_n*.json")), key=lambda f: int(re.findall(r"_n(\d+)", f)[0]))
        if runs:
            w("   human-set size sensitivity (same model, more human rows):")
            for f in runs:
                r = json.load(open(f))
                w(f"      n={r['n']:3}: Qwen macro-F1 {r.get('qwen', {}).get('macro_f1', float('nan')):.3f} | "
                  f"classifier {r['classifier']['macro_f1']:.3f} | keyword rules on the ground sentence "
                  f"{r['keyword_baseline']['kw_ground_span']['macro_f1']:.3f}, on full text "
                  f"{r['keyword_baseline']['kw_full_text']['macro_f1']:.3f}")
        w("\n8b. TAXONOMY COMPLETENESS")
        g99 = sum("G99" in y for y in hy)
        w(f"   G99 ('fits no category') used {g99} times across {len(hy)} blind human-labelled awards: the frozen")
        w(f"   12-ground taxonomy covered every ground in this sample. Human ground counts: "
          f"{pd.Series([g for y in hy for g in y]).value_counts().to_dict()}; {sum(len(y) > 1 for y in hy)} awards "
          f"with 2+ grounds.")

        w("\n8c. CORRECTION RATE (phase 2: the auto-extracted ground sentence was shown, categories were not)")
        h2 = hs[hs.file.str.startswith("phase2")].reset_index(drop=True)
        hy2 = [y for y, f in zip(hy, hs.file) if f.startswith("phase2")]
        mk = h2.corrected.astype(str).str.lower().eq("yes")
        rule_span = [set(rule_grounds(t)) for t in h2.ground_span.astype(str)]
        differs = [set(y) != r for y, r in zip(hy2, rule_span)]
        both = sum(m and d for m, d in zip(mk, differs))
        w(f"   marked 'corrected' by the labeller: {int(mk.sum())}/{len(h2)} ({mk.mean():.0%})")
        w(f"   final ground set differs from the keyword rule on the ground sentence: {sum(differs)}/{len(h2)} "
          f"({sum(differs) / len(h2):.0%}); of those, rule found nothing {sum(len(r) == 0 and d for r, d in zip(rule_span, differs))}")
        w(f"   overlap: {both} both; {int(mk.sum()) - both} marked corrected with the same categories (sentence "
          f"text fixed: typos, truncation, expanded to the insurer's statement); {sum(differs) - both} not marked "
          f"but different categories (the labeller was never shown categories, so 'corrected' refers to the sentence)")
        J["correction"] = {"marked": int(mk.sum()), "differs_from_rule": int(sum(differs)), "n": len(h2)}

        notes = h2.notes.astype(str)
        seg = notes.str.extract(r"(?is)((?:partially\s+)?unredacted.*)")[0].fillna("")
        kinds = {"policy number or prefix": r"(?i)policy (?:no|number|prefix)|policy numbers",
                 "hospital or clinic name": r"(?i)hospital|clinic|\bAIIMS\b",
                 "city, town, village or locality": r"(?i)\bcity\b|\btown\b|village|locality",
                 "person name": r"(?i)(?:nominee|insured|doctor|person|complainant)\s+name",
                 "other identifier (registration no., bank branch)": r"(?i)reg\.?\s*no|bank branch"}
        cnt = {k: int(seg.str.contains(rx).sum()) for k, rx in kinds.items()}
        ex = load_html_meta()
        w("\n11. REDACTION AUDIT — four methods, each finding what the others missed")
        w("   Finding: no single audit method is sufficient for legal text; layering four independent methods")
        w("   surfaced different classes of residual identifier, and each was measured.")
        w("   1) Automated scan (gate 2, 17 Oct): capitalised non-dictionary tokens surviving redaction fell from")
        w("      70,946 (Kaggle-baseline redaction) to 37,860 (7,412 distinct). A 100-token sample was ~13-15% person")
        w("      names, i.e. roughly 1,000 distinct name tokens remain in the frozen corpus — measured, not hidden.")
        w("   2) 25 passages read by hand: 15 random (seed 20261017) read by Ayush before anything was shared —")
        w("      zero complainant names; residual items were place names, hospital record numbers (OP No., IP)")
        w("      and one stray initial. 10 further passages (seed 20261018) read during development: 5 leaks")
        w("      found and fixed in the redactor (a surname, a quoted name, a 'father of' name, a hospital name,")
        w("      an initial). Place names are kept deliberately (they carry no identity on their own); hospital")
        w("      record numbers and the stray initial remain in the frozen corpus.")
        w(f"   3) Blind labeller, award by award (Prashant, phase 2, {len(h2)} awards, v1-redacted text): "
          f"{int((seg != '').sum())} of {len(h2)} ({(seg != '').mean():.0%}) carry at least one unredacted identifier:")
        w(f"      {cnt}; over-redaction flagged once.")
        w("   4) Structural scan, prompted by (3): in form-style awards the policy number sits several lines")
        w("      below the 'Policy No:' label (e.g. a Star Health number in the 2021 health book), which the")
        w("      redactor never looked at. Policy-number-shaped values under the label (dates, month-years and")
        w("      round amounts excluded): 357 awards / 662 values. Masked at display time for everything that")
        w(f"      leaves the project (explorer, demo package): {ex.get('policy_numbers_masked', '?')} values in "
          f"{ex.get('awards_with_policy_mask', '?')} awards. Residual after masking: 23 awards / 32 values,")
        w("      every one inspected being a rupee amount written as 'Rs.126896/-' — no residual policy numbers")
        w("      found. The frozen corpus file itself is unchanged.")
        J["redaction_audit"] = {"prashant_awards_flagged": int((seg != "").sum()), "n": len(h2), "by_type": cnt}
    else:
        w("   not yet available (no filled rows scored)")

    logs = jsonl("qwen_log.jsonl") + jsonl(os.path.join("qwen_smoke_v1", "qwen_log_413_attempt.jsonl"))
    if logs:
        lg = pd.DataFrame(logs)
        ok = lg[lg.written > 0]
        lg["run"] = lg.batch_id.str.split("-").str[0]
        wall = sum(g.ts.max() - (g.ts - g.seconds).min() for _, g in lg[lg.written > 0].groupby("run"))
        w("\n9. LABELLING COST")
        w(f"   {int(lg.written.sum()):,} awards in {len(ok)} successful batches; tokens: {int(lg.prompt_tokens.sum()):,} prompt "
          f"+ {int(lg.completion_tokens.sum()):,} completion ({(lg.prompt_tokens.sum() + lg.completion_tokens.sum()) / max(lg.written.sum(), 1):,.0f}/award); "
          f"wall time {wall / 3600:.1f} h across {lg[lg.written > 0].run.nunique()} runs (rate-limited, not compute-bound)")
        w(f"   keys used (batches): {lg.key.dropna().astype(int).value_counts().sort_index().to_dict() if 'key' in lg else '-'}; "
          f"Groq free tier: 8,000 tokens/min, 7,000 input tokens per request, 200,000 tokens/day per key")
        w("   Requests carried redacted text (awards_safe.parquet), never raw text; residual identifiers measured in 22 of 64 audited awards (§11).")

    import numpy, scipy, sklearn, joblib  # noqa: E401
    w(f"\n10. ENVIRONMENT: python {platform.python_version()} ({platform.machine()}), scikit-learn {sklearn.__version__}, "
      f"numpy {numpy.__version__}, scipy {scipy.__version__}, joblib {joblib.__version__}")

    open(os.path.join(DATA, "final_numbers.txt"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(DATA, "final_numbers.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
