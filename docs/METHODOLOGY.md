# Methodology

## 1. Acquisition (`src/acquisition/`)
`discover.py` queries the Internet Archive CDX API for award compilations published by the Insurance
Ombudsman offices (388 URLs). `download.py` tries the live site first, then the archived copy
(`web.archive.org/web/{ts}id_/`), with a 2-second delay between requests.

## 2. Inventory and truncation (`src/parsing/inventory.py`)
pymupdf reads every PDF page by page, so one bad page cannot lose a book. A file counts as truncated
if pymupdf had to repair its cross-reference table on open, the `%%EOF` trailer is missing, or it is
exactly 1,048,576 bytes (the Internet Archive's size cap). Outcome: 111 PDFs — 98 with text,
3 image-only scans, 10 unreadable (all at the archive cap), 22 truncated in total, 86 clean.

## 3. Segmentation (`src/parsing/segment.py`)
A compilation stitches together sections from different Ombudsman centres, each with its own header
style. Award boundaries are the **union** of several header patterns (complaint ids with spaced
digits, "Case No", "Complaint (Ref) No", award numbers, ids under "<City> Ombudsman Centre", Bhopal
serials). Dashes are normalised one-to-one so match positions stay valid. Hits close together form
one header (joint awards chain at short distance). A repeated id is a page header only while it
belongs to the current award; elsewhere it is a reprint, deduplicated keep-longest. Prose such as
"case no. 74/13 which is pending" is rejected. Boundaries are pulled back to an "AWARD IN THE MATTER
OF / PROCEEDINGS BEFORE THE INSURANCE OMBUDSMAN" block when a party line follows, so the next award's
header does not bleed into the previous one. A weld detector reads ids the same way the splitter does
and reports multi-header segments. Dates: the "Date of Award/Order" field wins, then inline markers;
there is no first-date fallback (it returned policy dates). `ground_span` is the sentence after
"repudiated … on the ground(s) that/of", with periods before digits and after abbreviations masked so
"Rs. 5,000" does not end the sentence. `is_repudiation` is set from repudiation language or a filled
"Date of repudiation" field.

## 4. Redaction (`src/privacy/redact.py`)
Two passes. Pass 1 is the original notebook's patterns with three fixes: postal PINs only in postal
context (so rupee amounts survive), doctor names limited to two words and stopped at insurer names,
and name lines stopped at stopwords. Pass 2 adds party lines before "Vs", form blocks (name/address
of the complainant, representatives), Ombudsman signatures, relation mentions ("X, father of"),
quoted names, hospital names and acronyms with a place, street addresses, address lines, 7-digit and
postal-tracking numbers. Every name harvested in an award is then scrubbed wherever it recurs in that
award, and capitalised tokens touching a `[NAME]` tag are absorbed into it. "Common word" exclusions
are learned from the corpus, because dictionary word lists contain common Indian surnames.

Audit, four independent methods (results in RESULTS.md §11): an automated scan of surviving
capitalised non-dictionary tokens; 25 passages read by hand; a blind labeller flagging identifiers
award by award; and a structural scan that found policy numbers several lines below a "Policy No:"
label. Those are masked at display time (`src/privacy/display_mask.py`) in everything published;
the frozen corpus is left unchanged so results stay reproducible. The outcome is a measured residual
rate, not a guarantee: identifiers remained in 22 of 64 awards audited one by one — 17 policy numbers or prefixes, 8 hospital or clinic names, 4 place names, 2 other identifiers, 1 person name (categories overlap). That audit was of the v1 redaction of those awards; the API received a later, improved redaction. Compared word by word, the later redaction masks everything v1 masked in the text the two versions share, so 22/64 is an upper bound for that text. It is not an upper bound for everything the API saw: re-segmentation added 689 words to 15 of the 64 awards that no human audited, and the improved redaction as a whole has not been measured. The redacted text, residuals included, is what the labelling
API received; raw text never left the machine.

## 5. Labelling sample (`src/labelling/sample_for_labeling.py`)
A free-tier LLM API allows ~200k tokens per key per day, so 1,200 of the 3,831 repudiation awards
were sampled: all 79 human-labelled awards, all 170 health awards (the only health source), and a
fill stratified by claim type and spread across books in proportion to their size. Seed 20261018.

## 6. LLM labelling (`src/labelling/qwen_label.py`)
The prompt gives the 12 grounds of the frozen taxonomy (`docs/TAXONOMY.md`) with their "use when"
text and the decision rules verbatim (G1 vs G2, G1 vs G7, G1/G7 vs G3, G6 as fallback, …). Rules:
only the **insurer's** stated ground — not the Ombudsman's reasoning or the outcome; every ground with
an **exact** quote; quote the insurer's own statement, and never a sentence where the Ombudsman
rejects the insurer's argument; mark each quote's source (insurer/ombudsman); unknown grounds go to
G99 with a description. Requests are packed to a token budget, paced under the per-minute limit,
checkpointed every batch, and rotate across API keys on a daily-cap error. Malformed JSON is retried
once. Every quote is checked against the award text (whitespace normalised, nothing else); failures
are kept and flagged, so the verification rate is a measured hallucination rate. Repeated grounds
within an award collapse to one.

## 7. Keyword rules (`src/labelling/rules.py`)
Regular expressions per ground, encoding the same decision rules, applied to the extracted ground
sentence (`rule_labels.py`) or to full text. They serve as the fallback labeller and as the baseline
the classifier must beat. They are deliberately not tuned on the judgement grounds (G3, G6, G7): the
gap between rules and LLM there is the finding that justifies the LLM layer.

## 8. Classifier (`src/modelling/train.py`)
TF-IDF word 1–3-grams plus character 3–5-grams (robust to OCR noise), min_df 3, sublinear tf;
one-vs-rest logistic regression with balanced class weights; all 13 classes modelled, low-support
ones flagged rather than dropped. Award separators ("-----", "*****") are stripped before
vectorising; the regex is stored in the model. The headline split is grouped by source book (no book
in both train and test) and stratified by claim type; because every health award comes from one book,
health is scored separately within that book and labelled optimistic. The human-labelled awards are
always held out of training, and the model records which awards it trained on.

## 9. Evaluation (`src/modelling/human_eval.py`, `learning_curve.py`)
An independent labeller (no AI tools, see `labelling/handoff.py`) assigned grounds to 64 awards
blind. The classifier, the LLM labels and the keyword rules are each scored against those labels,
per ground with support. The learning curve retrains on nested random subsamples (90/180/270/364),
6 seeds, and scores each against the same human set.

## 10. Reporting (`src/reporting/`)
`corpus_stats.py` (counts with denominators and caveats), `outcome_analysis.py` (outcome by claim
type, era, insurer; outcome is a heuristic on the closing text, spot-checked), `build_explorer.py`
(a single offline HTML file where every number clicks through to the awards and the highlighted
quote), `final_numbers.py` (every quoted figure in one generated file, `docs/RESULTS.md`),
`build_demo.py` (the LinuxONE package).
