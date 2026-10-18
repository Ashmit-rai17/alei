# Limitations

## Selection bias — read this first
Every award is a dispute that a policyholder took to the Insurance Ombudsman and that reached a
written decision. That is a small, non-random fraction of all repudiations. The data describes the
grounds that were **challenged**, not the grounds insurers typically use, and cannot estimate how
often any ground is used across the market. Outcome rates (who won) describe contested cases only.

## Corpus coverage
- 111 PDFs from archived copies; 10 unreadable (cut off at the archive's 1 MB limit), 3 image-only
  scans (not OCR'd), 12 more partly readable.
- No compilation covers awards decided in 2016–17; 32% of awards have no extracted year. Do not draw
  time trends across that gap.
- LIC of India is 63% of repudiation awards; death claims are 83%. "All insurer" figures are mostly LIC.
- All 170 health repudiation awards come from one book (April 2021); health results cannot be tested
  on an unseen book.
- Insurer is not extracted for 18% of awards, the Ombudsman centre for 30%, the outcome for 38% of
  repudiation awards. Outcome is a rule on the closing text (29 of 32 spot-checked correct), not a field.

## Segmentation
- 5 segments exceed 20,000 characters (long single awards, one joint award, one printed twice).
- An independent check found 3–4 segments that still contain two awards without a readable second
  case number, 7 awards printed twice back-to-back, and 7 segments that end with the next award's header.

## Redaction (measured; see RESULTS.md §11)
- Roughly 1,000 distinct person-name tokens are estimated to survive in the frozen redacted corpus.
- Hospital record numbers (OP/IP), place names (kept deliberately) and a stray initial remain.
- Policy numbers placed below a "Policy No:" label were missed by the redactor; they are masked at
  display time in everything published, but remain in the frozen corpus file. This is why no award
  text other than three masked samples is committed.

## Labels
- 443 of the 1,200 sampled awards were LLM-labelled before the free-tier daily token caps ran out.
- 93.8% of quotes verify verbatim; the rest are near-verbatim (punctuation) or paraphrased, and are
  kept and flagged rather than dropped.
- The LLM over-predicts G7 (pre-existing disease) and under-predicts G12 (cause outside cover) against
  the human labeller; both rest on n ≤ 4.
- Human evaluation is one labeller on 64 awards (no inter-annotator agreement); per-ground support is
  small outside G1 (32), G9 (11) and G10 (10). A second batch of 15 was not returned in time.
- Privacy, three levels: raw award text never left the development machine; redacted text
  (`awards_safe.parquet`) was sent to the labelling API (Groq), and in it identifiers remained in 22 of 64 awards audited one by one — 17 policy numbers or prefixes, 8 hospital or clinic names, 4 place names, 2 other identifiers, 1 person name (categories overlap);
  published outputs additionally mask policy numbers at display time, while the frozen corpus is left
  unchanged so results stay reproducible.

## Classifier
- macro-F1 0.541 on 364 training awards, level with a keyword baseline (0.546). The learning curve
  is still rising, so it is data-limited; it is a cheap offline approximation, not an accuracy claim.
- The book-grouped held-out test set is small (32 awards); its scores move with the split.

## Taxonomy (v1.0, frozen)
- Mis-selling is out of scope (a sales grievance, not a repudiation).
- G6 (excluded by clause) is a heterogeneous bucket by design; G11 mixes "reduced" and "refused".
- Life-specific grounds are thin: G12 merges suicide clauses, rider conditions and early-death windows.
- G99 was used 10 times by the LLM (mostly free-look cancellation refusals) and 0 times by the human.
