# Project Alei — Repudiation Grounds Taxonomy v1.0 (FROZEN)

**Frozen 18 Oct 2026.** Do not add, rename or merge categories after this point. Anything that doesn't fit gets `G99` plus a note — that's the designed escape hatch.

Derived from three independent inputs: frequency analysis of the corpus's own repudiation clauses, and an independent human read of 15 awards, reconciled by hand.

---

## The categories

### Shared — apply to any claim type

| ID | Name | Use when the insurer said… |
|---|---|---|
| **G1** | `non_disclosure_proposal` | the policyholder withheld or misstated material facts **on the original application / proposal form** — almost always a health condition |
| **G2** | `non_disclosure_revival` | the policyholder withheld material facts **when reviving or reinstating a lapsed policy** — on the declaration of good health |
| **G3** | `fraud_misrepresentation` | the claim or application was **deliberately** false — age misstatement, manipulated documents, fabricated dates, alleged fraud |
| **G4** | `policy_not_in_force` | the policy had **lapsed**, premium was unpaid, or the event fell outside the policy period |
| **G5** | `documents_not_submitted` | the **claimant** failed to provide required records — FIR, post-mortem, past treatment history, proof of identity |
| **G6** | `excluded_by_clause` | a **named exclusion clause** in the policy applies, cited by number or title, and no other category fits better |

### Health / medical claims

| ID | Name | Use when the insurer said… |
|---|---|---|
| **G7** | `pre_existing_disease` | the condition **existed before cover began** and PED is excluded — whether or not concealment is alleged |
| **G8** | `waiting_period` | a **waiting period had not elapsed** — initial (30/45-day), PED (2/4-year), or condition-specific |
| **G9** | `hospitalisation_not_justified` | admission **wasn't medically necessary**, could have been OPD or home care, or was under 24 hours and not an approved day-care procedure |
| **G10** | `treatment_not_covered` | the **procedure itself** isn't covered — not on the day-care list, standalone treatment excluded, outside the surgical schedule |
| **G11** | `amount_limited` | the claim is valid but **capped or reduced** — sub-limits, room rent, non-medical items, consumables, deductions |

### Life-specific

| ID | Name | Use when the insurer said… |
|---|---|---|
| **G12** | `cause_outside_cover` | the **event itself** isn't covered — suicide clause, murder treated as non-accidental, non-accidental death inside an initial waiting window, rider conditions unmet |

### Escape hatch

| ID | Name | |
|---|---|---|
| **G99** | `other` | A genuine ground that fits none of the above. **Mandatory note describing it.** Using this is correct behaviour, not failure — it's how we find out what v1.0 missed. |

---

## Decision rules — read these, they decide consistency

These are the calls that will otherwise be made differently by different labelers and different models.

**G1 vs G2 — which form was lied on?**
Original proposal → G1. Declaration of good health at revival/reinstatement → G2. Both, in the same award → tick both. This split exists because the legal test differs: the Section 45 clock restarts at revival, so the two grounds don't behave the same way on appeal. Keeping them separate is a finding, not bookkeeping.

**G1 vs G7 — concealment, or just not covered?**
The distinction is what the insurer *argued*:
- "He had diabetes and didn't tell us" → **G1** (the wrong is the concealment)
- "This condition pre-dated the policy, so it's excluded" → **G7** (the wrong is the condition's timing)
- Both arguments made → tick **both**

This is the single most important distinction in the dataset, because "how often does non-disclosure survive independent review" is the question the project exists to answer. Collapsing them destroys it.

**G1/G7 vs G3 — was deliberateness alleged?**
G3 needs an accusation of intent: "deliberate misstatement", "fraudulent", "manipulated", "suppression with intent". Ordinary non-disclosure stays G1. Most awards use "suppression of material facts" as boilerplate — that alone is **not** G3.

**G6 is the fallback, not the default.**
If a named clause excludes a pre-existing disease, that's G7, not G6. If it excludes a specific procedure, G10. Reach for G6 only when the exclusion doesn't fit a more specific category — an unusual clause, a cohabitation condition, an occupation restriction.

**G9 vs G10 — the admission, or the treatment?**
"This didn't need hospitalisation" → G9. "This procedure isn't covered wherever you had it" → G10. An award arguing both ("not 24 hours *and* not on the day-care list") gets both.

**G5 is about the claimant, not the policyholder.**
Failure to supply documents *after* the claim. Not the same as withholding facts on the application.

**Partial payments still carry a ground.**
"We paid but deducted non-medical items" → G11. "We paid the base claim but rejected the accident rider" → G12 (plus whatever ground the rider rejection used).

**Multiple grounds are normal.** Roughly a third of awards cite two or more. Record every ground the insurer relied on, not the strongest one.

**Only the insurer's stated ground counts.** Not the Ombudsman's reasoning, not the outcome. If the Ombudsman says "the real issue is X" but the insurer said Y, the label is Y.

---

## Worked examples, from the human-read set

| Award | Grounds | Why |
|---|---|---|
| Hypertension 5 years before policy, hidden on application | **G1** | Concealment at proposal |
| Lapsed policies revived; chronic alcoholism and liver hospitalisation not declared on the health statement | **G2** | Revival declaration, not original proposal |
| Diabetes pre-existing, policy in year 4 of a 4-year PED waiting period, **and** claimant didn't supply treatment records | **G7 + G8 + G5** | Three distinct arguments in one award |
| COVID claim: mild symptoms, home quarantine per guidelines; later partial pay with non-medical deductions | **G9 + G11** | Admission not justified, then amount capped |
| Eye injections: not 24 hours, not on day-care list, ARMD excluded | **G9 + G10 + G6** | Admission, procedure, and a named exclusion |
| Cohabitation exclusion clause **and** occupation not declared | **G6 + G1** | Unusual clause plus proposal non-disclosure |
| Death on day 43, policy has a 45-day non-accidental-death clause | **G12** | The event is outside cover |
| Accident rider rejected: undisclosed hemiplegia **and** murder isn't an accident | **G1 + G12** | Non-disclosure plus event outside rider cover |
| Age understated by 21 years, death certificate possibly manipulated | **G3** | Deliberateness explicitly alleged |

---

## Recording format

`ground_1` … `ground_5`, left-packed, using the **ID** (`G1`, `G7`), not the name. Empty cells after the last ground. `G99` always needs a note.

---

## Known gaps in v1.0

Stated now so they're disclosed rather than discovered:

- **Misselling is out of scope.** It's a sales grievance, not a claim repudiation. Those awards stay in the corpus, excluded from grounds analysis.
- **G6 will be heterogeneous.** It's a bucket of one-off clauses by design. Report its share and don't over-interpret it.
- **Life-specific grounds are thin.** `cause_outside_cover` merges suicide clauses, accident-rider conditions and initial-waiting-window deaths. A larger corpus would split these.
- **G11 blurs "reduced" and "refused".** A capped payment isn't a repudiation in the strict sense, but insurers argue it with the same clause language, so it's tracked.
