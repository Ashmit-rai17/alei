"""ALEI keyword baseline — standalone. Pure Python 3 standard library: no scikit-learn, no numpy.

  python3 predict_rules.py < award.txt
  python3 predict_rules.py < sample_awards.txt     # several awards, each starting "===== AWARD … ====="

Prints each matched ground and the phrase that triggered it. Same rules as the project's
scripts/rules.py (copied verbatim below); taxonomy v1.0, 12 grounds + G99.
"""


import re

F = re.I | re.S

NONDISC = re.compile(
    r"suppress|conceal|non[- ]?disclos|not\s+disclos|did\s+not\s+(?:disclose|reveal|mention)|withh[eo]ld|"
    r"mis[- ]?stat|mis[- ]?represent|false\s+(?:answer|statement|declaration|information)|incorrect\s+"
    r"(?:answer|statement|information)|wrong\s+(?:answer|statement|information)|not\s+(?:declared|revealed)|"
    r"material\s+(?:facts?|information)|"
    r"(?:suffering|treatment|known\s+case|ailing|diagnosed)[^.]{0,150}?(?:before|prior\s+to)\s+(?:he\s+|she\s+)?"
    r"(?:the\s+)?propos|pre[- ]proposal", F)
REVIVAL = re.compile(r"reviv|reinstat|good\s+health|personal\s+statement\s+(?:regarding|of)\s+health|DGH\b", F)
PROPOSAL = re.compile(r"proposal|application\s+form|at\s+the\s+time\s+of\s+(?:taking|effecting|purchas|buying|"
                      r"proposing|inception)|before\s+(?:taking|purchas)", F)

RULES = {
    "G3": r"deliberate|fraud|intentional|with\s+(?:an\s+)?intent|manipulat|fabricat|forg(?:ed|ery)|tamper|"
          r"(?:under|mis)[- ]?stat\w*\s+(?:his|her|the)?\s*age|age\s+(?:was\s+)?(?:under|mis)[- ]?stat|"
          r"age\s+proof\s+(?:was\s+)?(?:false|fake|incorrect)|fake\b|bogus|impersonat|ground\s+of\s+(?:wrong\s+|incorrect\s+)?age\b|"
          r"age\s+(?:discrepancy|mismatch)",
    "G4": r"\blapse|not\s+in\s+force|premiums?\s+(?:was|were|had)?\s*(?:not\s+(?:been\s+)?paid|unpaid|in\s+default)|"
          r"non[- ]payment\s+of\s+premium|grace\s+period|outside\s+the\s+(?:policy|cover(?:age)?)\s+period|"
          r"policy\s+(?:had\s+)?(?:expired|terminated)|after\s+(?:the\s+)?expiry",
    "G5": r"(?:not|non)[- ]?(?:submi|produc|furnish|provid|suppl)\w*\s.{0,60}?(?:document|record|paper|certificate|"
          r"\bFIR\b|post[- ]?mortem|treatment|requirement|report)|(?:documents?|records?|requirements?)\s+(?:were\s+|was\s+)?"
          r"(?:not\s+(?:submitted|produced|furnished|provided|complied)|pending|outstanding|incomplete)|"
          r"failed\s+to\s+(?:submit|produce|furnish|provide)",
    "G7": r"pre[- ]?existing|\bPED\b|pre[- ]?exist|existed\s+(?:prior|before)|prior\s+to\s+(?:the\s+)?"
          r"(?:inception|commencement|start)\s+of\s+(?:the\s+)?(?:policy|cover)",
    "G8": r"waiting\s+period|first\s+(?:30|thirty|45|forty[- ]five)\s+days|(?:two|four|2|4|three|3)[- ]years?\s+"
          r"(?:waiting|exclusion)|initial\s+(?:30|45)",
    "G9": r"hospitali[sz]ation\s+(?:was\s+)?(?:not|un)\s*(?:necessary|required|justified|warranted)|not\s+"
          r"(?:medically\s+)?necessary|could\s+have\s+been\s+(?:treated|managed|done)\s.{0,30}?(?:OPD|out[- ]?patient|home)|"
          r"\bOPD\s+(?:basis|treatment|procedure)|less\s+than\s+24\s+hours|(?:not|un)\s*justified\s+admission|"
          r"only\s+for\s+(?:investigation|evaluation|diagnostic)|consecutive\s+hours|in[- ]?patient\s+care[^.]{0,40}?hours",
    "G10": r"day[- ]?care|(?:procedure|treatment|surgery|injection|therapy)\s.{0,40}?not\s+(?:covered|payable)|"
           r"not\s+(?:a|an)\s+(?:listed|approved)\s|experimental|unproven|cosmetic|not\s+(?:covered\s+)?under\s+"
           r"(?:the\s+)?(?:surgical|list\s+of)|(?:surgery|disease|treatment|procedure|illness|therapy)[^.]{0,60}?not\s+(?:covered|included|payable)",
    "G11": r"sub[- ]?limit|room\s+rent|non[- ]?medical|consumables?|deduct|capped|reasonable\s+and\s+customary|"
           r"proportionate|co[- ]?pay|partial(?:ly)?\s+(?:settled|paid|settlement)|(?:restricted|limited)\s+to\s+(?:Rs|₹|INR|\d|the\s+(?:sum|amount))",
    "G12": r"suicide|accident(?:al)?\s+(?:death\s+)?benefit|double\s+accident|\bDAB\b|not\s+(?:an?\s+)?accident|"
           r"non[- ]?accidental|murder|homicide|\brider\b|under\s+the\s+influence|intoxicat|self[- ]inflicted",
    "G6": r"exclusion\s+(?:clause|no\.?|number)|excluded\s+under|clause\s*(?:no\.?\s*)?\d+(?:\.\d+)*\s.{0,40}?exclu|"
          r"permanent(?:ly)?\s+exclu|exclusion\s+\d|excluded\s+as\s+per|policy\s+exclusion",
}
RULES = {k: re.compile(v, F) for k, v in RULES.items()}


def rule_grounds(text):
    t = text or ""
    out = set()
    if NONDISC.search(t):
        rev, prop = bool(REVIVAL.search(t)), bool(PROPOSAL.search(t))
        if rev:
            out.add("G2")
        if prop or not rev:
            out.add("G1")
    for g, rx in RULES.items():
        if g == "G6" and out & {"G7", "G8", "G10"}:
            continue
        if rx.search(t):
            out.add(g)
    return sorted(out, key=lambda g: int(g[1:]))


def _words(t, m):
    """The match widened to whole words, for display ('not disclos' -> 'not disclosed')."""
    a, b = m.start(), m.end()
    while a > 0 and t[a - 1].isalnum():
        a -= 1
    while b < len(t) and t[b].isalnum():
        b += 1
    return t[a:b]


def rule_explain(text):
    """Same decision as rule_grounds, plus the phrase that triggered each ground: [(id, phrase)]."""
    t = text or ""
    out = {}
    m = NONDISC.search(t)
    if m:
        rev, prop = REVIVAL.search(t), PROPOSAL.search(t)
        if rev:
            out["G2"] = f"{_words(t, m)} … {_words(t, rev)}"
        if prop or not rev:
            out["G1"] = _words(t, m) + (f" … {_words(t, prop)}" if prop else "")
    for g, rx in RULES.items():
        if g == "G6" and set(out) & {"G7", "G8", "G10"}:
            continue
        mm = rx.search(t)
        if mm:
            out[g] = _words(t, mm)
    return sorted(((g, " ".join(p.split())) for g, p in out.items()), key=lambda x: int(x[0][1:]))


NAMES = {"G1": "non_disclosure_proposal", "G2": "non_disclosure_revival", "G3": "fraud_misrepresentation", "G4": "policy_not_in_force", "G5": "documents_not_submitted", "G6": "excluded_by_clause", "G7": "pre_existing_disease", "G8": "waiting_period", "G9": "hospitalisation_not_justified", "G10": "treatment_not_covered", "G11": "amount_limited", "G12": "cause_outside_cover", "G99": "other"}
_SEP = re.compile(r"(?m)^=+ *(AWARD[^\n=]*?) *=+ *$")


def _main():
    import sys
    text = sys.stdin.read()
    if not text.strip():
        sys.exit("no text on stdin")
    parts = _SEP.split(text)
    awards = [("input", text)] if len(parts) == 1 else list(zip(parts[1::2], parts[2::2]))
    for title, body in awards:
        if len(awards) > 1:
            print(f"\n=== {title.strip()} ({len(body):,} chars) ===")
        hits = rule_explain(body)
        if not hits:
            print("no rule fired")
        for g, phrase in hits:
            print(f"* {g:<4} {NAMES[g]:<30} \"{phrase}\"")


if __name__ == "__main__":
    _main()
