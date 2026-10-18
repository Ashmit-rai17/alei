"""Display-time masking for text that leaves the project (explorer, demo package).

The corpus (awards_safe.parquet) is frozen; this is applied only when building outputs.

mask_policy(text) -> (masked_text, [masked values])
  In the block that follows a "Policy No" / "Policy Number" label — up to the next numbered form
  form item ("3. Name of …"), a "Name of the insurer/complainant" label, or 400 chars — replace every policy-number-shaped
  value with [POLICY_NO]. Form-style awards put the value several lines below the label
  ("2 Policy No: / Type of Policy / Duration … / P/######/01/2021/######"), which the redactor
  missed because it only looked right after the label.

Policy-number-shaped: >= 6 digits, may contain / - letters (P/######/01/2021/######, 1234567,
91XXX1234). Not masked: dates (20/07/2020, 07-Feb-2020, 2013/10/10), month-years (05/1997),
round amounts without Rs (250000), financial years (2011-12, 2018-2019),
rupee amounts (Rs.250000/-), redaction tags.
"""

import re

LABEL = re.compile(r"(?i)polic(?:y|ies)\s*(?:no|nos|number|numbers)\b\.?\s*[:\-]?")
# next numbered form item ("3. Name of the insurer"); case-sensitive so "75 Years" / "9 was" do not stop it
STOP = re.compile(r"\n\s*\d{1,2}\s*[.)]?\s*\n?\s*(?:Name|Date|Nature|Type|Amount|Reason|Complaint|Representation|"
                  r"Sum|Premium|Relief)\b|(?i:name\s+of\s+(?:the\s+)?(?:insurer|complainant))")
TOKEN = re.compile(r"(?<![\w/\[])[A-Z]{0,4}[/-]?\d[\w/\-]*\d[\w]*(?![\w\]])")
DATE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
FY = re.compile(r"^(?:19|20)\d{2}[-/](?:\d{2}|(?:19|20)\d{2})$")
MONTH_YEAR = re.compile(r"^\d{1,2}[/-](?:19|20)\d{2}$")
DATE2 = re.compile(r"(?i)^(?:\d{1,2}[-/ ][a-z]{3,9}[-/ ]\d{2,4}|(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})$")
ROUND = re.compile(r"^\d{1,4}0{3,}$")          # 250000, 1500000: an amount written without Rs
BLOCK = 400


def _is_number(tok, before):
    t = tok.rstrip(".-/")
    if sum(c.isdigit() for c in t) < 6:
        return False
    if DATE.match(t) or FY.match(t) or MONTH_YEAR.match(t) or DATE2.match(t) or ROUND.match(t):
        return False
    if re.search(r"(?i)(?:rs\.?|inr|₹)\s*$", before) or tok.endswith("/-"):
        return False
    return True


def mask_policy(text):
    if not text:
        return text, []
    spans, found = [], []
    for m in LABEL.finditer(text):
        end = min(len(text), m.end() + BLOCK)
        s = STOP.search(text, m.end(), end)
        if s:
            end = s.start()
        for k in TOKEN.finditer(text, m.end(), end):
            if _is_number(k.group(0), text[max(0, k.start() - 6):k.start()]):
                spans.append((k.start(), k.end()))
                found.append(k.group(0))
    if not spans:
        return text, []
    out, last = [], 0
    for a, b in sorted(set(spans)):
        if a < last:
            continue
        out.append(text[last:a])
        out.append("[POLICY_NO]")
        last = b
    out.append(text[last:])
    return "".join(out), found


def mask_quote(quote, found):
    """Apply the same replacements to a quote taken from the award, so highlighting still matches."""
    for v in sorted(set(found), key=len, reverse=True):
        quote = quote.replace(v, "[POLICY_NO]")
    return quote
