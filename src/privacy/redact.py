"""Stage 5 — redact: data/awards.parquet -> data/awards_safe.parquet.

awards.parquet never leaves this machine. awards_safe.parquet is the only
shareable artifact — and only after a human has read the printed samples.

Two passes per string:
  1. scrub_v1   — the Kaggle notebook patterns in their order, with three
                  amended: PIN needs postal context (a bare six-digit match ate rupee
                  amounts and cheque numbers), DOCTOR and NAME_LINE stop at
                  stopwords (they swallowed "Vs Star Health" and "Rs.").
                  The unmodified version is kept as scrub_kaggle, used only as
                  the baseline of the leak check.
  2. scrub_v2   — patterns added to close the known gaps (ALL-CAPS names,
                  initials, party lines before "Vs", "In the matter of",
                  deceased/relations, the complainant name/address form block).
Then, per award, every name token harvested from the raw text is scrubbed
wherever else it appears in that award (second mentions without an
honorific are the commonest leak).

Columns redacted: text, every sec_*, ground_span (it is a sentence copied out
of text), and award_id (ids from the loose "Complaint No." pattern can run on
into the following words).
"""

import os
import random
import re
import sys
from collections import Counter

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from inventory import DATA  # noqa: E402
from segment import AWARDS, INSURERS, INSURER_RE  # noqa: E402

SAFE = os.path.join(DATA, "awards_safe.parquet")
REVIEW = os.path.join(DATA, "redaction_review.txt")


# ------------------------------------------------------------
# pass 1 — Kaggle patterns (scrub_kaggle is verbatim; scrub_v1 is what runs)
# ------------------------------------------------------------

NAME_LINE = re.compile(
    r"(Complainant|Insured|Life Assured|Nominee|Shri|Smt|Mr\.?|Mrs\.?|Ms\.?|"
    r"Late Shri|Late Smt)\s+((?:[A-Z][a-z]+\.?\s*){1,4})")
POLICY = re.compile(
    r"\b(?:Policy|Certificate|Claim|Membership|Card|UHID|Proposal)\s*"
    r"(?:No\.?|Number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]{4,25})", re.I)
LONGNUM = re.compile(r"\b\d{8,}\b")
PHONE = re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w{2,}\b")
HOSPITAL = re.compile(
    r"\b([A-Z][A-Za-z&.\s]{2,35}?\s(?:Hospital|Nursing Home|Medical Centre|"
    r"Medical Center|Clinic|Healthcare|Health Care|Institute of Medical Sciences))\b")
DOCTOR = re.compile(r"\bDr\.?\s+([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){0,3})")
PIN = re.compile(r"\b\d{6}\b")


def scrub_kaggle(t):
    """The notebook's scrub, unmodified. Baseline for the leak check only."""
    if not isinstance(t, str):
        return t
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[PHONE]", t)
    t = POLICY.sub(lambda m: m.group(0).replace(m.group(1), "[POLICY_NO]"), t)
    t = LONGNUM.sub("[NUM]", t)
    t = HOSPITAL.sub("[HOSPITAL]", t)
    t = DOCTOR.sub("Dr. [NAME]", t)
    t = NAME_LINE.sub(lambda m: f"{m.group(1)} [NAME]", t)
    t = PIN.sub("[PIN]", t)
    return t


# ------------------------------------------------------------
# pass 2 — gap-closing patterns
# ------------------------------------------------------------

# words that end a name when they follow one (Title-case or CAPS)
STOP = {w.lower() for w in """
vs versus the of and or in on at to for by with from as is was were has had be
insurance insurer insured insurers life assured assurance company co ltd limited pvt private
corporation corp india indian policy policies claim claims complainant complainants
respondent respondents nominee nominees proposer proposal death maturity survival benefit
hospital nursing home clinic medical centre center health care bank branch division divisional
office officer manager agent advocate dated date award awards case no number ombudsman forum
rule rules sum plan premium premiums person persons member members group scheme
general state government lic dla la dp late deceased shri smt sri sh mr mrs ms miss dr kumari
jan january feb february mar march apr april may jun june jul july aug august sep sept september
oct october nov november dec december name address complaint complaints ref facts brief
contention contentions observation observations decision reason reasons conclusion findings
hon honble court supreme high district consumer commission national hearing
name num pin policy_no hospital email phone name_address mis selling misselling
he she his her him they their who which that this
lakh lakhs lac lacs crore crores time times year years month months day days more such other
rs inr rupees insured policy award complaint insurer company limited ltd dated dt order cheque amount
bonus section act proceedings before misc miscellaneous signed digitally signer place
monday tuesday wednesday thursday friday saturday sunday
however thus further therefore hence moreover accordingly also again after since while although
though whereas meanwhile finally firstly secondly subsequently consequently besides nevertheless
regarding considering according being having during under upon it there here then now when where
if but so yet both all any each every not only said these those we our its an
north south east west central new navi greater
andhra pradesh arunachal assam bihar chhattisgarh chattisgarh goa gujarat haryana himachal jharkhand
karnataka kerala madhya maharashtra manipur meghalaya mizoram nagaland odisha orissa punjab
rajasthan sikkim tamil nadu tamilnadu telangana tripura uttar uttarakhand uttaranchal bengal jammu
kashmir ladakh andaman nicobar dadra haveli daman diu lakshadweep puducherry pondicherry
ahmedabad bengaluru bangalore bhopal bhubaneswar chandigarh chennai madras delhi guwahati hyderabad
secunderabad jaipur ernakulam kochi cochin kolkata calcutta lucknow mumbai bombay noida patna pune
thane cuttack coimbatore madurai nagpur indore kanpur varanasi allahabad prayagraj agra meerut
ludhiana amritsar jalandhar surat vadodara baroda rajkot nashik aurangabad visakhapatnam vizag
vijayawada trivandrum thiruvananthapuram kozhikode calicut mysore mysuru mangalore hubli belgaum
panaji ranchi jamshedpur dhanbad raipur bilaspur dehradun shimla srinagar gurgaon gurugram faridabad
ghaziabad howrah durgapur siliguri asansol gwalior jabalpur jodhpur udaipur kota ajmer bikaner
vellore salem tiruchirappalli trichy tirunelveli nellore guntur warangal goa
""".split()}
STOP |= {w.lower() for ins in INSURERS for w in re.split(r"\W+", ins) if w}
# insurer brand words not in the segment list (so absorbing never eats "Aegon Religare")
STOP |= set("""religare aegon metlife met birla sun reliance nippon tata aia aig max new york bharti axa aviva
ing vysya dhfl edelweiss tokio sahara shriram future generali star union daiichi idbi federal canara hsbc
obc indiafirst kotak exide pnb sbi hdfc icici prudential lombard bupa apollo munich cigna manipal care
iffco cholamandalam royal sundaram universal sompo liberty magma digit acko raheja qbe oriental
national united ergo""".split())
# after a [NAME], these are surname parts, not honorifics ("[NAME] Kumari")
STOP_AFTER = STOP - {"kumari", "kum", "km", "late"}

TOKEN = r"(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z](?=[.\s]))"
# extraction often puts one word per line, so a name may cross one newline
NAME = TOKEN + r"(?:(?:[ \t]*\.[ \t]*\n?[ \t]*|[ \t]*\n[ \t]*|[ \t]+)" + TOKEN + r"){0,5}\.?"

_HON_WORDS = ["Late Shri", "Late Smt", "Shri", "Smt", "Sri", "Sh", "Mrs", "Mr", "Ms", "Miss",
              "Kumari", "Kum", "Km", "Master", "Dr", "Late", "Lt", "Mst"]
HON = r"(?:" + "|".join(sorted({w for h in _HON_WORDS for w in (h, h.upper())} | {"late"},
                               key=len, reverse=True)) + r")"
# "Mr. X", "Mr X", "Shri.M.Vasantha", "MrL. Ramesh"
HON_NAME = re.compile(r"\b(" + HON + r")(\b\)?[ \t]*\.?[ \t]*\n?[ \t]*|(?<=[a-z])(?=[A-Z]))(" + NAME + r")")

IN_MATTER = re.compile(r"(?i:\b(in the matter of)\b)([ \t:\-]*(?:" + HON + r"\.?[ \t]*)?)(" + NAME + r")")

RELATION = re.compile(
    r"(?i:\b(deceased(?:[ \t]+life[ \t]+assured)?|D\.?L\.?A\.?|life[ \t]+assured|L\.A\.|insured|"
    r"complainants?|nominee|proposer|claimant|policy[ \t]?holder|wife|husband|son|daughter|father|"
    r"mother|brother|sister|widow|[sdwc]/o|one|(?:in[ \t]+the[ \t]+)?name[ \t]+of)\b)"
    r"([ \t,:.\-]*(?:" + HON + r"\.?[ \t]*)?)(" + NAME + r")")

NAME_IS = re.compile(
    r"(?i:\b(names?\b[^.]{0,40}?\b(?:is|as|was|were)))([ \t:]*[”“\"‘’'―‖]?)(" + NAME + r")")

_VS = r"(?:Vs|VS|vs|V/s|V/S|v/s|Versus|VERSUS|versus)\.?"
_VS_END = r"(?=[ \t\n]|[A-Z])"
_PARTY_LABEL = (r"(?i:[ \t]*(?:case[ \t]+of[ \t]+(?:the[ \t]+)?(?:complainant)?|in[ \t]+the[ \t]+matter[ \t]+of|"
                r"complainant|between)[ \t]*[:\-–]?)?[ \t]*")
# "<Firstname Surname> vs HDFC Standard Life" — anything before vs is a person
PARTY_LINE = re.compile(r"(?m)^(" + _PARTY_LABEL + r")([^\n]{2,80}?)([ \t]+" + _VS + _VS_END + r")")
# "ARNEST MOHANBHAI CHRISTIAN\nVS\nRESPONDENT ..."
PARTY_PREV = re.compile(r"(?m)^(" + _PARTY_LABEL + r")([^\n]{2,80}?)([ \t]*(?:\n[ \t]*)+" + _VS + r"[ \t]*$)")

# form block: name + address of the complainant, up to the next numbered item
# where a form value ends: the next numbered item, dotted or not, at line
# start or run onto the same line ("Complainant Mr X 2 Policy No:")
_NEXT_ITEM_ALTS = (r"\n[ \t]*\d{1,2}(?:[ \t]*[.)]|[ \t]+[A-Z])"
                   r"|\b\d{1,2}[ \t]*[.)]?[ \t]*(?:Policy|Type|Name|Date|Nature|Amount|Reason|Duration|Sum|"
                   r"Complaint)\b|\Z")
_NEXT_ITEM = "(?=" + _NEXT_ITEM_ALTS + ")"

# label may wrap and some books drop doubled letters ("Adres", "Complain\nant")
FORM_ADDR = re.compile(
    r"(Name\s*(?:&|and)\s*Ad{1,2}res{1,2}\s+of\s+(?:the\s+)?Complain\s*ant[ \t:\-]*)"
    r"([\s\S]{1,400}?)" + _NEXT_ITEM, re.I)
# two-column layout: "Name & Address of :\n Shri X,\n complainant\n <address>"
FORM_ADDR2 = re.compile(
    r"(Name\s*(?:&|and)\s*Ad{1,2}res{1,2}\s+of\s*:\s*)(?=[\s\S]{0,80}?\bcomplainant\b)"
    r"([\s\S]{1,400}?)(?=\n[^\n]*Name\s*(?:&|and)\s*Ad{1,2}res{1,2}|\n[^\n]*Present on behalf|\Z)", re.I)
# representatives at the hearing: "b) For the Insurer\n<Officer Name>\n13."
FORM_REP = re.compile(
    r"(For[ \t]+the[ \t]+(?:complainant|insurer|respondent|insurance[ \t]+company|company)[ \t:\-]*)"
    r"([\s\S]{1,120}?)(?=\n[ \t]*[a-z]\)|" + _NEXT_ITEM_ALTS + ")", re.I)
# facilities the verbatim HOSPITAL pattern misses (lower-case suffix, labs)
FACILITY = re.compile(
    r"\b((?:[A-Z][A-Za-z&.'’]+[ \t]*\n?[ \t]*){1,4}(?i:hospitals?|nursing[ \t]+home|medical[ \t]+cent(?:re|er)|"
    r"clinic|health[ \t]?care|health[ \t]+cent(?:re|er)|diagnostics?|laborator(?:y|ies)|polyclinic))\b")
FORM_NAME = re.compile(
    r"(?m)(^[ \t]*(?:\d{1,2}[ \t]*[.)]?[ \t]*)?Name[ \t]+of[ \t]+(?:the[ \t]+)?(?:life[ \t]+)?"
    r"(?:insured|complainant|policy[ \t]?holder|proposer|life[ \t]+assured|nominee|patient|deceased|"
    r"claimant|la|dla)\b[ \t:\-]*)"
    r"([\s\S]{1,250}?)" + _NEXT_ITEM, re.I)

# a short line that looks like a postal address, not a sentence
ADDRESS_LINE = re.compile(
    r"(?m)^(?=[^\n]{0,100}$)(?=[^\n]*(?:,|\d))(?![^\n]*\b(?:the|was|were|is|has|had|which|who|from|with|"
    r"for|that|his|her|of the)\b)(?![^\n]*\bRoad[ \t]+(?:Traffic|accident))[^\n]*\b(?:Street|St\.|Road|Rd\.?|"
    r"Nagar|Colony|Lane|Marg|Avenue|Sector|Village|Vill\.?|Vil\.|VPO|P\.[ \t]?O\.|Teh\.?|Tehsil|Distt?\.?|"
    r"District|Taluka?|Mandal|Ward|Flat|Apartments?|Society|Layout|Cross|Bazaa?r|Chowk|Mohalla|Gali|"
    r"Puram|Pally|Wadi|H\.[ \t]?No\.?|House[ \t]+No\.?)\b[^\n]*$")
# "<Firstname Middle Surname>, father of the complainant" — a bare name in apposition to a relation
NAME_RELATION = re.compile(
    r"(" + NAME + r")(?=,?[ \t]+(?:the[ \t]+)?(?:father|mother|husband|wife|son|daughter|brother|sister|"
    r"widow|nominee)[ \t]+of\b)")
# "Holy spirit hospital": a capitalised first word, then lower-case words
LOWER_HOSPITAL = re.compile(r"\b([A-Z][a-z]+(?:[ \t]+[a-z]+){0,2}[ \t]+(?:hospital|nursing home|clinic))\b")
# hospital acronyms followed by a place: "SSSIMS, Puttaparthy", "CMC, Vellore", "NIMS Hyderabad"
HOSP_ACRONYM = re.compile(
    r"\b(?:SSSIMS|SVIMS|NIMS|KIMS|SIMS|AIIMS|JIPMER|NIMHANS|PGIMER|PGI|CMC|KMC|KMCH|DMC|SCB|KEM|AMRI|MGM|"
    r"ESI|ASRAM|MOSC|CSI|GKNM|SRM|MKCG|[A-Z]{1,4}IMS|[A-Z]{2,3}MCH)\b"
    r"(?:[ \t]*(?:Hospital|Medical[ \t]+College))?,?[ \t]*\n?[ \t]*[A-Z][a-z]{3,}")
# inline street address: "59-B, <Name> Street, Kolkata", "4, C.R. AVENUE, KOLKATA"
STREET = re.compile(
    r"\b\d{1,4}[ \t]?[-/]?[ \t]?[A-Z]?,[ \t]*(?:[A-Z][A-Za-z.]*[ \t]+){1,4}"
    r"(?i:road|rd\.|street|st\.|marg|avenue|lane)\b(?:,?[ \t]*[A-Z][A-Za-z]+)?")
# 7-digit policy numbers (8+ are LONGNUM already) — not amounts
NUM7 = re.compile(r"(?<!Rs)(?<!Rs\.)(?<!Rs\. )(?<!INR )(?<![\d,.])\d{7}(?!\d)(?!,\d)")
# postal tracking numbers: EM244096161IN, ED230485440 IN
TRACKING = re.compile(r"\b[A-Z]{2}[ \t]?\d{9}[ \t]?IN\b")


PLACEHOLDER = re.compile(r"\[(?:NAME|NUM|PIN|POLICY_NO|HOSPITAL|EMAIL|PHONE|NAME_ADDRESS|ADDRESS)\]")


def _trim(name, max_words=None):
    """Leading run of tokens that look like a name; '' if none.
    max_words caps the number of non-initial words kept."""
    toks = re.findall(r"[A-Za-z]+\.?", name)
    keep, words = [], 0
    for tok in toks:
        low = tok.rstrip(".").lower()
        if low in STOP or low in COMMON:
            break
        if len(tok.rstrip(".")) > 1:
            if max_words is not None and words >= max_words:
                break
            words += 1
        keep.append(tok)
    if not keep:
        return ""
    # cut name at the end of the last kept token
    pos = 0
    for tok in keep:
        pos = name.index(tok, pos) + len(tok)
    return name[:pos]


# A six-digit number is a postcode only in postal context: after "PIN"/"Pin
# Code", after a dash ("Kolkata - 700001"), or after a place name at the end of
# a line. Never after Rs/amount/premium/No. — those are money or identifiers.
PIN_CAND = re.compile(r"(?<![\d,./])[1-9]\d{2}[ \t]?\d{3}(?![\d,]|/-|/=)")
PIN_LABEL = re.compile(r"(?i)\bpin(?:[ \t]*code)?\b[ \t]*[:\-–.]?[ \t]*$")
PIN_DASH = re.compile(r"[-–][ \t]?$")
PIN_PLACE = re.compile(r"\b[A-Z][A-Za-z]+,?[ \t]+$")
PIN_NOT = re.compile(r"(?i)(?:\b(?:rs\.?|inr|amount|sum[ \t]+assured|sum[ \t]+insured|premium|nos?\.?|"
                     r"number|cheque|dated|dt\.?)|₹)[ \t:.\-]*$")


def scrub_pin(t):
    def f(m):
        pre = t[max(0, m.start() - 25):m.start()]
        if PIN_NOT.search(pre):
            return m.group(0)
        if PIN_LABEL.search(pre) or PIN_DASH.search(pre):
            return "[PIN]"
        rest = t[m.end():t.find("\n", m.end()) if "\n" in t[m.end():] else len(t)]
        if PIN_PLACE.search(pre) and re.fullmatch(r"[ \t]*[.,]?[ \t]*", rest):
            return "[PIN]"
        return m.group(0)
    return PIN_CAND.sub(f, t)


def scrub_v1(t):
    if not isinstance(t, str):
        return t
    t = EMAIL.sub("[EMAIL]", t)
    t = PHONE.sub("[PHONE]", t)
    t = POLICY.sub(lambda m: m.group(0).replace(m.group(1), "[POLICY_NO]"), t)
    t = LONGNUM.sub("[NUM]", t)
    t = HOSPITAL.sub("[HOSPITAL]", t)

    def doctor(m):          # at most two name words; never "Vs Star Health"
        core = _trim(m.group(1), max_words=2)
        return f"Dr. [NAME]{m.group(1)[len(core):]}" if core else m.group(0)
    t = DOCTOR.sub(doctor, t)

    def name_line(m):       # "Sum Insured Rs.2,00,000" is not a name
        core = _trim(m.group(2))
        return f"{m.group(1)} [NAME]{m.group(2)[len(core):]}" if core else m.group(0)
    t = NAME_LINE.sub(name_line, t)
    return scrub_pin(t)


def _is_party_name(s):
    s = PLACEHOLDER.sub("", s).strip(" \t-–:,.")
    return (len(s) >= 2 and not INSURER_RE.search(s)
            and not re.search(r"(?i)\b(?:insurance|assurance|company|co\.|ltd|limited|corporation|"
                              r"bank|court|commission|forum|ombudsman|respondent|insurer)\b", s)
            and re.search(r"[A-Za-z]{2,}", s))


# Ombudsman / signatory names. "(G.S.Shrivastava)" near "Ombudsman"; a line
# that is only initials + surname ("K.B.Saha", "C.S. PRASAD"); the name line
# above "Insurance Ombudsman"; "Ombudsman Name : X", "Signer: X".
_SIG_TOKS = r"((?:[A-Z][A-Za-z]*\.?[ \t]*){1,5})"
_OMB = r"(?i:o\s?m\s?b\s?u\s?d\s?s\s?m\s?a\s?n)"     # survives "O\nmbudsman"
# groups are (head, sep, name) for Scrubber._name_sub; trailing context is lookahead
PAREN_SIG = re.compile(r"(\()(\s*)" + _SIG_TOKS + r"(?=\s*\)[\s\S]{0,400}?" + _OMB + ")")
# at line start, or right-aligned after a wide gap ("Noida.      C.S. PRASAD")
INITIALS_LINE = re.compile(r"(?m)(^[ \t]*|[ \t]{3,})()((?:[A-Z]\.[ \t]*\n?[ \t]*){1,3}[A-Z][A-Za-z]{2,})(?=[ \t]*$)")
ABOVE_OMB = re.compile(r"(?m)^([ \t]*)()" + _SIG_TOKS + r"(?=[ \t]*\n(?:[ \t]*\n){0,12}[ \t]*"
                       r"(?:(?i:insurance)\s*" + _OMB + r"|(?i:dated\b)))")
# single-line name only: extraction breaks words across lines ("STAT\nE OF ODISHA")
_NAME_1L = TOKEN + r"(?:(?:[ \t]*\.[ \t]*|[ \t]+)" + TOKEN + r"){0,5}\.?"
OMB_LABEL = re.compile(r"(?i:\b(ombudsman(?:[ \t]+name)?|signer))([ \t]*[:\-–œ,]*[ \t]*(?:" + HON +
                       r"\.?[ \t]*)?)(" + _NAME_1L + r")")


class Scrubber:
    """Pass-2 substitutions; counts every replacement and harvests name tokens."""

    def __init__(self):
        self.hits = Counter()
        self.names = set()

    def _name_sub(self, label):
        def f(m):
            head, sep, name = m.group(1), m.group(2), m.group(3)
            core = _trim(name)
            if not core:
                # "(Late) Smt. V.Haseena": the outer match swallowed an inner
                # honorific that the trimmer rejects — retry on the remainder
                inner = HON_NAME.sub(self._name_sub("honorific_any_case"), name)
                return f"{head}{sep}{inner}"
            self.hits[label] += 1
            self.names.add(core)
            return f"{head}{sep}[NAME]{name[len(core):]}"
        return f

    def _party_sub(self, label):
        def f(m):
            if not _is_party_name(m.group(2)):
                return m.group(0)
            self.hits[label] += 1
            self.names.add(m.group(2))
            return f"{m.group(1)}[NAME]{m.group(3)}"
        return f

    def _form_sub(self, label, repl):
        def f(m):
            if not m.group(2).strip() or PLACEHOLDER.fullmatch(m.group(2).strip()):
                return m.group(0)
            self.hits[label] += 1
            self.names.update(x for x in m.group(2).split("\n")[:3] if x.strip())
            return f"{m.group(1)}{repl}"
        return f

    def scrub(self, t):
        if not isinstance(t, str):
            return t
        t = FORM_ADDR.sub(self._form_sub("form_name_address", "[NAME_ADDRESS]"), t)
        t = FORM_ADDR2.sub(self._form_sub("form_name_address", "[NAME_ADDRESS]\n"), t)
        t = FORM_NAME.sub(self._form_sub("form_name", "[NAME]\n"), t)
        t = FORM_REP.sub(self._form_sub("form_representative", "[NAME]"), t)
        t = PARTY_PREV.sub(self._party_sub("party_line_prev"), t)
        t = PARTY_LINE.sub(self._party_sub("party_line"), t)
        for rx in (PAREN_SIG, INITIALS_LINE, ABOVE_OMB, OMB_LABEL):
            t = rx.sub(self._name_sub("ombudsman_signatory"), t)
        t = IN_MATTER.sub(self._name_sub("in_the_matter_of"), t)
        t = HON_NAME.sub(self._name_sub("honorific_any_case"), t)
        t = RELATION.sub(self._name_sub("deceased_relation"), t)
        t = NAME_IS.sub(self._name_sub("name_is"), t)

        def name_rel(m):
            core = _trim(m.group(1))
            if not core or core != m.group(1).rstrip(". "):
                return m.group(0)
            self.hits["name_relation"] += 1
            self.names.add(core)
            return "[NAME]"
        t = NAME_RELATION.sub(name_rel, t)

        def low_hosp(m):
            first = m.group(1).split()[0].lower()
            if first in STOP or first in COMMON:
                return m.group(0)
            self.hits["facility"] += 1
            return "[HOSPITAL]"
        t = LOWER_HOSPITAL.sub(low_hosp, t)
        t, n = ADDRESS_LINE.subn("[ADDRESS]", t)
        self.hits["address_line"] += n
        t, n = TRACKING.subn("[NUM]", t)
        self.hits["tracking_no"] += n
        t, n = NUM7.subn("[NUM]", t)
        self.hits["num_7digit"] += n
        t, n = HOSP_ACRONYM.subn("[HOSPITAL]", t)
        self.hits["hospital_acronym"] += n
        t, n = STREET.subn("[ADDRESS]", t)
        self.hits["street_address"] += n
        t, n = FACILITY.subn("[HOSPITAL]", t)
        self.hits["facility"] += n
        return t


# ------------------------------------------------------------
# A capitalised token touching a [NAME] tag is part of that name: "<Firstname> [NAME]",
# "[NAME] Kumari", "Dr. X.Y .[NAME]". Stopwords (honorifics, insurer words,
# common words) are left alone, so "Shri [NAME]" and "[NAME] Life" are kept.
_ABS_TOK = r"(?:[A-Z][a-z]+|[A-Z]{2,})"
_INITIALS = r"(?:[A-Z][ \t]?\.[ \t]?){1,3}"
ABS_BEFORE = re.compile(r"(?<![\w\]])(" + _INITIALS + "|" + _ABS_TOK + r")[ \t]*\.?[ \t]*\[NAME\]")
ABS_AFTER = re.compile(r"\[NAME\][ \t]*((?:[A-Z]\.?[ \t]+){0,3}[A-Z][A-Za-z'’]+|(?![AI]\b)[A-Z](?=[ \t]*\[NAME\]))\b")


def absorb(t):
    if not isinstance(t, str) or "[NAME]" not in t:
        return t

    def before(m):
        tok = m.group(1)
        if re.fullmatch(_INITIALS, tok):
            return "[NAME]"
        low = tok.lower()
        return m.group(0) if low in STOP or low in COMMON else "[NAME]"

    def after(m):
        low = re.split(r"[ \t.]+", m.group(1).strip())[-1].lower()
        return m.group(0) if low in STOP_AFTER or low in COMMON else "[NAME]"

    for _ in range(4):
        new = ABS_AFTER.sub(after, ABS_BEFORE.sub(before, t))
        if new == t:
            break
        t = new
    return re.sub(r"\[NAME\](?:[ \t.]*\[NAME\])+", "[NAME]", t)


# second mentions: scrub every harvested name token elsewhere in the award
# ------------------------------------------------------------

# Common words are learned from the corpus itself: a word is common if it
# is regularly written in lower case. Names almost never are. (A system
# dictionary does not work: /usr/share/dict/words lists "singh", "patel",
# "reddy", "khan", "boral" as lower-case words.)
COMMON = set()
SEEN_LOWER = set()     # looser: used only by the leak check


def build_common(texts, min_lower=5, ratio=0.25):
    lower, title = Counter(), Counter()
    for t in texts:
        lower.update(re.findall(r"\b[a-z]{3,}\b", t))
        title.update(w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", t))
    COMMON.clear()
    # sentence-initial words ("However") are capitalised so often that the
    # ratio alone misses them; anything written in lower case 50+ times is common
    COMMON.update(w for w, n in lower.items()
                  if n >= min_lower and (n >= ratio * title[w] or n >= 50))
    SEEN_LOWER.clear()
    SEEN_LOWER.update(w for w, n in lower.items() if n >= 2)


def name_tokens(names):
    toks = set()
    for n in names:
        for tok in re.findall(r"[A-Za-z]{3,}", PLACEHOLDER.sub(" ", n)):
            low = tok.lower()
            if low in STOP or low in COMMON:
                continue
            toks.add(low)
    return toks


def token_regex(toks):
    if not toks:
        return None
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, toks), key=len, reverse=True)) + r")\b", re.I)


def propagate(t, rx):
    if not isinstance(t, str) or rx is None or not t:
        return t
    t = rx.sub("[NAME]", t)
    return re.sub(r"\[NAME\](?:[ \t.]*\[NAME\])+", "[NAME]", t)


def safe_id(cid):
    """Alphabetic runs of 5+ letters in an id are run-on words, not codes."""
    return re.sub(r"[A-Za-z]{5,}", "~", cid)


# ------------------------------------------------------------
# leak measurement
# ------------------------------------------------------------

CAP_TOKEN = re.compile(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{3,})\b")


def oov_capitals(t):
    """Capitalised tokens that are not dictionary words, stopwords or insurer
    words: an upper bound on surviving names (also catches places, drugs,
    product names)."""
    if not isinstance(t, str):
        return []
    out = []
    for m in CAP_TOKEN.finditer(PLACEHOLDER.sub(" ", t)):
        low = m.group(0).lower()
        if low not in STOP and low not in SEEN_LOWER:
            out.append(m.group(0))
    return out


def harvested_leaks(raw, red, toks):
    """Harvested name tokens still present in redacted text."""
    if not toks or not isinstance(red, str):
        return 0
    return sum(1 for tok in toks if re.search(r"\b" + re.escape(tok) + r"\b", red, re.I))


# ------------------------------------------------------------

def redact(n_samples=15, seed=20261017):
    df = pd.read_parquet(AWARDS)
    build_common(df.text)
    cols = ["text", "ground_span", "ground_span_v1"] + [c for c in df.columns if c.startswith("sec_")]

    sc = Scrubber()
    before_rows = after_rows = before_tok = after_tok = 0
    oov_before, oov_after = Counter(), Counter()
    out = {c: [] for c in cols}
    ids = []

    for i in range(len(df)):
        row = df.iloc[i]

        # harvest from the raw text: v1 names + every v2 name pattern
        harvest = Scrubber()
        harvest.scrub(row.text)
        v1_names = [m.group(2) for m in NAME_LINE.finditer(row.text)] + \
                   [m.group(1) for m in DOCTOR.finditer(row.text)]
        toks = name_tokens(list(harvest.names) + v1_names)
        rx = token_regex(toks)

        v1_text = scrub_v1(row.text)
        for c in cols:
            v = row[c]
            if not isinstance(v, str) or not v:
                out[c].append(v)
                continue
            out[c].append(absorb(propagate(sc.scrub(v1_text if c == "text" else scrub_v1(v)), rx)))
        ids.append(safe_id(propagate(row.award_id, rx)))

        kaggle_text = scrub_kaggle(row.text)
        lb = harvested_leaks(row.text, kaggle_text, toks)
        la = harvested_leaks(row.text, out["text"][-1], toks)
        before_tok += lb; after_tok += la
        before_rows += lb > 0; after_rows += la > 0
        oov_before.update(oov_capitals(kaggle_text))
        oov_after.update(oov_capitals(out["text"][-1]))

    safe = df.copy()
    for c in cols:
        safe[c] = out[c]
    safe["award_id"] = ids
    safe.to_parquet(SAFE, index=False)

    print(f"redacted {len(safe):,} awards, {len(cols)} text columns -> {SAFE}")
    print("\npass-2 replacements by pattern (all columns; sec_* overlap text):")
    for k, v in sc.hits.most_common():
        print(f"  {k:22} {v:>8,}")
    print("\nLEAK CHECK 1 — harvested names surviving (names found in raw text by any pattern,")
    print("searched for anywhere in that award's redacted text):")
    print(f"  Kaggle scrub only      {before_rows:>6,} awards leak   {before_tok:>7,} name tokens")
    print(f"  + added patterns       {after_rows:>6,} awards leak   {after_tok:>7,} name tokens")
    print("\nLEAK CHECK 2 — capitalised non-dictionary tokens left in text (upper bound;")
    print("includes places, drugs, plan names):")
    print(f"  Kaggle scrub only      {sum(oov_before.values()):>8,} tokens  {len(oov_before):>6,} distinct")
    print(f"  + added patterns       {sum(oov_after.values()):>8,} tokens  {len(oov_after):>6,} distinct")
    print("  most frequent survivors:", ", ".join(w for w, _ in oov_after.most_common(60)))

    # gate 2 — print redacted passages in full for a human to read
    rng = random.Random(seed)
    idx = rng.sample(range(len(safe)), n_samples)
    with open(REVIEW, "w", encoding="utf-8") as f:
        for k, j in enumerate(idx, 1):
            r = safe.iloc[j]
            block = (f"\n{'#' * 78}\n# SAMPLE {k}/{n_samples}   award_id={r.award_id}   "
                     f"source={r.source_pdf}   {r.n_chars:,} chars\n{'#' * 78}\n{r.text}\n")
            f.write(block)
            print(block)
    print(f"\n{n_samples} random redacted awards (seed {seed}) printed above and written to {REVIEW}")
    return safe


if __name__ == "__main__":
    redact()
