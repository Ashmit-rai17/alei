"""Stage 4 — segment: cut each compilation into individual awards.

Reads data/inventory.parquet and the PDFs it marks readable (has_text — the
readable front of truncated books included). Writes data/awards.parquet.

awards.parquet is RAW: real names, policy numbers, medical histories.
It never leaves this machine.
"""

import datetime as dt
import os
import re
import sys

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from inventory import DATA, INVENTORY, PDF_DIR, YEAR_MAX, YEAR_MIN, page_text  # noqa: E402

AWARDS = os.path.join(DATA, "awards.parquet")
MAX_CHARS = 20_000          # validation gate: longer = several awards welded together


# ------------------------------------------------------------
# award boundaries — three header formats, all needed
# ------------------------------------------------------------

# BHU-L-029-1718-0123. Broadened (a superset of the Kaggle pattern): extraction
# breaks digit groups — "BNG\n- L - 029\n- 1819\n- 0 202", "CHN- L - 0 29- 1 819- 036"
_D = r"\d(?:[ \t]?\d)"
# Serial is 1-5 digits ("CHD- L - 041- 1718- 1"); to stay strict, the centre
# must be a real Ombudsman centre code and the year a financial year (1718,
# 2122: second pair = first + 1) — checked in valid_complaint_id().
CENTRES = {"AHD", "BNG", "BHP", "BPL", "BHU", "CHD", "CHN", "DEL", "GUW", "HYD", "JPR", "KOC", "ERN",
           "KOL", "LCK", "MUM", "NOI", "PAT", "PUN", "PUNE", "THN"}
COMPLAINT_ID = re.compile(
    r"\b([A-Z]{2,4})\s*[-/]\s*([LGH])\s*[-/]\s*(" + _D + r"{1,3})\s*[-/]\s*(" + _D + r"{3})\s*[-/]\s*(" +
    _D + r"{0,4})\b")


def valid_complaint_id(m):
    fy = re.sub(r"\s", "", m.group(4))
    return m.group(1) in CENTRES and (int(fy[2:]) - int(fy[:2])) % 100 == 1

# Case No.LI/HDFC/158/13 (the Kaggle form, anywhere on a line), or — broadened
# for the other centres — any line-start "Case No" whose id contains a digit:
#   Case No. 21 - 001 - 0180          spaces around separators
#   Case No. LIC / 420 / Chandigarh / word segments
#   Case No: LI/BIRLA/609/19          colon
#   CASE NO. Reliance/489/Mumbai/...  upper case;  Case No. : IO (CHN)/21.03.2279
# The full-line form is tried first so a header yields its whole id
# ("OIO/BBSR/21-004-0208", not the Kaggle branch's "OIO/BBSR/21-004").
# separators seen: ".", ":", ":-", ";" and the id on a following line ("No. \n;Guw-L-006-...")
_LOOSE_ID = (r"(?:[ \t]*\n){0,5}[ \t]*[:;\-.]{0,2}[ \t]*(?=[^\n]{0,40}\d)"
             r"([0-9A-Z(][0-9A-Za-z/\-\.():, &]{3,40})")
COMPLAINT_CASE = re.compile(
    # line start, or after a column gap ("Order No. BPL/LI/13-14/      Case No. LIC/37-21/05-10/BPL")
    r"(?im)(?:^[ \t]*|[ \t]{3,})Case[ \t]*No\.?" + _LOOSE_ID +
    r"|Case\s*No\.?\s*[:\-]?\s*([A-Z]{1,4}[/\-][A-Za-z]{2,12}[/\-]\d{1,5}[/\-]\d{2,4})")

# Complaint No. 370/21/001/L/08/09-10.
# (broadened the same way: a line-start "Complaint (Ref) No" with a digit in
# the id, e.g. "COMPLAINT No. I.O.(HYD) L-21-009-456/2012-13", "IO (CHN) / L 029/ ...")
COMPLAINT_ALT = re.compile(
    r"(?im)^[ \t]*Complaint[ \t]*(?:Ref\.?[ \t]*:?[ \t]*)?No\b\.?" + _LOOSE_ID +
    r"|Complaint\s+(?:Ref\.?\s+)?No\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-\. ]{6,40})")

# Compilations stitch together sections from different Ombudsman centres,
# each with its own header style, so boundaries are the union of all patterns.
# Chennai 2013-14 sections head each award only with its award number:
# "AWARD  No: IO (CHN) L-029 /2013-14   Dated 16/07/2013"
AWARD_NO = re.compile(
    r"(?im)^[ \t]*AWARD\s*(?:No\.?|NO)?\s*[:.\-]?\s*(I\.?\s*O\.?\s*[(/](?:[^\n]|\n(?!\s*\n)){3,30}?\d{2,4}(?:[-/]\d{2,4})?)")

# Chandigarh / Delhi synopses: a bare id on the line after "<City> Ombudsman Centre"
#   "Chandigarh Ombudsman Centre\nAviva/048/Gurgaon/Ludhiana/22/09\n<Complainant Name> Vs ..."
CENTRE_ID = re.compile(
    r"(?im)Ombudsman[ \t]+Cent(?:re|er)[ \t]*\n[ \t]*(?=[^\n]{0,50}/)(?=[^\n]{0,50}\d)([A-Za-z0-9][^\n]{4,50}?)[ \t]*$")
# Bhopal synopses: "SL.No.2.\n\nOFFICE OF THE INSURANCE OMBUDSMAN\n\nSmt. X  V/S ..."
# Serials restart in every section, so the id gets the text offset appended.
SL_NO = re.compile(r"(?im)^[ \t]*SL\.?[ \t]*No\.?[ \t]*(\d{1,3})\b(?=[\s\S]{0,200}?Ombudsman)")

# priority order: when several patterns hit one header, the id comes from the first
PATTERNS = [
    ("primary", COMPLAINT_ID, 0),
    ("case", COMPLAINT_CASE, 1),
    ("alt", COMPLAINT_ALT, 1),
    ("award_no", AWARD_NO, 1),
    ("centre_id", CENTRE_ID, 1),
    ("sl_no", SL_NO, 1),
]
JOINT_STEP = 80         # consecutive hits this close extend a header (joint awards)
HEADER_GAP = 200        # hits this close together are one header block
REPEAT_WINDOW = 30_000  # an id recurring within this distance is a page header, not a new award


# ------------------------------------------------------------
# insurer — fixed list, not regex-guessing
# ------------------------------------------------------------

INSURERS = [
    "Star Health", "Care Health", "Niva Bupa", "Max Bupa", "HDFC ERGO",
    "ICICI Lombard", "Bajaj Allianz Life", "Bajaj Allianz", "Reliance General",
    "TATA AIG", "SBI General", "Cholamandalam", "Future Generali India Life",
    "Future Generali", "IFFCO Tokio", "Universal Sompo", "Liberty General",
    "Go Digit", "ManipalCigna", "Aditya Birla Health", "Raheja QBE",
    "Kotak Mahindra General", "New India Assurance", "United India Insurance",
    "Oriental Insurance", "National Insurance", "Life Insurance Corporation",
    "LIC of India", "HDFC Standard Life", "HDFC Life", "ICICI Prudential",
    "SBI Life", "Max Life", "Kotak Mahindra Life", "Kotak Life",
    "Aditya Birla Sun Life", "TATA AIA", "PNB MetLife", "Exide Life",
    "Canara HSBC", "Bharti AXA", "Aviva Life", "Ageas Federal", "IDBI Federal",
    "Edelweiss Tokio", "Shriram Life", "Pramerica", "IndiaFirst",
    "Reliance Nippon", "Sahara India Life", "Star Union Dai-ichi", "LIC",
]
INSURER_RE = re.compile(r"\b(" + "|".join(re.escape(i) for i in INSURERS) + r")\b", re.I)


# ------------------------------------------------------------
# sections — the Kaggle keyword list, but a heading must stand at the start
# of a line (optionally numbered, with nothing after it but a colon/dash) or
# be followed by a colon. The bare word AWARD runs through every page header
# ("AWARD NO: ...", "Award dated ...") and made has_decision ~100%.
# ------------------------------------------------------------

_SECTION_KW = (
    r"BRIEF FACTS(?: OF THE CASE)?|FACTS OF THE CASE|FACTS|"
    r"CONTENTION[S]? OF THE COMPLAINANT|CONTENTION[S]? OF THE RESPONDENT|"
    r"SUBMISSION[S]? OF THE COMPLAINANT|SUBMISSION[S]? OF THE RESPONDENT|"
    r"OBSERVATION[S]?(?:\s+(?:AND|&)\s+CONCLUSION[S]?)?|FINDINGS|"
    r"REASON[S]? FOR (?:THE )?DECISION|REASONS?|DECISION|AWARD|CONCLUSION"
)
SECTION = re.compile(
    r"(?:^[ \t]*(?:\d{1,2}\s*[.)][ \t]*)?(" + _SECTION_KW + r")[ \t]*[:\-–]?[ \t]*$"
    r"|\b(" + _SECTION_KW + r")[ \t]*:)",
    re.I | re.M,
)
DECISION_KEYS = ("DECISION", "AWARD", "CONCLUSION", "REASON FOR DECISION", "REASONS FOR DECISION",
                 "REASON FOR THE DECISION", "REASONS FOR THE DECISION",
                 "OBSERVATION AND CONCLUSION", "OBSERVATIONS AND CONCLUSIONS",
                 "OBSERVATION AND CONCLUSIONS", "OBSERVATIONS AND CONCLUSION")


# ------------------------------------------------------------
# dates
# ------------------------------------------------------------

AWARD_DT = re.compile(r"AWARD\s*(?:NO[^\n]{0,30})?DATED[\s:–\-]*([^\n]{5,40})", re.I)
DATE_NUM = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
DATE_ORD = re.compile(r"\b(\d{1,2})\s*(?:ST|ND|RD|TH)?\s+"
                      r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+(\d{4})\b", re.I)
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

# The 2008-era synopses put "Award dated ..." on the line *before* the case
# number, so it belongs to the award that follows it: look just behind the
# boundary first, and ignore a marker in the last few hundred chars of a body
# (that one is the next award's header).
PRE_CONTEXT = 150
TAIL_GUARD = 300


def _valid(d, m, y):
    if not YEAR_MIN <= y <= YEAR_MAX:
        return None
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def first_date(s):
    """Earliest valid dd.mm.yyyy or ordinal date in s, as ISO."""
    found = []
    for m in DATE_NUM.finditer(s):
        iso = _valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            found.append((m.start(), iso))
    for m in DATE_ORD.finditer(s):
        iso = _valid(int(m.group(1)), MONTHS[m.group(2).upper()[:3]], int(m.group(3)))
        if iso:
            found.append((m.start(), iso))
    return min(found)[1] if found else ""


# Task 4: the form field wins, then inline award-date markers, else nothing.
# There is no first-date fallback: it returned policy dates (8.4% agreement
# with the form field in v1), and an empty date is better than a wrong one.
DATE_FIELD = re.compile(r"Date\s+of\s+(?:the\s+)?Award\s*/\s*Order[\s:\-]*", re.I)
INLINE_MARKERS = [
    AWARD_DT,                                                                     # Award dated / AWARD NO ... DATED
    re.compile(r"AWARD\s*NO[^\n]{0,40}?\bdt\.?[\s:\-]*([^\n]{5,40})", re.I),            # Award no-IO(CHN)/... dt 28thMarch 2011
    re.compile(r"Date\s+of\s+(?:the\s+)?Award(?!\s*/\s*Order)[\s.…:\-]*([^\n]{5,40})", re.I),   # Date of Award ….. 26.12.2012
    re.compile(r"Award\s+passed\s+on[\s:\-]*([^\n]{5,40})", re.I),                     # Award Passed on 31.03.2015
]


def _ordinal_glued(s):
    """'28thMarch 2011' -> '28th March 2011' so DATE_ORD can read it."""
    return re.sub(r"(?i)(\d)(st|nd|rd|th)(?=[A-Z])", r"\1\2 ", s)


def field_date(body):
    m = DATE_FIELD.search(body[:4000])
    return first_date(_ordinal_glued(body[m.end():m.end() + 60])) if m else ""


def inline_date(pre, body):
    for rx in INLINE_MARKERS:
        for m in rx.finditer(pre):
            iso = first_date(_ordinal_glued(m.group(1)))
            if iso:
                return iso, "marker_before"
    found = []
    for rx in INLINE_MARKERS:
        for m in rx.finditer(body[:2500]):
            if m.start() > len(body) - TAIL_GUARD:      # the next award's header
                break
            iso = first_date(_ordinal_glued(m.group(1)))
            if iso:
                found.append((m.start(), iso))
                break
    return (min(found)[1], "marker") if found else ("", "")


def award_date(pre, body):
    f = field_date(body)
    if f:
        return f, "field"
    return inline_date(pre, body)


# ------------------------------------------------------------
# ground_span — the sentence stating why the claim was repudiated
# ------------------------------------------------------------

GROUND_V1 = re.compile(
    r"([^.]{0,200}\brepudiat\w*[^.]{0,400}?\b(?:on the ground|on the grounds?|"
    r"stating that|on account of)\b[^.]{0,400}\.)", re.I)

# Task 3: a period does not end the sentence when a digit follows (06.05.2013,
# Rs.1,00,000) or when it closes a known abbreviation. Those periods are
# swapped 1:1 for a placeholder in a copy of the text, GROUND_V1 runs on the
# copy, and the span is cut from the original (a per-character alternation
# inside the pattern backtracked badly).
_ABBR = ["O", "O.P", "No", "Nos", "Rs", "Ltd", "Co", "Smt", "Shri", "Sri", "Sh", "Mr", "Mrs", "Ms", "Dr",
         "viz", "i.e", "e.g", "Pvt", "Ins", "Corpn", "St"]
_SOFT_DOT = re.compile(r"\.(?=\s?\d)|" + "|".join(rf"(?<=\b{re.escape(a)})\." for a in _ABBR), re.I)
GROUND = GROUND_V1


def _mask_soft_dots(body):
    return _SOFT_DOT.sub("\u2024", body)       # ONE DOT LEADER, same length


def ground_span(body, rx=None):
    if rx is GROUND_V1:
        m = GROUND_V1.search(body)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    m = GROUND.search(_mask_soft_dots(body))
    return re.sub(r"\s+", " ", body[m.start(1):m.end(1)]).strip() if m else ""


# is_repudiation: the award is about a rejected claim. Form labels ("Date of
# Repudiation", "Reason for repudiation") appear in every modern award even
# when the answer is NA, so they are removed before the text test and the
# Date-of-Repudiation field counts only with a real (non-NA) value.
_REP_LABELS = re.compile(r"(?i)(?:date\s+of|reason\s+for)\s+repudiation(?:\s*/\s*rejection)?")
_REP_TEXT = re.compile(r"(?i)\brepudiat\w*|\bclaim\s+was\s+rejected\b|\brejection\s+of\s+(?:the\s+)?claim\b")
_REP_FIELD = re.compile(r"(?i)date\s+of\s+repudiation(?:\s*/\s*rejection)?[ \t:\-]*\n?\s*([^\n]*)")


def is_repudiation(body):
    if _REP_TEXT.search(_REP_LABELS.sub(" ", body)):
        return True
    m = _REP_FIELD.search(body)
    if m:
        v = m.group(1).strip()          # first non-empty line after the label
        return (bool(re.search(r"\d", v))
                and not re.fullmatch(r"(?i)(?:na|n\.?a\.?|nil|not applicable|-+)", v))
    return False


# ------------------------------------------------------------

# Diagnostic only (does not move boundaries): distinct case/complaint numbers
# heading lines inside one segment. >= 2 means several awards probably welded
# together because the book switches header format between centres.
# Weld diagnostic. A header label at line start ("Case No", "Complaint (Ref.) No",
# "REF: NO"), then the id on the same line or the next few (books break ids
# across lines: "CASE NO\n- CHD\n- L - 041\n- 1718\n- 1"). The id is read with
# the splitter's own normalisation (dash map, COMPLAINT_ID, id_key). Numbers
# in the first JOINT_SPAN chars are one header (joint awards).
HEADER_LABEL = re.compile(
    r"(?im)^[ \t]*(?:\d{1,2}[.)][ \t]*)?(?:(?:Case|Complaint)[ \t]*(?:Ref\.?[ \t]*:?[ \t]*)?No\b|REF\.?[ \t]*:?[ \t]*NO\b)")
JOINT_SPAN = 300


def header_ids(body):
    """[(pos, key)] for every header label in body whose id can be read."""
    body = body.translate(DASHES)
    out = []
    for m in HEADER_LABEL.finditer(body):
        if is_prose(body, m.start()):
            continue
        window = body[m.end():m.end() + 90]
        idm = COMPLAINT_ID.search(window)
        if idm and idm.start() < 15 and valid_complaint_id(idm):
            cid = idm.group(0)
        else:
            lm = re.match(r"[\s.:;\-]*([0-9A-Z(][0-9A-Za-z/\-.():, &]{3,40})", window)
            cid = lm.group(1) if lm and len(re.findall(r"\d", lm.group(1))) >= 3 else ""
        if cid:
            out.append((m.start(), id_key(re.sub(r"\s+", "", cid).upper())))
    # a bare complaint id opening a line ("NOI-L-019-2122-1072" — modern headers have no label)
    for m in COMPLAINT_ID.finditer(body):
        line_start = body.rfind("\n", 0, m.start()) + 1
        if body[line_start:m.start()].strip() == "" and valid_complaint_id(m):
            out.append((m.start(), id_key(re.sub(r"\s+", "", m.group(0)).upper())))
    return sorted(out)


def n_headers(body, own_id=None):
    """Distinct award headers in a segment. Ids chained within JOINT_STEP of each
    other are one header (a joint award's list of numbers). The first header —
    plus the segment's own id, which often has no "Case No" label — is the
    award itself; each later header that brings a new number counts once."""
    groups = []
    for p, k in header_ids(body):
        if groups and (p - groups[-1][-1][0] <= JOINT_STEP or p < JOINT_SPAN):
            groups[-1].append((p, k))
        else:
            groups.append([(p, k)])
    seen = {k for _, k in groups[0]} if groups else set()
    if own_id:
        seen.add(id_key(re.sub(r"\s+", "", own_id.translate(DASHES)).upper()))
    n = 1 if seen else 0
    for g in groups[1:]:
        new = {k for _, k in g} - seen
        if new:
            n += 1
            seen |= new
    return n


# en/em dashes as separators ("L - 029 -1415 – 0273") — matched as "-";
# a 1:1 replacement, so positions in the original text are unchanged
# (U+0374 GREEK NUMERAL SIGN and U+2015 HORIZONTAL BAR are used as dashes in the Karnataka/Tamil Nadu books)
DASHES = str.maketrans({"–": "-", "—": "-", "‐": "-", "‑": "-", "―": "-", "\u0374": "-", "\u02b9": "-"})


_PROSE = re.compile(r"\b[a-z]{2,}[ \t]+[a-z]{2,}[ \t]+[a-z]{2,}\b")


def is_prose(text, start):
    """A 'case no.' / 'complaint no.' that runs on into ordinary lower-case words
    ("case no.121/2011 on the same subject", "complaint no. 188/11 which is
    pending") is a reference inside an award, not a header. Headers end after
    the id or a capitalised title ("Complaint No-21-001-1289  Death Claim")."""
    end = text.find("\n", start)
    line = text[start:end if end != -1 else len(text)]
    line = re.sub(r"(?i)^\s*(?:case|complaint)\s*(?:ref\.?\s*:?\s*)?no\b", "", line)
    return bool(_PROSE.search(line))


def id_key(cid):
    """Comparable form of an id: run-on words stripped ("21-001-1289DEATHCLAIM"),
    unless that leaves too little to identify it."""
    full = re.sub(r"\s+", "", cid).upper()
    k = re.sub(r"[A-Z]{4,}.*$", "", full).strip(".-/")
    return k if len(re.findall(r"\d", k)) >= 3 else full


def boundaries(text):
    """All patterns at once -> [(start, id, pattern)].

    Hits within HEADER_GAP of the first hit of a header form one header
    (complaint number + award number + case number of the same award).
    """
    hits = sorted(
        (m.start(), pri, ("SL.NO." if name == "sl_no" else "") + re.sub(r"\s+", "", m.group(0) if grp == 0 else
                                next(g for g in m.groups() if g)).upper()
         + (f"@{m.start()}" if name == "sl_no" else ""), name)
        for pri, (name, rx, grp) in enumerate(PATTERNS)
        for m in rx.finditer(text.translate(DASHES))
        # verbatim COMPLAINT_ALT also matches "Nature of Complaint\nNon payment"
        if (name != "alt" or re.match(r"(?i)\s*Complaint\s*(?:Ref\.?\s*:?\s*)?No\b", m.group(0)))
        and not is_prose(text, m.start())
        and (name != "primary" or valid_complaint_id(m)))
    clusters = []
    for h in hits:
        # measured from the cluster's first hit, not the previous hit: chaining
        # glued short synopses (Case No -> Award dated -> next Case No) together
        # ...but a joint award lists several numbers ~45 chars apart, so a hit
        # right after the previous one (JOINT_STEP) still belongs to the header
        if clusters and (h[0] - clusters[-1][0][0] <= HEADER_GAP or h[0] - clusters[-1][-1][0] <= JOINT_STEP):
            clusters[-1].append(h)
        else:
            clusters.append([h])

    # A repeated id is a page header only while it belongs to the award being
    # read. Once another award has started, the same id again is a reprint
    # (books print some awards twice) and starts its own segment; the
    # keep-longest dedupe in segment() then collapses the copies.
    current, last_pos, out = set(), -REPEAT_WINDOW - 1, []
    for c in clusters:
        in_window = c[0][0] - last_pos <= REPEAT_WINDOW
        # hits that are not the current award's own number; a cluster can hold
        # the current award's footer followed by the next award's header
        # within one header, a truncated copy of another hit's id ("BHU-L-041-2324-"
        # cut at a line break, beside the full "BHU-L-041-2324-0006") is the same id
        # keep the higher-priority pattern's id of any prefix-related pair
        def shadowed(h):
            k = id_key(h[2])
            return any(o is not h and (id_key(o[2]).startswith(k) or k.startswith(id_key(o[2])))
                       and id_key(o[2]) != k and o[1] < h[1] for o in c)
        c = [h for h in c if not shadowed(h)]
        fresh = [h for h in c if not (in_window and id_key(h[2]) in current)]
        if not fresh:
            last_pos = c[0][0]
            continue
        best = min(fresh, key=lambda h: h[1])
        out.append((fresh[0][0], best[2], best[3]))
        current, last_pos = {id_key(h[2]) for h in fresh}, fresh[0][0]
    return out


# Header blocks that precede the case number. Without these, an award's tail
# carried the next award's "AWARD IN THE MATTER OF ..." / "PROCEEDINGS OF THE
# INSURANCE OMBUDSMAN, <city>" lines. Used only to move a boundary back, and
# only when a party line follows within PARTY_WITHIN chars.
def _spaced(word):
    """'BEFORE' -> B\\s*E\\s*F... : extraction breaks words anywhere ("B\\nEF\\nORE")."""
    return r"\s*".join(re.escape(ch) for ch in word)


BLOCK_START = re.compile(
    r"(?im)^[ \t]*(?:" + _spaced("AWARD") + r"\s*" + _spaced("IN") + r"\s*" + _spaced("THE") + r"\s*M\s*A\s*T+\s*E\s*R\s*"
    + _spaced("OF") + r"|P\s*R\s*O\s*C\s*E\s*E?\s*D\s*I\s*N\s*G\s*S?\s*(?:" + _spaced("OF") + "|" + _spaced("BEFORE")
    + r")\s*(?:" + _spaced("THE") + r"\s*)?" + _spaced("INSURANCE") + r"\s*" + _spaced("OMBUDSMAN") + ")")
PARTY = re.compile(r"(?i)(?:^|[\s.])(?:vs\.?|v/s\.?|vrs\.?)(?=\s)")
PARTY_WITHIN = 400
LOOKBACK = 1500
MIN_KEEP = 300          # never pull back into the previous award's own header


def pull_back(text, bounds):
    """Move each boundary back to a header-block start sitting just before it."""
    out, moved = [], 0
    for i, (pos, cid, name) in enumerate(bounds):
        floor = (out[-1][0] if out else 0) + MIN_KEEP
        best = None
        for m in BLOCK_START.finditer(text, max(floor, pos - LOOKBACK), pos):
            if PARTY.search(text, m.end(), m.end() + PARTY_WITHIN):
                best = m.start()
                break                       # earliest qualifying block start
        if best is not None and best < pos:
            pos, moved = best, moved + 1
        out.append((pos, cid, name))
    return out, moved


def tail_header(body, tail=800):
    """The segment ends with the next award's header block (bleed)."""
    for m in BLOCK_START.finditer(body, max(0, len(body) - tail)):
        if PARTY.search(body, m.end(), m.end() + PARTY_WITHIN):
            return True
    return False


def split_awards(text):
    """Cut at the union of all patterns' headers -> ([(start, id, body, pattern)], counts)."""
    bounds, _ = pull_back(text, boundaries(text))
    out = []
    for i, (pos, cid, name) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        body = text[pos:end].strip()
        if len(body) >= 400:              # drop table-of-contents entries
            out.append((pos, cid, body, name))
    counts = {name: sum(1 for *_, n in out if n == name) for name, _, _ in PATTERNS}
    return out, counts


def sections(body):
    marks = [(m.start(), re.sub(r"\s+", " ", (m.group(1) or m.group(2))).upper().replace("&", "AND"))
             for m in SECTION.finditer(body)]
    d = {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        d.setdefault(name, body[pos:end].strip())
    return d


def length_report(lengths, label):
    s = pd.Series(lengths)
    q = s.quantile([0, .1, .25, .5, .75, .9, .99, 1]).round().astype(int)
    print(f"\n{label}: n={len(s):,}  " + "  ".join(f"p{int(k*100)}={v:,}" for k, v in q.items()))
    print(f"  over {MAX_CHARS:,} chars: {(s > MAX_CHARS).sum()}")


def segment():
    inv = pd.read_parquet(INVENTORY)
    usable = inv[inv.has_text & ~inv.broken]
    print(f"segmenting {len(usable)} pdfs (truncated: {usable.truncated.sum()} — front portion only)\n")

    rows, failed, per_pdf = [], [], []
    for _, f in usable.iterrows():
        text = page_text(os.path.join(PDF_DIR, f.file))
        awards, counts = split_awards(text)
        per_pdf.append(dict(file=f.file, claim_type=f.claim_type, truncated=f.truncated,
                            n=len(awards), **{f"n_{k}": v for k, v in counts.items()}))
        if not awards:
            failed.append(f.file)
            continue

        for pos, cid, body, pattern in awards:
            secs = sections(body)
            ins = INSURER_RE.search(body)
            pre = text[max(0, pos - PRE_CONTEXT):pos]
            date, date_src = award_date(pre, body)
            rows.append(dict(
                award_id=cid,
                source_pdf=f.file,
                claim_type=f.claim_type,
                insurer=ins.group(1) if ins else "",
                award_date=date,
                date_source=date_src,
                date_field=field_date(body),
                date_inline=inline_date(pre, body)[0],
                is_repudiation=is_repudiation(body),
                ground_span=ground_span(body),
                ground_span_v1=ground_span(body, GROUND_V1),
                n_chars=len(body),
                n_sections=len(secs),
                n_headers=n_headers(body, cid),
                tail_header=tail_header(body),
                has_decision=any(k in secs for k in DECISION_KEYS),
                pattern=pattern,
                text=body,
                **{"sec_" + re.sub(r"\W+", "_", k.lower()).strip("_"): v for k, v in secs.items()},
            ))

    raw = pd.DataFrame(rows)
    length_report(raw.n_chars, "segment lengths BEFORE dedupe")

    # duplicate (award_id, source_pdf): keep the LONGEST — short duplicates
    # are split failures that drop the back half of an award
    df = (raw.sort_values("n_chars", ascending=False)
             .drop_duplicates(subset=["award_id", "source_pdf"], keep="first")
             .sort_values(["source_pdf", "award_id"])
             .reset_index(drop=True))
    sec_cols = [c for c in df.columns if c.startswith("sec_")]
    df[sec_cols] = df[sec_cols].fillna("")
    df.to_parquet(AWARDS, index=False)
    length_report(df.n_chars, "segment lengths AFTER dedupe (written)")

    pp = pd.DataFrame(per_pdf)
    pp.to_csv(os.path.join(DATA, "segment_per_pdf.csv"), index=False)
    kept = df.groupby("source_pdf").size()
    pp["kept"] = pp.file.map(kept).fillna(0).astype(int)
    v1 = os.path.join(DATA, "v1_shipped", "awards.parquet")
    if os.path.exists(v1):
        pp["n_v1"] = pp.file.map(pd.read_parquet(v1, columns=["source_pdf"]).groupby("source_pdf").size()
                                 ).fillna(0).astype(int)
    else:
        pp["n_v1"] = 0

    has = lambda c: f"{(df[c] != '').sum():,}  ({(df[c] != '').mean():.1%})"
    print(f"\n{'='*64}")
    print(f"  pdfs segmented      {len(usable)}   matched: {len(usable) - len(failed)}   no pattern: {len(failed)}")
    print(f"  awards extracted    {len(df):,}   (dropped as shorter duplicates: {len(raw) - len(df):,})")
    print(f"  insurer identified  {has('insurer')}")
    print(f"  award_date found    {has('award_date')}   by source: {df.date_source.value_counts().to_dict()}")
    print(f"  ground_span found   {has('ground_span')}")
    print(f"  has_decision        {df.has_decision.sum():,}  ({df.has_decision.mean():.1%})")
    print(f"  median length       {df.n_chars.median():,.0f} chars")
    print(f"  over {MAX_CHARS:,} chars    {(df.n_chars > MAX_CHARS).sum()}")
    print(f"  weld suspects       {(df.n_headers >= 2).sum():,}  (>= 2 distinct case/complaint numbers in one segment)")
    print(f"  tail bleed          {df.tail_header.sum():,}  (segment ends with the next award's header block)")
    yrs = pd.to_datetime(df.award_date[df.award_date != ""]).dt.year
    print(f"  award_date years    {yrs.min()} - {yrs.max()}")
    print(f"  pattern used        {df.pattern.value_counts().to_dict()}")
    print(f"{'='*64}")
    print("\nawards per claim_type:")
    print(df.claim_type.value_counts().to_string())
    print("\nper source pdf  (kept = after dedupe; n_v1 = v1 single-pattern rule; n_* = boundaries by pattern):")
    print(pp[["file", "claim_type", "truncated", "n_v1", "kept"] + [f"n_{n}" for n, _, _ in PATTERNS]]
          .to_string(index=False))
    if failed:
        print("\nno pattern matched:")
        for name in failed:
            print(f"  {name}")
    over = df[df.n_chars > MAX_CHARS]
    if len(over):
        print(f"\nOVER {MAX_CHARS:,} CHARS:")
        print(over[["source_pdf", "award_id", "n_chars", "pattern"]].to_string(index=False))
    return df


if __name__ == "__main__":
    segment()
