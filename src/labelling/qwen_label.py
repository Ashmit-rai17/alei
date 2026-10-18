"""Task A — label repudiation awards against TAXONOMY.md with Qwen.

Reads data/awards_safe.parquet only — redacted text, never raw text. Redaction is measured, not complete:
residual identifiers remained in 22 of 64 awards audited on the earlier v1 redaction — an upper bound only
for the audited text; the improved redaction sent to the API has not been measured (docs/LIMITATIONS.md).

  QWEN_API_KEY, QWEN_API_KEY_2, QWEN_API_KEY_3, …   one or more keys (in .env); used in order,
                 moving to the next when one hits its daily cap. .env is re-read whenever a key is
                 exhausted, so keys appended to it reach a running job. Key values are never printed.
  QWEN_BASE_URL  OpenAI-compatible endpoint (Groq: https://api.groq.com/openai/v1)
  QWEN_MODEL     Groq: qwen/qwen3.8-27b

  Fallback, documented and unused: local Ollama. Point QWEN_BASE_URL at http://localhost:11434/v1
  and QWEN_MODEL at alei-qwen3-8b (qwen3:8b with num_ctx 24576); the script then switches to
  batches of 3, one worker, no pacing, and its own checkpoint (qwen_labels_local.jsonl).

  python scripts/qwen_label.py --show-prompt     # print the system prompt, call nothing
  python scripts/qwen_label.py                   # label; resumable; checkpoints every batch
  python scripts/qwen_label.py --finalize        # jsonl -> data/labels.parquet + report

Checkpoint: data/qwen_labels.jsonl, one line per award, append-only. A resume skips every
uid already there. Per-batch tokens, wall time and errors go to data/qwen_log.jsonl.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
SAFE = os.path.join(DATA, "awards_safe.parquet")
TAXONOMY = os.path.join(ROOT, "docs", "TAXONOMY.md")
CKPT = os.path.join(DATA, "qwen_labels.jsonl")
LOG = os.path.join(DATA, "qwen_log.jsonl")
OUT = os.path.join(DATA, "labels.parquet")

def load_env(path=os.path.join(ROOT, ".env")):
    """KEY=VALUE lines into os.environ (existing variables win). Values are never printed."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
# local Ollama: nothing leaves the machine, no rate limits; own checkpoint so cloud labels stay intact
LOCAL = bool(re.match(r"https?://(localhost|127\.0\.0\.1)[:/]", BASE_URL))
if LOCAL:
    CKPT = os.path.join(DATA, "qwen_labels_local.jsonl")
    LOG = os.path.join(DATA, "qwen_log_local.jsonl")
BATCH = 10                    # upper bound; the token budget usually binds first
# Groq on_demand: one request may carry at most 7,000 input tokens per minute (ITPM), 8,000 tokens
# per minute in all. Measured 3.7-4.0 chars/token on this corpus; 3.5 is used to stay under.
CHARS_PER_TOKEN = 3.7
REQ_INPUT_TOKENS = 6_600      # estimated input tokens per request; at the lowest measured 3.66 chars/token that is < 6,700 real
TPM_BUDGET = 7_600            # tokens (in + out) sent per rolling minute, across all workers
EST_OUT_PER_AWARD = 120
MAX_AWARD_CHARS = 15_000      # an award longer than this alone is cut (marked) to fit one request
WORKERS = 2
LOCAL_BATCH = 3               # an 8b model starts copying its previous answer on long batches
LOCAL_MAX_BATCH_CHARS = 60_000
# optional request fields; one the endpoint rejects with a 400 is dropped for the rest of the run
OPTIONAL = {"response_format": {"type": "json_object"}, "reasoning_effort": "none"}
if LOCAL:
    WORKERS = 1               # local inference is serial; parallel requests only queue
    # measured: "think": false is ignored on the OpenAI path, /no_think only halves reasoning,
    # reasoning_effort "none" turns it off (10 completion tokens vs 386)
    OPTIONAL = {"response_format": {"type": "json_object"}, "reasoning_effort": "none"}
    MAX_AWARD_CHARS = 40_000  # no per-request cap locally; the largest award is ~24k chars
NO_THINK = os.environ.get("QWEN_NO_THINK", "") == "1"     # append /no_think to the system prompt
STATS = {"http_429": 0, "dropped_params": []}
SAMPLE = os.path.join(DATA, "label_sample.csv")
STOP = threading.Event()


DEADLINE = None               # epoch seconds; no batch starts after it (set by --until, IST)
STOP_REASON = []
LABEL_MODEL = {"qwen/qwen3.8-27b": "qwen3-32b-groq", "alei-qwen3-8b": "qwen3-8b-ollama"}


class DailyCap(RuntimeError):
    """Every key is exhausted: stop the run, do not retry into it."""


class Keys:
    """QWEN_API_KEY, QWEN_API_KEY_2, … from .env (re-read on exhaustion) or the environment.
    Only key NAMES are ever logged or printed."""

    def __init__(self, path=os.path.join(ROOT, ".env")):
        self.path, self.exhausted, self.name = path, {}, None
        self.lock = threading.Lock()
        self.refresh()

    def read(self):
        found = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                m = re.match(r"(QWEN_API_KEY(?:_(\d+))?)\s*=\s*(.+)", line)
                if m:
                    found[m.group(1)] = m.group(3).strip().strip('"').strip("'")
        for k, v in os.environ.items():
            if re.fullmatch(r"QWEN_API_KEY(?:_\d+)?", k):
                found.setdefault(k, v)
        order = lambda k: int(k.rsplit("_", 1)[1]) if k[-1].isdigit() else 1
        return dict(sorted(found.items(), key=lambda kv: order(kv[0])))

    def refresh(self):
        self.keys = self.read()
        if self.name not in self.keys or self.name in self.exhausted:
            self.name = next((k for k in self.keys if k not in self.exhausted), None)

    def current(self):
        with self.lock:
            if self.name is None:
                self.refresh()
            return self.name, self.keys.get(self.name)

    def index(self, name):
        return 1 if name == "QWEN_API_KEY" else int(name.rsplit("_", 1)[1])

    def exhaust(self, name, why):
        """Mark `name` used up; re-read .env; return the next usable key name or None."""
        with self.lock:
            if name not in self.exhausted:
                self.exhausted[name] = why
                print(f"  KEY {self.index(name)} ({name}) exhausted at {time.strftime('%H:%M:%S')}: {why}", flush=True)
                with LOCK:
                    SENT.clear()                  # the next key has its own per-minute budget
            self.refresh()
            if self.name is not None and self.name != name:
                print(f"  switching to KEY {self.index(self.name)} ({self.name}); "
                      f"{len(self.keys)} key(s) in .env, {len(self.exhausted)} exhausted", flush=True)
            return self.name


KEYS = None


def uid(award_id, source_pdf):
    return f"{award_id}|{source_pdf}"


def norm_ws(s):
    return " ".join(str(s).split())


# ---------------------------------------------------------------- taxonomy + prompt

def load_taxonomy(path=TAXONOMY):
    """Return (rows, rules, ids). rows: the markdown table lines whose first cell is an id
    (G1…G12, G99), verbatim. rules: verbatim sections whose heading mentions G1/G7, G1/G2
    or G3 together with vs / rule / decision."""
    if not os.path.exists(path):
        sys.exit(f"missing {path} — put the frozen TAXONOMY.md in the project root")
    md = open(path, encoding="utf-8").read()
    rows, ids = [], []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and re.fullmatch(r"\*{0,2}G\d{1,2}\*{0,2}", cells[0]):
            rows.append(line.strip())
            ids.append(cells[0].strip("*"))
    # the whole "Decision rules" section, verbatim, minus its heading and the trailing rule line
    m = re.search(r"(?ms)^## Decision rules[^\n]*\n(.*?)(?=^---|^## )", md)
    rules = [m.group(1).strip()] if m else []
    real = [i for i in ids if i != "G99"]
    if len(real) != 12:
        sys.exit(f"TAXONOMY.md: expected 12 ground rows (+G99), parsed {len(real)}: {ids}. "
                 "Check the table format before spending calls.")
    need = ["**G1 vs G2", "**G1 vs G7", "**G1/G7 vs G3"]
    if not rules or not all(n in rules[0] for n in need):
        sys.exit(f"TAXONOMY.md: decision rules section missing one of {need}. Check before spending calls.")
    if "G99" not in ids:
        ids.append("G99")
    return rows, rules, ids


def system_prompt(rows, rules):
    return f"""You are labeling Indian Insurance Ombudsman awards.

For each award, identify every ground on which the INSURER repudiated
or rejected the claim. Not the Ombudsman's reasoning. Not the outcome.
Only what the insurer said when it refused.

Use ONLY these category IDs:
{chr(10).join(rows)}

Decision rules (apply exactly):
{(chr(10) + chr(10)).join(rules)}

Rules:
- More than one ground per award is normal. List all of them.
- For each ground, quote the EXACT sentence from the award text that
  supports it. Copy it verbatim. Do not paraphrase, shorten or fix typos.
- If no ground is stated, return an empty grounds list.
- If a real ground fits none of the categories, use G99 and describe it
  in a "description" field alongside the quote.
- Tags such as [NAME] and [POLICY_NO] are deliberate redactions; copy them as they appear.
- Quote the sentence stating the INSURER's own position — from its repudiation letter, its
  written submissions, or a line such as 'the insurer repudiated the claim on the ground that…'.
  If the award states the insurer's ground nowhere else, you may quote the Ombudsman's
  restatement of it, but NEVER a sentence where the Ombudsman is rejecting, doubting or
  overturning the insurer's argument.
- For each ground set "src": "insurer" if the quote is the insurer's own statement (or the award
  reporting what the insurer said), "ombudsman" if it is the Ombudsman's restatement.

Return strict JSON, no prose:
{{"results":[{{"award_id":"...","grounds":[{{"id":"G1","quote":"...","src":"insurer"}}]}}]}}"""


def user_prompt(batch):
    parts = [f"Label these {len(batch)} awards. Use the award_id exactly as given (A1, A2, ...).\n"]
    for k, r in enumerate(batch, 1):
        parts.append(f"=== award_id: A{k} ===\n{r.get('sent', r['text'])}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- API

def call(system, user):
    """Returns (content, usage, key_name)."""
    attempt = 0
    while attempt < 6:
        name, key = KEYS.current()
        if key is None:
            STOP.set()
            raise DailyCap("no usable key left")
        body = {"model": MODEL, "temperature": 0, **OPTIONAL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=1800 if LOCAL else 300,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as e:
            err = f"network: {type(e).__name__}"
        else:
            if r.status_code == 200:
                j = r.json()
                return j["choices"][0]["message"]["content"] or "", j.get("usage", {}), name
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 400:
                bad = [p for p in OPTIONAL if p in r.text]
                if bad:                                   # unsupported option: drop it and resend
                    with LOCK:
                        for p in bad:
                            if OPTIONAL.pop(p, None) is not None:
                                STATS["dropped_params"].append(p)
                    continue
            if r.status_code == 429 and not LOCAL:
                with LOCK:
                    STATS["http_429"] += 1
                wait = float(r.headers.get("retry-after", 0) or 0)
                low = r.text.lower()
                if "per day" in low or "(tpd)" in low or "(rpd)" in low or "quota" in low or wait > 600:
                    m = re.search(r"\((T|R)P[DM]\): Limit (\d+), Used (\d+)", r.text)
                    why = f"daily cap{(' ' + m.group(0)) if m else ''}, retry-after {wait:.0f}s"
                    if KEYS.exhaust(name, why) is None:
                        STOP.set()
                        raise DailyCap(f"all keys exhausted; last: {why}")
                    continue                              # same request, next key
                time.sleep(max(wait, 2 ** attempt * 3))
                attempt += 1
                continue
            if r.status_code in (401, 403) and not LOCAL:      # revoked / invalid key: treat as exhausted
                if KEYS.exhaust(name, f"HTTP {r.status_code} (invalid or revoked key)") is None:
                    STOP.set()
                    raise DailyCap("all keys exhausted or invalid")
                continue
            if r.status_code not in (500, 502, 503, 504):
                raise RuntimeError(err)
        time.sleep(2 ** attempt * 3)
        attempt += 1
    raise RuntimeError(err)


def parse(content, n):
    s = re.sub(r"(?s)<think>.*?</think>", "", content).strip()      # reasoning blocks, if any
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    if not s.startswith("{"):                                        # prose around the JSON
        a, b = s.find("{"), s.rfind("}")
        s = s[a:b + 1] if a >= 0 < b else s
    obj = json.loads(s)
    out = {}
    for res in obj["results"]:
        m = re.fullmatch(r"A(\d+)", str(res.get("award_id", "")).strip())
        if not m or not 1 <= int(m.group(1)) <= n:
            continue
        out[int(m.group(1))] = [g for g in res.get("grounds", []) if isinstance(g, dict)]
    return out


# ---------------------------------------------------------------- run

def load_pool():
    """The awards in data/label_sample.csv (scripts/sample_for_labeling.py), from the safe file."""
    assert os.path.basename(SAFE) == "awards_safe.parquet"
    df = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "is_repudiation", "text"])
    want = set(pd.read_csv(SAMPLE).uid)
    df = df[(df.award_id + "|" + df.source_pdf).isin(want)].reset_index(drop=True)
    assert len(df) == len(want), f"{len(want) - len(df)} sample uids not in awards_safe.parquet"
    tagged = df.text.str.contains(r"\[(?:NAME|POLICY_NO|HOSPITAL|NUM)\]").mean()
    assert tagged > 0.5, f"only {tagged:.0%} rows carry redaction tags — is this really the safe file?"
    df["uid"] = [uid(a, s) for a, s in zip(df.award_id, df.source_pdf)]
    assert df.uid.is_unique
    return df


def done_uids():
    if not os.path.exists(CKPT):
        return set()
    with open(CKPT, encoding="utf-8") as f:
        return {json.loads(l)["uid"] for l in f if l.strip()}


def make_batches(rows, system_chars):
    budget = LOCAL_MAX_BATCH_CHARS if LOCAL else int(REQ_INPUT_TOKENS * CHARS_PER_TOKEN) - system_chars - 200
    cap = LOCAL_BATCH if LOCAL else BATCH
    batches, cur, size = [], [], 0
    for r in rows:
        r["sent"] = prep(r["text"])
        r["text_len"] = len(r["sent"]) + 40
    for r in rows:
        if cur and (len(cur) == cap or size + r["text_len"] > budget):
            batches.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += r["text_len"]
    if cur:
        batches.append(cur)
    return batches


LOCK = threading.Lock()
SENT = []                     # (time, tokens) for the rolling-minute budget


def prep(text):
    """Text as sent: whitespace runs collapsed (verification ignores whitespace); over-long cut."""
    t = re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text))
    return t if len(t) <= MAX_AWARD_CHARS else t[:MAX_AWARD_CHARS] + "\n[... award text cut here to fit the request limit ...]"


def est_tokens(chars):
    return int(chars / CHARS_PER_TOKEN)


def pace(tokens):
    """Block until sending `tokens` keeps the rolling 60 s total under TPM_BUDGET."""
    while True:
        with LOCK:
            now = time.time()
            SENT[:] = [(t, k) for t, k in SENT if now - t < 60]
            used = sum(k for _, k in SENT)
            if used + tokens <= TPM_BUDGET or not SENT:
                SENT.append((now, tokens))
                return
            wait = 60 - (now - SENT[0][0]) + 0.5
        time.sleep(max(wait, 0.5))


def append(path, recs):
    with LOCK, open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_batch(bid, batch, system, valid_ids):
    if DEADLINE and time.time() >= DEADLINE and not STOP.is_set():
        STOP.set()
        STOP_REASON.append("deadline")
    if STOP.is_set():
        return {"batch_id": bid, "n": len(batch), "written": 0, "error": "skipped: run stopped",
                "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0, "model": MODEL, "ts": time.time()}
    t0 = time.time()
    user = user_prompt(batch)
    est = est_tokens(len(system) + len(user)) + EST_OUT_PER_AWARD * len(batch)
    usage_all, err, got, key_name = [], None, None, None
    for attempt in (1, 2):                      # malformed JSON: retry once
        try:
            if not LOCAL:
                pace(est)
            content, usage, key_name = call(system, user)
            usage_all.append(usage)
            got = parse(content, len(batch))
            break
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            err = f"malformed JSON (attempt {attempt}): {e}"
        except DailyCap as e:
            err = f"DAILY CAP: {e}"
            break
        except RuntimeError as e:
            err = f"api: {e}"
            break
    recs = []
    if got is not None:
        for k, r in enumerate(batch, 1):
            if k not in got:
                continue                         # omitted by the model: left for a resume
            grounds, by_id = [], {}
            for g in got[k]:
                gid = str(g.get("id", "")).strip().upper()
                q = str(g.get("quote", ""))
                src = str(g.get("src", "")).strip().lower()
                grounds.append({"id": gid, "id_valid": gid in valid_ids, "quote": q,
                                "src": src if src in ("insurer", "ombudsman") else "missing",
                                "description": g.get("description", ""),
                                "quote_verified": bool(q.strip()) and norm_ws(q) in norm_ws(r["text"])})
            grounds = dedupe(grounds)
            recs.append({"uid": r["uid"], "award_id": r["award_id"], "source_pdf": r["source_pdf"],
                         "model": MODEL, "batch_id": bid, "cut": r["sent"].endswith("request limit ...]"), "grounds": grounds})
        append(CKPT, recs)
    log = {"batch_id": bid, "n": len(batch), "written": len(recs), "error": err if got is None else None,
           "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_all),
           "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_all),
           "seconds": round(time.time() - t0, 1), "model": MODEL, "ts": time.time(),
           "key": KEYS.index(key_name) if key_name else None,
           "n_grounds": sum(len(r["grounds"]) for r in recs),
           "n_verified": sum(g["quote_verified"] for r in recs for g in r["grounds"])}
    append(LOG, [log])
    return log


def dedupe(grounds):
    """One entry per ground id per award: keep the verified quote, then the insurer-sourced one,
    then the first."""
    best = {}
    for i, g in enumerate(grounds):
        rank = (not g["quote_verified"], g.get("src") != "insurer", i)
        if g["id"] not in best or rank < best[g["id"]][0]:
            best[g["id"]] = (rank, g)
    return [g for _, g in sorted(best.values(), key=lambda x: x[0][2])]


def label(limit=None, workers=WORKERS, until=None):
    global KEYS, DEADLINE
    KEYS = Keys()
    if not KEYS.keys:
        sys.exit("set QWEN_API_KEY in .env (and QWEN_BASE_URL / QWEN_MODEL)")
    if until:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        hh, mm = map(int, until.split(":"))
        DEADLINE = datetime.now(ist).replace(hour=hh, minute=mm, second=0, microsecond=0).timestamp()
    print(f"keys in .env: {[f'{KEYS.index(k)} ({k})' for k in KEYS.keys]}; starting with {KEYS.name}; "
          f"no batch starts after {until or '-'} IST", flush=True)
    rows_md, rules, ids = load_taxonomy()
    system = system_prompt(rows_md, rules) + ("\n/no_think" if NO_THINK else "")
    df = load_pool()
    done = done_uids()
    todo = df[~df.uid.isin(done)].sample(frac=1, random_state=20261018)   # mix books across batches
    # under a daily token cap, spend it where it matters: Prashant's rows, then health, then the rest
    prio = {"prashant": 0, "health": 1}
    reason = pd.read_csv(SAMPLE).set_index("uid").inclusion_reason.str.split(" ").str[0]
    todo = todo.assign(_p=todo.uid.map(reason).map(prio).fillna(2)).sort_values("_p", kind="stable").drop(columns="_p")
    if limit:
        todo = todo.head(limit)
    batches = make_batches(todo.to_dict("records"), len(system))
    run_id = time.strftime("%H%M%S")
    print(f"pool {len(df):,}; already labeled {len(done):,}; to do {len(todo):,} in {len(batches)} batches; "
          f"model {MODEL}; {workers} workers", flush=True)
    step = 100 if LOCAL else 50
    t0, n_ok, n_err, tok, next_report = time.time(), 0, 0, 0, step
    n_g = n_v = 0
    prev = [json.loads(l) for l in open(CKPT, encoding="utf-8") if l.strip()] if os.path.exists(CKPT) else []
    prev_g = [g for r in prev for g in dedupe(r["grounds"])]
    flagged = False
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_batch, f"{run_id}-{i:04d}", b, system, set(ids)) for i, b in enumerate(batches)]
        for i, f in enumerate(as_completed(futs), 1):
            log = f.result()
            n_ok += log["written"]
            n_err += log["error"] is not None
            tok += log["prompt_tokens"] + log["completion_tokens"]
            n_g += log.get("n_grounds", 0)
            n_v += log.get("n_verified", 0)
            if log["error"] and not log["error"].startswith("skipped"):
                print(f"  ERROR batch {log['batch_id']}: {log['error'][:300]}", flush=True)
            if STOP.is_set() and log["error"] and log["error"].startswith("DAILY CAP"):
                print(f"!! STOPPED: ALL KEYS EXHAUSTED at {time.strftime('%H:%M:%S')}: {len(done) + n_ok:,} of {len(df):,} "
                      f"sample awards labeled ({n_ok:,} this run). Exhausted: "
                      f"{ {KEYS.index(k): w for k, w in KEYS.exhausted.items()} }. Add a key to .env and resume.", flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break
            if STOP.is_set() and STOP_REASON:
                ex.shutdown(wait=True, cancel_futures=True)
                print(f"!! STOPPED AT DEADLINE {until} IST: {len(done) + n_ok:,} of {len(df):,} sample awards labeled "
                      f"({n_ok:,} this run)", flush=True)
                break
            if n_g >= 30 and n_v / n_g < 0.85 and not flagged:
                flagged = True
                print(f"!! FLAG verified-quote rate {n_v / n_g:.1%} this run (< 85%; was 94.2%)", flush=True)
            if n_ok >= next_report or i == len(batches):
                next_report = (n_ok // step + 1) * step
                el = time.time() - t0
                eta = (len(todo) - n_ok) / (n_ok / el) if n_ok else float("nan")
                cg, cv = len(prev_g) + n_g, sum(g["quote_verified"] for g in prev_g) + n_v
                print(f"  PROGRESS {time.strftime('%H:%M')} {len(done) + n_ok:,}/{len(df):,} awards "
                      f"({i}/{len(batches)} batches), verified {n_v / max(n_g, 1):.1%} this run "
                      f"({cv / max(cg, 1):.1%} cumulative, {cg} grounds), key {log.get('key')}, {n_err} failed batches, "
                      f"{tok:,} tokens, {STATS['http_429']} HTTP 429s, {el / 60:,.1f} min, ETA all {eta / 60:,.0f} min", flush=True)
    if STATS["dropped_params"]:
        print(f"endpoint rejected and dropped: {STATS['dropped_params']}", flush=True)
    print(f"done: {n_ok:,} awards, {n_err} failed batches, {tok:,} tokens, {STATS['http_429']} HTTP 429s, "
          f"{time.time() - t0:,.0f}s", flush=True)


# ---------------------------------------------------------------- finalize

def finalize():
    _, _, ids = load_taxonomy()
    recs = [json.loads(l) for l in open(CKPT, encoding="utf-8") if l.strip()]
    seen, rows = set(), []
    for r in recs:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"])
        mid = r.get("model") or "qwen/qwen3.8-27b"          # the first 139 records predate the field
        for g in dedupe(r["grounds"]):
            rows.append({"uid": r["uid"], "award_id": r["award_id"], "source_pdf": r["source_pdf"],
                         "label_source": "llm", "label_model": LABEL_MODEL.get(mid, mid), "model_id": mid,
                         "ground_id": g["id"], "quote": g["quote"], "quote_verified": g["quote_verified"],
                         "src": g.get("src", "missing"),
                         "id_valid": g["id_valid"], "description": g.get("description", ""),
                         "batch_id": r["batch_id"]})
    lab = pd.DataFrame(rows, columns=["uid", "award_id", "source_pdf", "ground_id", "quote", "quote_verified",
                                      "src", "id_valid", "description", "batch_id", "label_source", "label_model",
                                      "model_id"])
    raw = sum(len(r["grounds"]) for r in {r["uid"]: r for r in recs}.values())
    print(f"de-duplication: {raw:,} raw grounds -> {len(lab):,} (same ground repeated within an award collapsed)")
    lab.to_parquet(OUT, index=False)
    pd.DataFrame({"uid": sorted(seen)}).to_parquet(os.path.join(DATA, "labeled_awards.parquet"), index=False)
    awards = pd.Series({u: 0 for u in seen})
    awards.update(lab.groupby("uid").size())
    logs = pd.DataFrame([json.loads(l) for l in open(LOG, encoding="utf-8")]) if os.path.exists(LOG) else pd.DataFrame()

    print(f"awards labeled: {len(seen):,} of {len(load_pool()):,} in the labeling sample")
    print(f"label rows: {len(lab):,} -> {OUT}")
    print(f"grounds per award: mean {awards.mean():.2f}; distribution {awards.value_counts().sort_index().to_dict()}")
    print("per-category:", lab.ground_id.value_counts().to_dict())
    print(f"invalid ids: {(~lab.id_valid).sum()}")
    print(f"VERIFIED-QUOTE RATE: {lab.quote_verified.mean():.1%} ({lab.quote_verified.sum():,}/{len(lab):,})")
    print("verified by category:", lab.groupby("ground_id").quote_verified.mean().round(3).to_dict())
    print("quote source:", lab.src.value_counts().to_dict(), "| verified by source:",
          lab.groupby("src").quote_verified.mean().round(3).to_dict())
    g99 = lab[lab.ground_id == "G99"]
    print(f"G99: {len(g99)}")
    for _, r in g99.head(15).iterrows():
        print(f"  {r.award_id}: {r.description!s:.120} | {r.quote[:160]!r}")
    if len(logs):
        print(f"batches {len(logs)}, failed {logs.error.notna().sum()}; tokens prompt {logs.prompt_tokens.sum():,} "
              f"completion {logs.completion_tokens.sum():,}; wall {logs.ts.max() - (logs.ts - logs.seconds).min():,.0f}s "
              f"(first start to last end); model {logs.model.iloc[-1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--until", help="HH:MM IST; no batch starts after it")
    a = ap.parse_args()
    if a.show_prompt:
        rows_md, rules, _ = load_taxonomy()
        print(system_prompt(rows_md, rules))
    elif a.finalize:
        finalize()
    else:
        label(a.limit, a.workers, a.until)
