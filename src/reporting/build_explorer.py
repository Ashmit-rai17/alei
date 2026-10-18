"""Task C — evidence explorer: one self-contained HTML file, opens from file://.

  python scripts/build_explorer.py            # data/labels.parquet  -> data/explorer.html
  python scripts/build_explorer.py --stub     # data/labels_STUB.parquet -> data/explorer_STUB.html

Reads awards_safe.parquet only. Data is embedded as gzip+base64 JSON and inflated in the browser
with the native DecompressionStream (no library, no network).

Derived here for display only (the corpus is frozen and not modified):
  segment_quality  "suspect" when n_headers > 1, tail_header, n_chars > 20,000, or the award is on
                   the hand-checked weld / reprint / tail-bleed list; else "ok"
  outcome          heuristic on the last 2,500 chars: the latest of dismissed / allowed /
                   partly allowed / ex gratia; else "not extracted"
  insurer_group    case-folded insurer, LIC spellings merged; blank -> "(not extracted)"
"""

import argparse
import base64
import gzip
import json
import os
import re
import sys
import time

import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]
from display_mask import mask_policy, mask_quote  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
SAFE = os.path.join(DATA, "awards_safe.parquet")

NAMES = {"G1": "non_disclosure_proposal", "G2": "non_disclosure_revival", "G3": "fraud_misrepresentation",
         "G4": "policy_not_in_force", "G5": "documents_not_submitted", "G6": "excluded_by_clause",
         "G7": "pre_existing_disease", "G8": "waiting_period", "G9": "hospitalisation_not_justified",
         "G10": "treatment_not_covered", "G11": "amount_limited", "G12": "cause_outside_cover", "G99": "other"}

# hand-checked in the gate-2 cross-check (segment round, 17 Oct): real welds without a readable id,
# same award printed twice, residual tail bleed
CHECKED_SUSPECT = {
    "CHD-L-013-1718-0615": "weld", "24-01-071/L/05-06/GHY(LIC)": "weld", "NOI-L-029-1819-0275": "weld",
    "BHP-L-029-1718-0181": "possible weld", "CHD-L-046-1718-0704": "tail bleed", "BNG-H-044-2021-0819": "tail bleed",
    "KOL-L-019-2021-0872": "tail bleed", "KOL-L-025-2122-1518": "tail bleed", "NOI-L-008-2122-0982": "tail bleed",
    "GUW-L-033-2223-059": "tail bleed", "KOL-L-013-1718-1564": "tail bleed",
}

OUTCOMES = [
    ("partly allowed", r"(?i)partly allowed|partially allowed|allowed in part|allowed partially|allowed partly"),
    ("ex gratia", r"(?i)ex[- ]?gratia"),
    ("dismissed", r"(?i)\bdismiss(?:ed|al)\b|\bnot\s+(?:been\s+)?allowed\b|stands? closed|does not (?:warrant|call for) any interference|repudiation .{0,40}(?:is|was) (?:upheld|justified|in order)"),
    ("allowed", r"(?i)(?<!not )(?<!not been )\ballowed\b(?!\s+(?:partially|partly|in part))|is (?:hereby )?directed to (?:pay|settle)|shall pay|award is (?:hereby )?passed"),
]


def outcome(text):
    tail = text[-2500:]
    best, pos = "not extracted", -1
    for name, rx in OUTCOMES:
        for m in re.finditer(rx, tail):
            p = m.start() + (10_000 if name in ("partly allowed", "ex gratia") else 0)   # these win when present
            if p > pos:
                best, pos = name, p
    return best


ACRONYMS = {"axa", "sbi", "hdfc", "icici", "pnb", "aia", "aig", "idbi", "ergo", "lic", "dhfl", "irda", "ing", "max"}
RENAMES = [  # (prefix of the case-folded name, canonical) — case variants and known renames only
    ("lic", "LIC of India"), ("life insurance corporation", "LIC of India"),
    ("hdfc standard life", "HDFC Life"), ("hdfc life", "HDFC Life"),
    ("tata aig", "Tata AIA Life (formerly Tata AIG Life)"), ("tata aia", "Tata AIA Life (formerly Tata AIG Life)"),
    ("kotak life", "Kotak Mahindra Life"), ("kotak mahindra", "Kotak Mahindra Life"),
    ("bharti axa", "Bharti AXA Life"), ("max life", "Max Life"), ("pnb metlife", "PNB MetLife"),
]


def insurer_group(s):
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return "(not extracted)"
    low = s.lower()
    for pre, canon in RENAMES:
        if low.startswith(pre):
            return canon
    return " ".join(w.upper() if w.lower() in ACRONYMS else w[:1].upper() + w[1:].lower() for w in s.split())


def quality(r):
    reasons = []
    if r.n_headers > 1:
        reasons.append("2+ case numbers")
    if r.tail_header:
        reasons.append("next award's header at end")
    if r.n_chars > 20_000:
        reasons.append("over 20k chars")
    if r.award_id in CHECKED_SUSPECT:
        reasons.append("hand-checked: " + CHECKED_SUSPECT[r.award_id])
    return ("suspect" if reasons else "ok"), "; ".join(reasons)


def build(stub):
    lab_path = os.path.join(DATA, "labels_STUB.parquet" if stub else "labels.parquet")
    aw_path = os.path.join(DATA, "labeled_awards_STUB.parquet" if stub else "labeled_awards.parquet")
    out = os.path.join(DATA, "explorer_STUB.html" if stub else "explorer.html")
    safe = pd.read_parquet(SAFE, columns=["award_id", "source_pdf", "claim_type", "insurer", "award_date",
                                          "n_chars", "n_headers", "tail_header", "is_repudiation", "text"])
    safe["uid"] = safe.award_id + "|" + safe.source_pdf
    lab = pd.read_parquet(lab_path)
    uids = set(pd.read_parquet(aw_path).uid)
    rule_path = os.path.join(DATA, "labels_rule.parquet")
    rule_aw = os.path.join(DATA, "labeled_awards_rule.parquet")
    rlab = pd.read_parquet(rule_path) if not stub and os.path.exists(rule_path) else pd.DataFrame(columns=lab.columns)
    ruids = set(pd.read_parquet(rule_aw).uid) if not stub and os.path.exists(rule_aw) else set()
    rby = {u: g for u, g in rlab.groupby("uid")}
    samp_path = os.path.join(DATA, "label_sample.csv")
    reason = {} if stub or not os.path.exists(samp_path) else dict(
        pd.read_csv(samp_path)[["uid", "inclusion_reason"]].itertuples(index=False))
    d = safe                                      # every award is shown; labels cover a documented sample
    by = {u: g for u, g in lab.groupby("uid")}
    awards = []
    n_masked = n_masked_awards = 0
    for r in d.itertuples():
        q, why = quality(r)
        g = by.get(r.uid)
        text, found = mask_policy(r.text)            # display-time; the corpus itself stays frozen
        n_masked += len(found)
        n_masked_awards += bool(found)
        labels = [] if g is None else [
            {"g": x.ground_id if x.ground_id in NAMES else "G99", "q": mask_quote(x.quote, found), "v": bool(x.quote_verified),
             "s": str(getattr(x, "src", "") or "missing"),
             "n": str(getattr(x, "description", "") or "")} for x in g.itertuples()]
        rg = rby.get(r.uid)
        rlabels = [] if rg is None else [{"g": x.ground_id, "q": mask_quote(x.quote, found), "v": bool(x.quote_verified), "s": "rule", "n": ""}
                                         for x in rg.itertuples()]
        awards.append({"R": rlabels, "rb": r.uid in ruids, "u": r.uid, "id": r.award_id, "pdf": r.source_pdf, "ct": r.claim_type,
                       "ins": insurer_group(r.insurer), "insr": r.insurer or "", "dt": r.award_date or "",
                       "sq": q, "sqw": why, "oc": outcome(r.text), "t": text, "L": labels,
                       "lb": r.uid in uids, "rep": bool(r.is_repudiation), "why": reason.get(r.uid, "")})
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "stub": stub, "n_awards": len(awards),
            "n_labelled": sum(a["lb"] for a in awards), "n_rule": sum(a["rb"] for a in awards),
            "n_either": sum(a["lb"] or a["rb"] for a in awards),
            "policy_numbers_masked": n_masked, "awards_with_policy_mask": n_masked_awards,
            "labels_file": os.path.basename(lab_path), "names": NAMES,
            "verified_rate": float(lab.quote_verified.mean()) if len(lab) else None, "n_labels": len(lab)}
    blob = base64.b64encode(gzip.compress(json.dumps({"meta": meta, "awards": awards},
                                                     ensure_ascii=False).encode(), 9)).decode()
    html = TEMPLATE.replace("__BLOB__", blob).replace("__STUBBANNER__", STUB_BANNER if stub else "")
    open(out, "w", encoding="utf-8").write(html)
    sq = pd.Series([a["sq"] for a in awards]).value_counts().to_dict()
    oc = pd.Series([a["oc"] for a in awards]).value_counts().to_dict()
    print(f"{out}: {len(awards):,} awards ({meta['n_labelled']:,} labelled), {len(lab):,} labels, "
          f"{os.path.getsize(out) / 1e6:.1f} MB")
    print(f"segment_quality {sq}; outcome {oc}")


STUB_BANNER = ('<div class="stub">STUB DATA — keyword-rule labels for plumbing only. '
               'Not Qwen labels. Do not show or cite.</div>')

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alei Evidence Explorer</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66625a;--line:#e3dfd6;--card:#fff;--accent:#2f5d8a;--bar:#2f5d8a;
--warn-bg:#fff4d6;--warn-fg:#5c4500;--bad:#a33;--ok:#2d6a3e;--hl:#ffe08a;--hl2:#b9e0ff;--hlbad:#f6c7c7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#e8e6e1;--muted:#a19d94;--line:#34343a;
--card:#202025;--accent:#8db8e3;--bar:#5d8fc2;--warn-bg:#3b3215;--warn-fg:#f2dc9b;--bad:#f08a8a;--ok:#86c99a;--hl:#6b5714;--hl2:#1f4d70;--hlbad:#6e2a2a}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#e8e6e1;--muted:#a19d94;--line:#34343a;--card:#202025;--accent:#8db8e3;--bar:#5d8fc2;
--warn-bg:#3b3215;--warn-fg:#f2dc9b;--bad:#f08a8a;--ok:#86c99a;--hl:#6b5714;--hl2:#1f4d70;--hlbad:#6e2a2a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
header{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline}
header h1{font-size:18px;margin:0}nav a{margin-right:14px;color:var(--accent);text-decoration:none;font-weight:600}
nav a.on{text-decoration:underline}main{max-width:1150px;margin:0 auto;padding:16px}
.filters{display:flex;flex-wrap:wrap;gap:6px 18px;padding:10px 16px;border-bottom:1px solid var(--line);font-size:14px;color:var(--muted)}
.filters label{display:flex;gap:6px;align-items:center}.filters select{font:inherit}
.bias{background:var(--warn-bg);color:var(--warn-fg);border-radius:6px;padding:10px 12px;margin:0 0 16px;font-size:14px}
.stub{background:var(--bad);color:#fff;padding:8px 16px;font-weight:700}
.muted{color:var(--muted)}.n{font-variant-numeric:tabular-nums}
button.link,a.link{background:none;border:0;padding:0;color:var(--accent);cursor:pointer;font:inherit;text-decoration:underline;text-underline-offset:2px}
.bars{display:grid;grid-template-columns:minmax(150px,280px) 1fr auto;gap:6px 10px;align-items:center}
.bar{height:18px;background:var(--bar);border-radius:3px;cursor:pointer;min-width:2px}.bar:hover{opacity:.8}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--bg)}.scroll{overflow-x:auto}
.chip{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin:1px 3px 1px 0;font-size:12px;white-space:nowrap}
.chip.bad{border-color:var(--bad);color:var(--bad)}.sq-suspect{color:var(--bad)}.sq-ok{color:var(--ok)}
.meta{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:14px;margin-bottom:12px}
.lab{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:10px 14px;margin:10px 0;background:var(--card)}
.lab.unv{border-left-color:var(--bad)}.lab .q{font:italic 16px/1.6 Charter,"Iowan Old Style",Georgia,serif;margin:6px 0 2px}
.text{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px 26px;max-width:74ch;
font:17px/1.75 Charter,"Iowan Old Style","Source Serif Pro",Georgia,serif;overflow-wrap:break-word;hyphens:auto}
.tag{font:600 10.5px/1 system-ui,-apple-system,sans-serif;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
background:var(--line);border-radius:4px;padding:3px 5px;vertical-align:2px;white-space:nowrap}
mark{background:var(--hl);color:inherit;border-radius:3px;padding:1px 1px;-webkit-box-decoration-break:clone;box-decoration-break:clone}
mark.alt{background:var(--hl2)}mark.flash{outline:3px solid var(--accent);outline-offset:1px}
h2{font-size:18px;margin:22px 0 10px}tr:hover td{background:var(--card)}
.cell{cursor:pointer}.cell:hover{background:var(--warn-bg)}h2{font-size:17px;margin:4px 0 10px}
@media (max-width:600px){.bars{grid-template-columns:1fr auto}.bars .bar{grid-column:1/-1}}
</style></head><body>
__STUBBANNER__
<header><h1>Alei — Evidence Explorer</h1>
<nav><a href="#overview">Grounds</a><a href="#insurers">By insurer</a><a href="#outcomes">Outcome by ground</a><a href="#list">All awards</a></nav>
<span class="muted" id="metaLine"></span></header>
<div class="filters">
 <label><input type="checkbox" id="fOk"> exclude suspect segments</label>
 <label><input type="checkbox" id="fVer"> only labels with a verified quote</label>
 <label><input type="checkbox" id="fSrc"> only quotes from the insurer's own statement</label>
 <label>label source <select id="fLs"><option value="both">LLM where present, else keyword rule</option><option value="llm">LLM only</option><option value="rule">keyword rule only</option></select></label>
 <label>claim type <select id="fCt"><option value="">all</option></select></label>
 <span id="fCount" class="n"></span>
</div>
<main id="app"><p>Loading…</p></main>
<script id="blob" type="application/octet-stream">__BLOB__</script>
<script>
"use strict";
const BIAS='These are contested denials that reached an Ombudsman — a small fraction of all repudiations. This describes grounds that were challenged, not grounds that are typical.';
let D, NAMES, ORDER;
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=(a,b)=>b?(100*a/b).toFixed(1)+'%':'–';
const F={ok:false,ver:false,src:false,ct:'',ls:'both'};

async function load(){
  const b64=$('#blob').textContent.trim();
  const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
  if(typeof DecompressionStream==='undefined'){$('#app').innerHTML='<p>This browser lacks DecompressionStream; open in a current Chrome, Edge, Firefox or Safari.</p>';return}
  const txt=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  D=JSON.parse(txt);NAMES=D.meta.names;ORDER=Object.keys(NAMES);
  for(const a of D.awards)a.t=cleanText(a.t);
  const cts=[...new Set(D.awards.map(a=>a.ct))].sort();
  $('#fCt').innerHTML+=cts.map(c=>`<option>${esc(c)}</option>`).join('');
  $('#metaLine').textContent=`${D.meta.n_awards.toLocaleString()} awards · ${D.meta.n_labelled.toLocaleString()} labelled (documented sample) · labels: ${D.meta.labels_file} · built ${D.meta.generated}`;
  for(const [id,k] of [['#fOk','ok'],['#fVer','ver'],['#fSrc','src']]) $(id).onchange=e=>{F[k]=e.target.checked;route()};
  $('#fCt').onchange=e=>{F.ct=e.target.value;route()};
  $('#fLs').onchange=e=>{F.ls=e.target.value;route()};
  window.onhashchange=route;route();
}
// awards passing the filters, each with its effective label list; aggregates use labelled awards only
// label set per award: LLM labels, keyword-rule labels on ground_span, or LLM where present else rule
const has=a=>F.ls==='llm'?a.lb:F.ls==='rule'?a.rb:(a.lb||a.rb);
const base=a=>F.ls==='llm'?a.L:F.ls==='rule'?a.R:(a.lb?a.L:a.R);
const effL=a=>base(a).filter(l=>(!F.ver||l.v)&&(!F.src||l.s==='insurer'));
function view(all){
  const out=[];
  for(const a of D.awards){
    if(!all&&!has(a))continue;
    if(F.ok&&a.sq!=='ok')continue; if(F.ct&&a.ct!==F.ct)continue;
    const L=effL(a);
    out.push({a,L,G:new Set(L.map(l=>l.g))});
  }
  $('#fCount').textContent=all?`${out.length.toLocaleString()} awards in view (labelled and unlabelled)`:`${out.length.toLocaleString()} labelled awards in view`;
  return out;
}
const bias=()=>`<div class="bias" role="note">${BIAS}<br><br>Counts cover labelled awards only — ${D.meta.n_labelled.toLocaleString()} labelled by the LLM (a stratified sample: every health award and every human-checked award included) and ${D.meta.n_rule.toLocaleString()} run through keyword rules on the extracted ground sentence (awards where no rule fires show “none found”), of a ${D.meta.n_awards.toLocaleString()}-award corpus. The “label source” filter chooses which. Unlabelled awards are browsable under “All awards”.</div>`;
const go=(h)=>{location.hash=h};
function qs(o){return Object.entries(o).filter(([,v])=>v!==''&&v!=null).map(([k,v])=>k+'='+encodeURIComponent(v)).join('&')}
function parseHash(){const [p,q]=location.hash.slice(1).split('?');const o={};(q||'').split('&').filter(Boolean).forEach(kv=>{const [k,v]=kv.split('=');o[k]=decodeURIComponent(v||'')});return [p||'overview',o]}
function route(){
  if(!D)return; const [p,o]=parseHash();
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+p));
  ({overview,insurers,outcomes,list,award}[p]||overview)(o); window.scrollTo(0,0);
}
function overview(){
  const V=view(),N=V.length,cnt={};let none=0;
  for(const v of V){if(!v.G.size)none++;for(const g of v.G)cnt[g]=(cnt[g]||0)+1}
  const mx=Math.max(1,...Object.values(cnt));
  let h=bias()+`<h2>Grounds the insurer gave — awards citing each ground, of ${N.toLocaleString()} in view</h2><div class="bars">`;
  for(const g of ORDER){const c=cnt[g]||0;
    h+=`<div><b>${g}</b> ${esc(NAMES[g])}</div><div class="bar" style="width:${100*c/mx}%" title="open the ${c} awards" onclick="go('list?${qs({g})}')"></div>
    <button class="link n" onclick="go('list?${qs({g})}')">${c.toLocaleString()} of ${N.toLocaleString()} (${pct(c,N)})</button>`}
  h+=`<div class="muted">no ground found</div><div></div><button class="link n" onclick="go('list?g=none')">${none.toLocaleString()} of ${N.toLocaleString()}</button></div>
  <p class="muted">An award can cite several grounds, so the counts add up to more than ${N.toLocaleString()}. Every number opens the awards behind it.</p>`;
  $('#app').innerHTML=h;
}
function insurers(){
  const V=view(),by={};
  for(const v of V)(by[v.a.ins]=by[v.a.ins]||[]).push(v);
  const rows=Object.entries(by).filter(([,a])=>a.length>=20).sort((x,y)=>y[1].length-x[1].length);
  let h=bias()+`<h2>Grounds by insurer — insurers with at least 20 awards in view</h2><p class="muted">Each cell: awards citing the ground / that insurer's awards in view. Click to open them.</p><div class="scroll"><table><tr><th>insurer</th><th>awards</th>${ORDER.map(g=>`<th title="${esc(NAMES[g])}">${g}</th>`).join('')}</tr>`;
  for(const [ins,a] of rows){const n=a.length;
    h+=`<tr><td>${esc(ins)}</td><td><button class="link n" onclick="go('list?${qs({ins})}')">${n}</button></td>`;
    for(const g of ORDER){const c=a.filter(v=>v.G.has(g)).length;
      h+=c?`<td class="cell n" onclick="go('list?${qs({ins,g})}')" title="${c} of ${n} ${esc(ins)} awards">${c} / ${n}<br><span class="muted">${pct(c,n)}</span></td>`:`<td class="muted n">0 / ${n}</td>`}
    h+='</tr>'}
  const small=Object.values(by).filter(a=>a.length<20).reduce((s,a)=>s+a.length,0);
  h+=`</table></div><p class="muted">${small.toLocaleString()} awards from insurers with fewer than 20 awards are not shown as rows. "(not extracted)" means the insurer name could not be read from the award header.</p>`;
  $('#app').innerHTML=h;
}
function outcomes(){
  const V=view(),OC=['dismissed','allowed','partly allowed','ex gratia','not extracted'];
  let h=bias()+`<h2>Outcome by ground</h2><p class="muted">Outcome is a text heuristic on the award's closing passage (the last of: dismissed / allowed / partly allowed / ex gratia). Each cell: awards / awards citing that ground.</p><div class="scroll"><table><tr><th>ground</th><th>awards</th>${OC.map(o=>`<th>${o}</th>`).join('')}</tr>`;
  for(const g of ORDER){const a=V.filter(v=>v.G.has(g)),n=a.length;if(!n)continue;
    h+=`<tr><td><b>${g}</b> ${esc(NAMES[g])}</td><td><button class="link n" onclick="go('list?${qs({g})}')">${n}</button></td>`;
    for(const o of OC){const c=a.filter(v=>v.a.oc===o).length;
      h+=`<td class="cell n" onclick="go('list?${qs({g,oc:o})}')">${c} / ${n}<br><span class="muted">${pct(c,n)}</span></td>`}
    h+='</tr>'}
  $('#app').innerHTML=h+'</table></div>';
}
function list(o){
  const any=o.g||o.ins||o.oc;
  let V=view(!any);
  if(o.g==='none')V=V.filter(v=>!v.G.size);else if(o.g)V=V.filter(v=>v.G.has(o.g));
  if(o.ins)V=V.filter(v=>v.a.ins===o.ins); if(o.oc)V=V.filter(v=>v.a.oc===o.oc);
  const lim=+(o.lim||300);
  const title=[o.g&&(o.g==='none'?'no ground found':`${o.g} ${NAMES[o.g]}`),o.ins,o.oc].filter(Boolean).join(' · ')||'all awards';
  let h=`<h2>${esc(title)} — ${V.length.toLocaleString()} awards</h2><div class="scroll"><table><tr><th>award</th><th>insurer</th><th>date</th><th>claim type</th><th>grounds</th><th>outcome</th><th>segment</th></tr>`;
  for(const v of V.slice(0,lim)){const a=v.a;
    h+=`<tr><td><button class="link" onclick="go('award?${qs({u:a.u,g:o.g||''})}')">${esc(a.id)}</button><br><span class="muted">${esc(a.pdf)}</span></td><td>${esc(a.ins)}</td><td class="n">${esc(a.dt)}</td><td>${esc(a.ct)}</td>
    <td>${has(a)?([...v.G].map(g=>`<span class="chip">${g}</span>`).join('')||'<span class="muted">none found</span>')+` <span class="muted">${a.lb&&F.ls!=='rule'?'LLM':'rule'}</span>`:'<span class="chip bad">unlabelled</span>'}</td><td>${esc(a.oc)}</td><td class="sq-${a.sq}" title="${esc(a.sqw)}">${a.sq}</td></tr>`}
  h+='</table></div>';
  if(V.length>lim)h+=`<p><button class="link" onclick="go('list?${qs({...o,lim:lim+300})}')">show ${Math.min(300,V.length-lim)} more</button></p>`;
  $('#app').innerHTML=h;
}
// Display text: the PDF's hard line breaks, page footers, separator rules and space runs removed, so the
// award reads as paragraphs. Only whitespace and boilerplate change, so quotes still match (locate below
// ignores whitespace). Breaks are kept before list items and numbered points, after headings ending ":",
// and between paragraphs.
function cleanText(t){
  return t.replace(/\r/g,'')
    .replace(/^[ \t]*Page\s+\d+\s+of\s+\d+[ \t]*$/gmi,'')
    .replace(/^[ \t]*([-*=_#~.•·])\1{3,}[ \t]*$/gm,'')
    .replace(/[ \t]+/g,' ')
    .replace(/ *\n */g,'\n')
    .replace(/\n{3,}/g,'\n\n')
    .replace(/([^\n:;])\n(?!\n|\(?[0-9]{1,2}[.)]\s|\(?[a-z]{1,4}\)\s|[•·]\s|[A-Z][A-Z .&/]{8,}\n)/g,'$1 ')
    .trim();
}
const TAG_LABEL={NAME:'name',NAME_ADDRESS:'name & address',POLICY_NO:'policy no.',HOSPITAL:'hospital',NUM:'number',
  ADDRESS:'address',PIN:'pin',PHONE:'phone',EMAIL:'email'};
const tagify=h=>h.replace(/\[([A-Z_]+)\]/g,(m,k)=>`<span class="tag" title="redacted">${TAG_LABEL[k]||k.toLowerCase()}</span>`);
// locate a quote in the text: whitespace-insensitive, otherwise exact
function locate(text,q){
  // separator rules are dropped from the display text, so drop them from the quote as well
  const toks=q.replace(/([-*=_#~.•·])\1{3,}/g,' ').trim().split(/\s+/).filter(Boolean); if(!toks.length)return null;
  const re=new RegExp(toks.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('\\s+'));
  const m=re.exec(text); return m?[m.index,m.index+m[0].length]:null;
}
function award(o){
  const a=D.awards.find(x=>x.u===o.u); if(!a){$('#app').innerHTML='<p>Award not found.</p>';return}
  const L=effL(a);
  const spans=L.map((l,i)=>{const p=locate(a.t,l.q);return p?{s:p[0],e:p[1],i,g:l.g}:null}).filter(Boolean);
  // split text at every span boundary; each piece knows which labels cover it
  const cuts=[...new Set([0,a.t.length,...spans.flatMap(s=>[s.s,s.e])])].sort((x,y)=>x-y);
  let body='';
  for(let k=0;k<cuts.length-1;k++){const s=cuts[k],e=cuts[k+1],piece=tagify(esc(a.t.slice(s,e)));
    const cov=spans.filter(x=>x.s<=s&&x.e>=e);
    body+=cov.length?`<mark id="q${cov[0].i}" class="${cov[0].g===o.g||!o.g?'':'alt'}" title="${cov.map(c=>c.g).join(', ')}">${piece}</mark>`:piece}
  let h=`<p><button class="link" onclick="history.back()">← back</button></p><h2>${esc(a.id)}</h2>
  <div class="meta"><span class="muted">insurer</span><span>${esc(a.insr||'(not extracted)')}</span><span class="muted">award date</span><span>${esc(a.dt||'(not extracted)')}</span>
  <span class="muted">claim type</span><span>${esc(a.ct)}</span><span class="muted">source book</span><span>${esc(a.pdf)}</span>
  <span class="muted">labelling</span><span>${a.lb?'LLM (Qwen)'+(a.why?' — in sample as: '+esc(a.why):''):''}${a.lb&&a.rb?' · ':''}${a.rb?'keyword rules on the ground sentence':''}${!a.lb&&!a.rb?'<b class="sq-suspect">unlabelled</b>'+(a.rep?'':' (no repudiation found in text)'):''}</span>
  <span class="muted">outcome (heuristic)</span><span>${esc(a.oc)}</span><span class="muted">segment quality</span><span class="sq-${a.sq}">${a.sq}${a.sqw?' — '+esc(a.sqw):''}</span></div>
  <h2>Grounds the insurer gave (${L.length})</h2>`;
  if(!has(a))h+='<p class="muted">This award has no labels from the selected source. Its text is shown for completeness.</p>';
  else if(!L.length)h+='<p class="muted">No ground was found in this award (or none passes the current filters).</p>';
  L.forEach((l,i)=>{const found=spans.some(s=>s.i===i);
    h+=`<div class="lab ${l.v?'':'unv'}"><b>${l.g}</b> ${esc(NAMES[l.g]||'')} ${l.v?'<span class="chip">quote verified</span>':'<span class="chip bad">quote NOT found in text</span>'} <span class="chip">source: ${esc(l.s||'missing')}</span>
    ${l.n?`<div class="muted">note: ${esc(l.n)}</div>`:''}<div class="q">“${tagify(esc(l.q))}”</div>${found?`<button class="link" onclick="flash(${i})">show in text ↓</button>`:''}</div>`});
  h+=`<h2>Award text (redacted)</h2><div class="text">${body}</div>`;
  $('#app').innerHTML=h;
  const first=spans.find(s=>!o.g||s.g===o.g); if(first)setTimeout(()=>flash(first.i),50);
}
function flash(i){const m=document.getElementById('q'+i);if(!m)return;m.scrollIntoView({block:'center'});m.classList.add('flash');setTimeout(()=>m.classList.remove('flash'),1600)}
load();
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    build(ap.parse_args().stub)
