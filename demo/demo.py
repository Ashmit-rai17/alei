"""ALEI demo: keyword rules and the classifier, side by side, on 3 redacted awards.

Run through run_demo.sh. Prints the machine, model load time, both predictions per award with
the evidence behind them, and total runtime. One screen, no warnings.
"""

import time

T_START = time.perf_counter()

import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predict_rules import NAMES, rule_explain  # noqa: E402

W = 84
COL = 40
SEP = re.compile(r"(?m)^=+ *(AWARD[^\n=]*?) *=+ *$")


def read(path):
    try:
        return open(path).read()
    except OSError:
        return ""


def os_name():
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)', read("/etc/os-release"), re.M)
    if m:
        return m.group(1)
    if platform.system() == "Darwin":
        return "macOS " + platform.mac_ver()[0]
    return platform.platform()


def cpu_ram():
    cpu, ram = None, None
    info = read("/proc/cpuinfo")
    if info:
        vendor = re.search(r"^vendor_id\s*:\s*(.+)$", info, re.M)
        mach = re.search(r"machine\s*=\s*(\w+)", info)
        model = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
        if vendor and "S390" in vendor.group(1):
            cpu = f"IBM Z (machine type {mach.group(1)})" if mach else "IBM Z"
        elif model:
            cpu = model.group(1).strip()
        mem = re.search(r"^MemTotal:\s*(\d+)\s*kB", read("/proc/meminfo"), re.M)
        if mem:
            ram = int(mem.group(1)) * 1024
    elif platform.system() == "Darwin":
        import subprocess
        try:
            cpu = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            ram = int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        except (OSError, ValueError):
            pass
    cpu = cpu or platform.processor() or "unknown"
    return f"{cpu} x {os.cpu_count()} cores", (f"{ram / 2**30:.1f} GB" if ram else "unknown")


def explain(bundle, feats, names, text, top=3):
    if bundle.get("preprocess_regex"):                     # same cleaning the model was trained with
        text = re.sub(bundle["preprocess_regex"], bundle.get("preprocess_replace", " "), text)
    x = feats.transform([text])
    clf = bundle["pipeline"].named_steps["clf"]
    probs = clf.predict_proba(x)[0]
    out = []
    for c, p, est in zip(bundle["classes"], probs, clf.estimators_):
        if p < bundle["threshold"] or not hasattr(est, "coef_"):
            continue
        contrib = x.multiply(est.coef_.ravel()).tocsr()
        pairs = sorted(zip(contrib.indices, contrib.data), key=lambda t: -t[1])
        words = [names[i].split("__", 1)[1] for i, v in pairs if v > 0 and names[i].startswith("word__")][:top]
        out.append((c, float(p), words))
    return sorted(out, key=lambda r: -r[1])


def clip(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    import joblib
    import numpy
    import scipy
    import sklearn

    cpu, ram = cpu_ram()
    t0 = time.perf_counter()
    bundle = joblib.load(os.path.join(HERE, "alei_clf.joblib"))
    load_s = time.perf_counter() - t0
    feats = bundle["pipeline"].named_steps["features"]
    names = feats.get_feature_names_out()
    tag = (read(os.path.join(HERE, "MODEL_STATUS.txt")).strip() or "INTERIM").split()[0]

    print("=" * W)
    print(" ALEI · why Indian insurers reject claims · grounds from Ombudsman awards")
    print(f" uname -m {platform.machine()} · {os_name()} · Python {platform.python_version()}")
    print(f" CPU      {clip(cpu, 60)} · RAM {ram}")
    print(f" model    alei_clf.joblib ({tag}, {bundle['n_train']} training awards) loaded in {load_s:.2f} s")
    print(f"          scikit-learn {sklearn.__version__} · numpy {numpy.__version__} · scipy {scipy.__version__}")
    print("=" * W)

    text = open(os.path.join(HERE, "sample_awards.txt")).read()
    parts = SEP.split(text)
    awards = []
    for title, body in zip(parts[1::2], parts[2::2]):
        m = re.match(r"AWARD \d+: (.+?) \((.+?); (?:Qwen|LLM) label ([^)]+)\)", title.strip())
        awards.append((m.groups() if m else (title.strip(), "", ""), body))
    metrics = dict(line.split("=", 1) for line in read(os.path.join(HERE, "METRICS.txt")).splitlines() if "=" in line)

    t = time.perf_counter()
    rules = [rule_explain(body) for _, body in awards]
    t_rules = time.perf_counter() - t
    t = time.perf_counter()
    preds = [explain(bundle, feats, names, body) for _, body in awards]
    t_clf = time.perf_counter() - t

    print(f" 1  DETERMINISTIC EXTRACTOR — keyword rules, no ML · {metrics.get('rules', '')}")
    for k, ((aid, ctype, llm), _) in enumerate(awards):
        print(f"    award {k + 1}  {clip(aid, 30)} · {ctype} · reference label {llm}")
        for g, phrase in rules[k] or [("", "")]:
            print(f"       {g + ' ' + NAMES[g] if g else '(no rule fired)':34} {clip(chr(34) + phrase + chr(34), W - 42) if g else ''}")
    print("-" * W)
    print(f" 2  LEARNED CLASSIFIER — TF-IDF + logistic regression · {metrics.get('classifier', '')}")
    for k, ((aid, ctype, llm), _) in enumerate(awards):
        print(f"    award {k + 1}  {clip(aid, 30)} · reference label {llm}")
        for g, p, words in preds[k] or [("", 0.0, [])]:
            if g:
                print(f"       {g + ' ' + NAMES[g]:34} p={p:.2f}  {clip(' · '.join(words), W - 50)}")
            else:
                print("       (no ground above threshold)")
    print("-" * W)
    total = time.perf_counter() - T_START
    print(f" total {total:.2f} s  ·  model load {load_s:.2f} s  ·  rules {1000 * t_rules / len(awards):.1f} ms/award"
          f"  ·  classifier {1000 * t_clf / len(awards):.0f} ms/award")
    num = lambda k: (re.search(r"\d\.\d+", metrics.get(k, "")) or [None])[0]
    where = "an IBM Z mainframe (s390x)" if platform.machine() == "s390x" else f"{platform.machine()} ({os_name()})"
    if num("rules") and num("classifier"):
        print(f" A deterministic extractor and a learned classifier agree at {num('rules')} and {num('classifier')} macro-F1")
        print(f" against human labels ({metrics.get('n', '?')} awards labelled blind, none used in training),")
        print(f" on {where}, with nothing leaving the box.")
    print(" input: redacted award text; residual identifiers measured in 22 of 64 audited awards,")
    print(" policy numbers additionally masked for display")
    print("=" * W)


if __name__ == "__main__":
    main()
