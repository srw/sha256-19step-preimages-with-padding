"""Count and verify every archived witness at commit 3130a9a with my own SHA-256 (my_sha.py).
usage: my_witness_check.py WITNESS_DIR"""
import re, sys, hashlib, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_sha import hash_steps, comp_steps, compress, H0, pad

root = sys.argv[1]
FILES = ['witnesses_19step.txt', 'demos_19step.txt', 'examples_abc_hello.txt', 'witnesses_fresh_2026-09-29.txt']
recs = []
for fn in FILES:
    txt = open(f'{root}/{fn}').read()
    for chunk in re.split(r'(?=^\[(?:padded|comp|demo)\])', txt, flags=re.M):
        m = re.match(r'\[(padded|comp|demo)\]', chunk)
        if not m: continue
        kind = m.group(1)
        tgt = re.search(r'^\s*target\s*=\s*(?:SHA-256\("(.*?)"\)[^\n]*|([0-9a-f]{64}))', chunk, re.M)
        lab = re.search(r'label="(.*?)"', chunk)
        label = lab.group(1) if lab else None
        if kind == 'demo':
            lab2 = re.search(r'SHA-256\("(.*?)"\)', chunk)
            label = lab2.group(1) if lab2 else None
        target = re.search(r'^\s*target\s*=\s*.*?([0-9a-f]{64})?\s*$', chunk, re.M)
        # target hex: explicit "target = <hex>" line, else the sha256_19 line (demos print only that)
        t = re.search(r'^\s*target\s*=\s*([0-9a-f]{64})\s*$', chunk, re.M)
        s19 = re.search(r'^\s*sha256_19\s*=\s*([0-9a-f]{64})', chunk, re.M)
        msg = re.search(r'message_hex\s*=\s*([0-9a-f]+)', chunk)
        words = re.search(r'W\[0\.\.15\]\s*=\s*((?:[0-9a-f]{8}\s+){15}[0-9a-f]{8})', chunk)
        recs.append(dict(file=fn, kind=kind, label=label,
                         target=(t.group(1) if t else (s19.group(1) if s19 else None)),
                         msg=bytes.fromhex(msg.group(1)) if msg else None,
                         block=bytes.fromhex(''.join(words.group(1).split())) if words else None))
print("records parsed:", len(recs))
fails = 0
rows = []
for r in recs:
    checks = {}
    tgt = r['target']
    assert tgt, r
    if r['label'] is not None:
        checks['target==sha256(label)'] = hashlib.sha256(r['label'].encode()).hexdigest() == tgt
    if r['msg'] is not None:                       # padded (or demo) record
        m = r['msg']
        checks['len in {55,119}'] = len(m) in (55, 119)
        h = {s: hash_steps(m, s) for s in (18, 19, 20, 64)}
        zero_fill = comp_steps(m + bytes(64 - len(m)) if len(m) <= 64 else m[:64], 19)   # padding omitted: raw block
        kindname = 'padded'
        checks['19-step hit'] = h[19] == tgt
        checks['18/20/64-step miss'] = all(h[s] != tgt for s in (18, 20, 64))
        if len(m) == 55:
            checks['unpadded-raw-block would miss'] = zero_fill != tgt
    else:
        b = r['block']
        kindname = 'compression'
        checks['len 64'] = len(b) == 64
        h = {s: comp_steps(b, s) for s in (18, 19, 20, 64)}
        checks['19-step hit'] = h[19] == tgt
        checks['18/20/64-step miss'] = all(h[s] != tgt for s in (18, 20, 64))
        checks['as-padded-message misses'] = hash_steps(b, 19) != tgt
    ok = all(checks.values())
    fails += (not ok)
    rows.append((r['file'], r['kind'], kindname, len(r['msg']) if r['msg'] else 64, r['label'], tgt, ok))
    if not ok: print("FAIL", r['file'], r['label'], checks)
from collections import Counter
print("records:", len(rows), " failures:", fails)
print("by model:", Counter(x[2] for x in rows))
keys = {(x[2], (r['msg'] or r['block']).hex(), x[5]) for x, r in zip(rows, recs)}
print("distinct witnesses (model,data,target):", len(keys), Counter(k[0] for k in keys),
      " padded lengths:", Counter(len(bytes.fromhex(k[1])) for k in keys if k[0] == 'padded'))
print("per-file:", dict(Counter((x[0], x[2]) for x in rows)))
dups = Counter((x[2], (r['msg'] or r['block']).hex()) for x, r in zip(rows, recs))
print("duplicated records:", [(k[0], k[1][:16] + '...') for k, v in dups.items() if v > 1])
