"""Verify solver output with my own SHA-256: every 'W =' block must be a valid 55-byte padded block whose 19-step
hash equals the target, with 18/20/64-step controls.  usage: my_run_verify.py OUTFILE TARGETS_FILE LABELS_JSON|-"""
import sys, re, hashlib, json, statistics
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_sha import hash_steps, pad
out, tfile = sys.argv[1], sys.argv[2]
targets = open(tfile).read().strip().split(',')
txt = open(out).read()
founds = re.findall(r'FOUND ok=(\d) mode=(\w+) target=(\d+) contexts_tried=(\d+) contexts_used=(\d+) chains=(\d+) search_time=([0-9.]+)s\nW =((?: [0-9a-f]{8}){16})\nH19=([0-9a-f]{64})', txt)
print(f"{out}: {len(founds)} FOUND blocks for {len(targets)} targets")
ok_all = True; rows = []
for ok, mode, ti, ct, cu, ch, st, w, h19 in founds:
    ti = int(ti)
    block = bytes.fromhex(w.replace(' ', ''))
    msg = block[:55]
    tgt = targets[ti]
    checks = {
        'solver_ok': ok == '1' and mode == 'padded',
        'block is the standard padding of the 55-byte message': pad(msg) == block,
        '19-step hash == target': hash_steps(msg, 19) == tgt,
        'printed H19 == target': h19 == tgt,
        '18/20/64-step controls miss': all(hash_steps(msg, s) != tgt for s in (18, 20, 64)),
    }
    good = all(checks.values()); ok_all &= good
    if not good: print("FAIL target", ti, checks)
    rows.append((ti, int(ct), int(cu), int(ch), float(st), good))
print("all verified by my SHA-256:", ok_all, "; distinct targets solved:", len({r[0] for r in rows}), "of", len(targets))
ts = [r[4] for r in rows]
print("search CPU s: n=%d median=%.2f mean=%.2f min=%.1f max=%.1f" % (len(ts), statistics.median(ts), statistics.mean(ts), min(ts), max(ts)))
print("per-target:", [(r[0], r[4], r[2]) for r in rows])
