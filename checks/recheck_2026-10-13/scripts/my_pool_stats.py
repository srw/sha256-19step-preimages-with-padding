import re, math, statistics as st, json
import sys
LOGS, REPLAY, MINE = sys.argv[1], sys.argv[2], sys.argv[3]   # raw_solver_logs dir, replay .out, new-labels .out
pat = re.compile(r'FOUND ok=1 mode=padded target=(\d+) contexts_tried=(\d+) contexts_used=(\d+) chains=(\d+) search_time=([0-9.]+)s')
def load(path):
    return [tuple(map(float, m.groups()[1:])) + (int(m.group(1)),) for m in pat.finditer(open(path).read())]
# each tuple: (contexts_tried, contexts_used, chains, search_time, target_index)
L = LOGS.rstrip('/') + '/'
arch_single = load(L + 'padded_2_11.txt') + load(L + 'demo_abc.txt') + load(L + 'fresh_2026-09-29_padded.txt')
arch_two = load(L + 'demo_prefix.txt')
rep = load(REPLAY); mine = load(MINE)
rep_struct, rep_lab = rep[:2], rep[2:]
def summ(name, rows, idx=2):
    ch = [r[2] for r in rows]; ct = [r[0] for r in rows]; tm = [r[3] for r in rows]; cu = [r[1] for r in rows]
    srt = sorted(tm); n = len(tm)
    p90 = srt[min(n - 1, math.ceil(0.9 * n) - 1)]
    print(f"{name:<44} n={n:2d}  time med={st.median(tm):6.2f} mean={st.mean(tm):6.2f} min={min(tm):5.1f} p90={p90:5.1f} max={max(tm):5.1f} | "
          f"chains mean=2^{math.log2(st.mean(ch)):.2f} median=2^{math.log2(st.median(ch)):.2f} | ctx trials mean=2^{math.log2(st.mean(ct)):.2f} | contexts used: {sorted(cu)[:1]}..{max(cu):.0f} mean {st.mean(cu):.2f}")
summ("archived 55-byte (author machine) ", arch_single)
summ("archived 55-byte + 2-block demo", arch_single + arch_two)
summ("audit fresh labels (=my replay) [Xeon]", rep_lab)
summ("audit structured (all-1/all-0) [Xeon]", rep_struct)
summ("mine fresh labels [Xeon]", mine)
summ("Xeon all 22 (12 replay + 10 mine)", rep + mine)
summ("chain/ctx pooled over 32 label targets", arch_single + rep_lab + mine)
lab = arch_single + rep_lab + mine
ch = [r[2] for r in lab]
print("pooled 32 label targets: chains per solution mean = %.3g = 2^%.2f ; standard error of the mean = %.3g (%.0f%%)" % (st.mean(ch), math.log2(st.mean(ch)), st.stdev(ch) / math.sqrt(len(ch)), 100 * st.stdev(ch) / math.sqrt(len(ch)) / st.mean(ch)))
print("contexts used per solution (32 labels):", {k: sum(1 for r in lab if r[1] == k) for k in (1, 2, 3)})
print("fraction solved with the first context: %.2f" % (sum(1 for r in lab if r[1] == 1) / len(lab)))
