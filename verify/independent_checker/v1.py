import hashlib, os, re, sys, struct
from shar import *

# --- self-test of implementation at R=64 against hashlib
import random
rng = random.Random(12345)
n_ok = 0
for L in list(range(0, 200)) + [rng.randrange(0, 2000) for _ in range(300)]:
    msg = bytes(rng.randrange(256) for _ in range(L))
    assert sha256_r(msg, 64) == hashlib.sha256(msg).hexdigest(), L
    n_ok += 1
print("self-test: %d random messages (lengths 0..~2000), R=64 == hashlib: OK" % n_ok)
assert sha256_r(b"abc", 64) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

text = open(sys.argv[1]).read()
blocks = re.split(r'\n(?=\[)', text)
npad = ncomp = 0
bad = []
for blk in blocks:
    m = re.match(r'\[(padded|comp)\] label="([^"]*)"', blk.strip())
    if not m:
        continue
    kind, label = m.group(1), m.group(2)
    tgt = re.search(r'target\s*=\s*([0-9a-f]{64})', blk).group(1)
    ok_label = (hashlib.sha256(label.encode()).hexdigest() == tgt)
    if kind == 'padded':
        mh = re.search(r'message_hex\s*=\s*([0-9a-f]+)', blk).group(1)
        msg = bytes.fromhex(mh)
        d19 = sha256_r(msg, 19)
        d18 = sha256_r(msg, 18)
        d20 = sha256_r(msg, 20)
        d64 = sha256_r(msg, 64)
        # also an independent padding check: build block manually
        blk64 = msg + b'\x80' + struct.pack('>Q', 8*len(msg))
        assert len(blk64) == 64
        W = struct.unpack('>16I', blk64)
        d19b = words_to_hex(compress_ff(IV, W, 19))
        ok = ok_label and len(msg) == 55 and d19 == tgt and d19b == tgt and d18 != tgt and d20 != tgt
        print("%-8s %-18s len=%d label_ok=%s d19_ok=%s d18!=t:%s d20!=t:%s d64!=t:%s W13&ff=%02x W14=%x W15=%d  %s" % (
            kind, label, len(msg), ok_label, d19 == tgt and d19b == tgt, d18 != tgt, d20 != tgt, d64 != tgt,
            W[13] & 0xff, W[14], W[15], "OK" if ok else "FAIL"))
        npad += 1
    else:
        ws = re.search(r'W\[0\.\.15\]\s*=\s*([0-9a-f ]+)', blk).group(1).split()
        W = [int(x, 16) for x in ws]
        assert len(W) == 16
        d19 = words_to_hex(compress_ff(IV, W, 19))
        d18 = words_to_hex(compress_ff(IV, W, 18))
        d20 = words_to_hex(compress_ff(IV, W, 20))
        ok = ok_label and d19 == tgt and d18 != tgt and d20 != tgt
        print("%-8s %-18s label_ok=%s c19+IV_ok=%s d18!=t:%s d20!=t:%s  %s" % (
            kind, label, ok_label, d19 == tgt, d18 != tgt, d20 != tgt, "OK" if ok else "FAIL"))
        ncomp += 1
    if not ok:
        bad.append(label)
print("padded witnesses: %d, comp witnesses: %d, failures: %d %s" % (npad, ncomp, len(bad), bad))
