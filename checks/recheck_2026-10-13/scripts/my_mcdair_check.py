"""Recompute the three padded 17-step preimages posted by McDair (cryptography mailing list,
Thu 28 Mar 2024, subject "SHA-256 challenge", archive page 2024-March/038315.html) with my_sha.py.

The three messages and the challenge digest are copied from that post. For every step count from 1
to 64 the script hashes each message (standard IV, standard padding, feed-forward) and reports where
the digest equals the challenge. It also shows that the challenge is not SHA-256("McDair").

usage: my_mcdair_check.py        (no arguments; standard library only)"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_sha import hash_steps

CHALLENGE = "deb360ae3c1ff7a29f83731b33dcd4bf354a5e80de2dc50370ebf55a14216b85"
EXAMPLES = {
    "example 1": "716bb9216a2038dd92e63067398bf9b01389992ea9b039f8dad7d812cfa881",
    "example 2": ("3568454cf4467cb73b89951b5b0dde9209e02bc22697d6b51e4c92e50554069a"
                  "5361746f736869204e616b616d6f746f2072756c657321"),
    "example 3": ("c38e78ed0a66beb5ac22a275fc99b667ae8d95d0766a002540bf8edb8548c067"
                  "5361746f736869204e616b616d6f746f2072756c657321"),
}

print("challenge digest         :", CHALLENGE)
print("SHA-256(\"McDair\") (64)   :", hashlib.sha256(b"McDair").hexdigest(),
      "(different, so the challenge is not the hash of the name)")
all_ok = True
for name, hx in EXAMPLES.items():
    msg = bytes.fromhex(hx)
    hits = [s for s in range(1, 65) if hash_steps(msg, s) == CHALLENGE]
    h17 = hash_steps(msg, 17)
    same = sum(a == b for a, b in zip(h17, CHALLENGE))
    print(f"{name}: {len(msg)} bytes; 17-step digest {h17[:8]}...{h17[-4:]} "
          f"({same}/64 hex digits equal); step counts 1..64 that equal the challenge: {hits}; "
          f"64-step SHA-256 = {hashlib.sha256(msg).hexdigest()[:16]}... (not the challenge)")
    all_ok &= hits == [17]
print("all three match the challenge at exactly 17 steps:", all_ok)
sys.exit(0 if all_ok else 1)
