"""Verify the retained numerical evidence and published tables."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
digests = json.loads((ROOT/'results/SHA256.json').read_text())
for name, expected in digests.items():
    actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'File checksum mismatch: {name}')
print(f'Verified {len(digests)} retained files.')
