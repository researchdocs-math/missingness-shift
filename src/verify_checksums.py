"""Verify the SHA-256 inventory of a released benchmark."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'CHECKSUMS.json').read_text());bad=[]
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
    if bad:raise SystemExit('Checksum mismatch: '+', '.join(bad))
    print(f'PASS: {len(manifest)} files match the release checksums.')
if __name__=='__main__':main()
