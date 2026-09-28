"""Download the CC0 Poly Haven apartment kit, with integrity checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ASSET = "modular_urban_apartments_facade"
DEST = ROOT / "assets/scenes/photoreal/urban_street/source" / ASSET


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--max-time", "300", "--silent", "--show-error", url, "-o", str(path)], check=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = DEST / "files.json"
    fetch(f"https://api.polyhaven.com/files/{ASSET}", metadata)
    entry = json.loads(metadata.read_text())["blend"]["2k"]["blend"]
    entries = {f"{ASSET}_2k.blend": entry, **entry["include"]}

    def download(item):
        name, info = item
        path = DEST / name
        if not path.resolve().is_relative_to(DEST.resolve()):
            raise ValueError(name)
        def valid(p):
            return p.is_file() and p.stat().st_size == info["size"] and hashlib.md5(p.read_bytes()).hexdigest() == info["md5"]
        if not valid(path):
            partial = path.with_suffix(path.suffix + ".partial")
            fetch(info["url"], partial)
            if not valid(partial):
                raise ValueError(f"Checksum mismatch: {name}")
            partial.replace(path)
        print(f"Verified {name}", flush=True)
        return {"path": name, "url": info["url"], "bytes": info["size"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, entries.items()))
    (DEST / "manifest.json").write_text(json.dumps({"asset": ASSET, "source": f"https://polyhaven.com/a/{ASSET}", "license": "CC0-1.0", "license_url": "https://polyhaven.com/license", "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
