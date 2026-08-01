"""Fetch a slice of the OWT benchmark corpus used across the SuperBPE suite.

The README benchmarks run on ``owt_train.txt`` from the ``stanford-cs336/
owt-sample`` dataset. That gzip is large, so this streams and decompresses
only the first ``--mb`` megabytes into ``--out`` (default ``~/data/
owt_train.txt``) — enough for a local train+eval slice — instead of pulling
the whole file. No dependency beyond the stdlib.

    uv run --no-project benchmarks/superbpe/get_corpus.py --mb 700

Equivalent to the README's ``wget ...owt_train.txt.gz && gunzip`` but
truncated. Skips the download if the output already has at least --mb MB.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
import urllib.error
import urllib.request

URL = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"


def _hf_token() -> str | None:
    """The HF token, via gigatoken's discovery when available, else env."""
    try:
        from gigatoken.gigatoken_rs import get_hf_token

        return get_hf_token()
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _open_with_retry(url: str, tries: int = 5):
    headers = {"User-Agent": "gigatoken-superbpe-bench"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    delay = 5.0
    for attempt in range(1, tries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < tries:
                wait = float(e.headers.get("Retry-After", delay))
                print(f"  HTTP {e.code}; retrying in {wait:.0f}s ({attempt}/{tries})", file=sys.stderr)
                time.sleep(wait)
                delay *= 2
                continue
            raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.expanduser("~/data/owt_train.txt"))
    p.add_argument("--mb", type=float, default=700.0, help="decompressed MB to keep (default: %(default)s)")
    p.add_argument("--url", default=URL)
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    want = int(args.mb * 1e6)
    if os.path.exists(out) and os.path.getsize(out) >= want:
        print(f"{out} already has {os.path.getsize(out) / 1e6:.0f} MB (>= {args.mb} MB); skipping")
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"streaming {args.url}\n  -> {out} (first {args.mb:.0f} MB decompressed)", file=sys.stderr)
    written = 0
    with _open_with_retry(args.url) as resp, gzip.GzipFile(fileobj=resp) as gz, open(out, "wb") as f:
        while written < want:
            chunk = gz.read(min(8 << 20, want - written))
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            print(f"  {written / 1e6:6.0f} MB", end="\r", file=sys.stderr)
    print(f"\nwrote {written / 1e6:.0f} MB -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
