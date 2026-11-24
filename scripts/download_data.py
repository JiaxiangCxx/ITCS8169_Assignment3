"""Download the course Google Drive folder file by file, in parallel and resumable.

Usage: python download_data.py --dest data [--workers 8]
Re-running skips files that are already complete.
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gdown

FOLDER_ID = "1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M"


def looks_complete(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 100:
        return False
    with open(path, "rb") as f:
        head = f.read(16)
        if head.startswith(b"\xff\xd8"):  # JPEG: check end marker
            f.seek(-2, os.SEEK_END)
            return f.read(2) == b"\xff\xd9"
    return not head.lstrip().lower().startswith(b"<!doctype html") and not head.lstrip().lower().startswith(b"<html")


def fetch(item, dest, tries=6):
    out = os.path.join(dest, item.path)
    if looks_complete(out):
        return item.path, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    for attempt in range(tries):
        try:
            gdown.download(id=item.id, output=tmp, quiet=True)
            if looks_complete(tmp):
                os.replace(tmp, out)
                return item.path, "ok"
            err = "incomplete/html response"
        except Exception as e:  # quota, network, etc.
            err = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(min(60, 2 ** attempt + random.random() * 2))
    if os.path.exists(tmp):
        os.remove(tmp)
    return item.path, "FAIL " + err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True)
    items = [i for i in items if i.id]
    print(f"{len(items)} files listed", flush=True)

    done = fail = 0
    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, i, args.dest) for i in items]
        for n, fut in enumerate(as_completed(futs), 1):
            path, status = fut.result()
            if status.startswith("FAIL"):
                fail += 1
                failed.append(path)
                print(f"{status}  {path}", flush=True)
            else:
                done += 1
            if n % 100 == 0 or n == len(items):
                print(f"[{n}/{len(items)}] ok={done} fail={fail}", flush=True)
    if failed:
        with open(os.path.join(args.dest, "failed.txt"), "w") as f:
            f.write("\n".join(failed) + "\n")
    print("DONE" if not failed else f"DONE with {len(failed)} failures (re-run to retry)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
