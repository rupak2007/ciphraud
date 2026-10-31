"""Record a command's real stdout as an asciicast v2 file, using the real wall-clock time of each output chunk.

No output is edited, generated or re-timed: each event is the bytes the process actually wrote and the time they arrived.

    python -m demo.record_cast --out site/assets/lr_roundtrip.cast --log site/assets/lr_roundtrip.log -- <command...>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command

    header = {"version": 2, "width": 150, "height": 42, "timestamp": int(time.time()), "title": "Ciphraud: encrypted LR round trip (real run)", "env": {"TERM": "xterm-256color"}}
    events, plain = [], bytearray()
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    assert proc.stdout is not None
    while True:
        chunk = os.read(proc.stdout.fileno(), 4096)
        if not chunk:
            break
        events.append([round(time.perf_counter() - start, 6), "o", chunk.decode("utf-8", "replace").replace("\n", "\r\n")])
        plain += chunk
    rc = proc.wait()
    out, log = Path(args.out), Path(args.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([json.dumps(header)] + [json.dumps(e) for e in events]) + "\n", encoding="utf-8", newline="\n")
    log.write_bytes(bytes(plain))
    print(f"recorded {len(events)} events, {time.perf_counter() - start:.1f} s, exit code {rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
