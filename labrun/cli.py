import argparse
import sys
import time
from pathlib import Path

from .runner import Runner, Policy


def main(argv=None):
    ap = argparse.ArgumentParser(prog="labrun", description="Sandboxed, reproducible experiment runner")
    ap.add_argument("--store", default=".labrun")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a Python experiment file in the sandbox")
    r.add_argument("file")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--timeout", type=float, default=30)
    r.add_argument("--memory-mb", type=int, default=512)
    r.add_argument("--allow-network", action="store_true")

    p = sub.add_parser("replay", help="re-run a past run and verify identical output")
    p.add_argument("run_id")

    sub.add_parser("list", help="list recorded runs")

    s = sub.add_parser("show", help="show one run")
    s.add_argument("run_id")

    a = ap.parse_args(argv)
    runner = Runner(a.store)

    if a.cmd == "run":
        pol = Policy(timeout_s=a.timeout, max_memory_mb=a.memory_mb, block_network=not a.allow_network)
        rec = runner.run(Path(a.file).read_text(), seed=a.seed, policy=pol)
        print(rec.stdout, end="")
        if rec.stderr:
            print(rec.stderr, end="", file=sys.stderr)
        status = "TIMEOUT" if rec.timed_out else f"exit {rec.exit_code}"
        print(f"\n[labrun] run_id={rec.run_id} {status} {rec.duration_s}s", file=sys.stderr)
        return rec.exit_code if rec.exit_code and rec.exit_code > 0 else 0

    if a.cmd == "replay":
        new, ok = runner.replay(a.run_id)
        print(f"[labrun] replay {a.run_id} -> {new.run_id}: {'MATCH' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    if a.cmd == "list":
        for rec in runner.list():
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.created_at))
            flag = "TIMEOUT" if rec.timed_out else f"exit={rec.exit_code}"
            par = f" (replay of {rec.parent_run_id})" if rec.parent_run_id else ""
            print(f"{rec.run_id}  {t}  {flag}  {rec.duration_s}s  seed={rec.seed}{par}")
        return 0

    if a.cmd == "show":
        rec = runner.load(a.run_id)
        print(f"run_id:   {rec.run_id}\ncode_sha: {rec.code_sha256}\nseed:     {rec.seed}")
        print(f"exit:     {rec.exit_code}  timed_out={rec.timed_out}  {rec.duration_s}s")
        print(f"output:   {rec.output_sha256}\npolicy:   {rec.policy}\n--- code ---\n{rec.code}\n--- stdout ---\n{rec.stdout}")
        if rec.stderr:
            print(f"--- stderr ---\n{rec.stderr}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
