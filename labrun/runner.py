"""Core: run an experiment in an isolated subprocess, record everything, replay exactly."""
from __future__ import annotations

import hashlib
import shutil
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Injected before experiment code: seeds RNGs and (optionally) blocks network.
_PRELUDE = '''
import os, random, socket
_seed = int(os.environ.get("LABRUN_SEED", "0"))
random.seed(_seed)
try:
    import numpy as _np
    _np.random.seed(_seed)
except Exception:
    pass
if os.environ.get("LABRUN_BLOCK_NET") == "1":
    def _blocked(*a, **k):
        raise PermissionError("labrun policy: network access blocked")
    socket.socket.connect = _blocked
    socket.getaddrinfo = _blocked
'''


@dataclass
class Policy:
    timeout_s: float = 30.0
    max_memory_mb: int = 2048
    max_output_kb: int = 256
    block_network: bool = True


@dataclass
class RunRecord:
    run_id: str
    created_at: float
    code: str
    code_sha256: str
    seed: int
    inputs: dict
    policy: dict
    env: dict
    exit_code: Optional[int] = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    output_sha256: str = ""
    duration_s: float = 0.0
    parent_run_id: Optional[str] = None
    artifacts: dict = field(default_factory=dict)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _env_snapshot() -> dict:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=20,
        ).stdout.splitlines()
    except Exception:
        freeze = []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": sorted(freeze),
    }


class Runner:
    def __init__(self, store: str = ".labrun", policy: Optional[Policy] = None):
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.policy = policy or Policy()

    # ---- execute -------------------------------------------------------
    def run(self, code: str, inputs: Optional[dict] = None, seed: int = 0,
            policy: Optional[Policy] = None, _parent: Optional[str] = None,
            _env: Optional[dict] = None) -> RunRecord:
        policy = policy or self.policy
        inputs = inputs or {}
        rec = RunRecord(
            run_id=uuid.uuid4().hex[:12],
            created_at=time.time(),
            code=code,
            code_sha256=_sha(code),
            seed=seed,
            inputs=inputs,
            policy=asdict(policy),
            env=_env if _env is not None else _env_snapshot(),
            parent_run_id=_parent,
        )

        with tempfile.TemporaryDirectory(prefix="labrun_") as work:
            script = Path(work) / "exp.py"
            script.write_text(_PRELUDE + "\n" + code)
            (Path(work) / "inputs.json").write_text(json.dumps(inputs))

            env = {
                "PATH": os.environ.get("PATH", ""),
                "LABRUN_SEED": str(seed),
                "LABRUN_BLOCK_NET": "1" if policy.block_network else "0",
                "LABRUN_INPUTS": str(Path(work) / "inputs.json"),
                "PYTHONHASHSEED": str(seed),
            }

            def limits():
                mem = policy.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                os.setsid()

            start = time.time()
            try:
                p = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    cwd=work, env=env, capture_output=True, text=True,
                    timeout=policy.timeout_s, preexec_fn=limits,
                )
                rec.exit_code = p.returncode
                out, err = p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                rec.timed_out = True
                rec.exit_code = -9
                out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            rec.duration_s = round(time.time() - start, 4)

            # keep files the experiment wrote (results, plots, checkpoints)
            art_dir = self.store / rec.run_id / "artifacts"
            for f in sorted(Path(work).rglob("*")):
                if f.is_file() and f.name not in ("exp.py", "inputs.json"):
                    rel = f.relative_to(work)
                    dest = art_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    rec.artifacts[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()

        cap = policy.max_output_kb * 1024
        rec.stdout, rec.stderr = out[:cap], err[:cap]
        rec.output_sha256 = _sha(rec.stdout)
        self._save(rec)
        return rec

    # ---- replay --------------------------------------------------------
    def replay(self, run_id: str) -> tuple[RunRecord, bool]:
        """Re-run a past experiment with identical code/seed/inputs. Returns (new record, matches)."""
        orig = self.load(run_id)
        new = self.run(
            orig.code, orig.inputs, orig.seed,
            policy=Policy(**orig.policy), _parent=orig.run_id, _env=orig.env,
        )
        ok = new.output_sha256 == orig.output_sha256 and new.artifacts == orig.artifacts
        return new, ok

    # ---- storage -------------------------------------------------------
    def _save(self, rec: RunRecord) -> None:
        (self.store / f"{rec.run_id}.json").write_text(json.dumps(asdict(rec), indent=2))

    def load(self, run_id: str) -> RunRecord:
        data = json.loads((self.store / f"{run_id}.json").read_text())
        return RunRecord(**data)

    def list(self) -> list[RunRecord]:
        recs = [RunRecord(**json.loads(p.read_text())) for p in self.store.glob("*.json")]
        return sorted(recs, key=lambda r: r.created_at)
