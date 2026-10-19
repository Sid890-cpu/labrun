import tempfile

from labrun import Runner, Policy


def mk():
    return Runner(store=tempfile.mkdtemp())


def test_run_and_replay_match():
    r = mk()
    rec = r.run("import random\nprint(random.random())", seed=7)
    assert rec.exit_code == 0
    new, ok = r.replay(rec.run_id)
    assert ok and new.parent_run_id == rec.run_id


def test_different_seed_differs():
    r = mk()
    a = r.run("import random\nprint(random.random())", seed=1)
    b = r.run("import random\nprint(random.random())", seed=2)
    assert a.output_sha256 != b.output_sha256


def test_timeout():
    r = mk()
    rec = r.run("while True: pass", policy=Policy(timeout_s=1))
    assert rec.timed_out


def test_network_blocked():
    r = mk()
    rec = r.run("import socket\nsocket.create_connection(('example.com', 80))")
    assert rec.exit_code != 0 and "blocked" in rec.stderr


def test_memory_limit():
    r = mk()
    rec = r.run("x = bytearray(2 * 1024**3)", policy=Policy(max_memory_mb=128))
    assert rec.exit_code != 0


def test_artifacts_captured_and_replayed():
    r = mk()
    rec = r.run("open('out.txt','w').write('hello')\nprint('done')")
    assert "out.txt" in rec.artifacts
    new, ok = r.replay(rec.run_id)
    assert ok


import json as _json
from pathlib import Path as _Path


def _tamper(r, run_id, field, value):
    f = _Path(r.store) / f"{run_id}.json"
    d = _json.loads(f.read_text())
    d[field] = value
    f.write_text(_json.dumps(d))


def test_replay_mismatch_on_each_field():
    cases = {
        "exit_code": 99,
        "timed_out": True,
        "output_sha256": "0" * 64,
        "stderr_sha256": "0" * 64,
        "artifacts": {"fake.txt": "0" * 64},
    }
    for field, bad in cases.items():
        r = mk()
        rec = r.run("open('out.txt','w').write('x')\nprint('hi')")
        _tamper(r, rec.run_id, field, bad)
        _, ok = r.replay(rec.run_id)
        assert not ok, f"replay should be MISMATCH when {field} differs"


def test_nondeterministic_run_mismatches():
    r = mk()
    rec = r.run("import time\nprint(time.time())")
    _, ok = r.replay(rec.run_id)
    assert not ok


def test_diff_reports_stdout_line():
    from labrun.runner import diff_runs
    r = mk()
    rec = r.run("print('a')\nprint('b')")
    bad = r.load(rec.run_id)
    bad.stdout = "a\nX\n"
    bad.output_sha256 = "x"
    reasons = diff_runs(r.load(rec.run_id), bad)
    assert any("stdout differs at line 2" in x for x in reasons)
