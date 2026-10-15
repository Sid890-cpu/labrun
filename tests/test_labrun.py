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
