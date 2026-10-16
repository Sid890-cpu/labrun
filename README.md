# labrun

Sandboxed, reproducible experiment runner for autonomous research agents.

Agents write and run code. `labrun` runs it isolated, records everything, and lets you replay any run and verify the result is identical.

## Quickstart

```
pip install git+https://github.com/Sid890-cpu/labrun
labrun run experiment.py --seed 42
labrun list
labrun replay <run_id>      # prints MATCH if output and files are identical
```

Linux or macOS. On Windows, use WSL or Google Colab.

## Use from Python

```python
from labrun import Runner, Policy

runner = Runner(policy=Policy(timeout_s=10, max_memory_mb=512, block_network=True))
rec = runner.run("import random; print(random.random())", seed=42)
print(rec.run_id, rec.stdout)

new, matches = runner.replay(rec.run_id)   # same code, seed, inputs
print("reproducible:", matches)
```

## Who it's for

Teams building autonomous research agents that run experiments on their own. When an agent claims a result, you need to know it is real and repeatable, and you should not run its untrusted code on your machine.

## What it records

Code and hash, seed, inputs, policy, Python and package versions, stdout and stderr, output hash, duration, and every file the experiment writes (saved and hashed).

## Policy

Timeout, memory cap, output cap, network block.

## Status

v0.1. Subprocess isolation only, not a hardened sandbox. Do not run hostile code without container isolation. Works with numpy and other installed libraries.

## Roadmap

Container backend, compute and cost tracking, run diffing, multi-run experiment trees, verification checks for claimed discoveries.

Feedback wanted: open an issue and say what breaks when your agent runs experiments.
