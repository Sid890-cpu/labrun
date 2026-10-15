# labrun

Sandboxed, reproducible experiment runner for autonomous research agents.

Agents write and run code. `labrun` runs it isolated, records everything, and lets you replay any run and verify the result is identical.

## Use

```python
from labrun import Runner, Policy

runner = Runner(policy=Policy(timeout_s=10, max_memory_mb=256, block_network=True))
rec = runner.run("import random; print(random.random())", seed=42)
print(rec.run_id, rec.stdout)

new, matches = runner.replay(rec.run_id)   # re-run with same code/seed/inputs
print("reproducible:", matches)
```

CLI:

```
labrun run experiment.py --seed 42
labrun list
labrun replay <run_id>
labrun show <run_id>
```

## What it records

Code and hash, seed, inputs, policy, Python and package versions, stdout/stderr, output hash, duration.

## Policy

Timeout, memory cap, output cap, network block.

## Status

v0.1. Subprocess isolation only, not a hardened sandbox. Do not run hostile code without container isolation.

## Roadmap

Container backend, compute/cost tracking, run diffing, multi-run experiment trees, verification checks for claimed discoveries.
