# HATSim policies

One policy per directory, with a `main.py` exposing `select_actions(sim) -> dict[str, str]`.

```text
policies/
  highest_reward/main.py    # Existing B0, unchanged logic
  sticky/main.py            # Existing B1, unchanged logic
  reward_per_task/main.py   # Example agent/task reward-rate baseline
  your_policy/
    __init__.py
    main.py                 # Add your code here
    helpers.py              # Optional supporting modules
```

Supported actions for each robot ID: `"keep"`, `"release"`, `"claim:<task_id>"`.
Policies are called once per allocation period (default 2 seconds). The simulator
resolves simultaneous claims and rejects already-owned tasks. Policies should
avoid modifying simulator state directly; return proposed actions only.

To add a policy:

```python
# policies/my_policy/main.py
def select_actions(sim):
    return {robot.id: "keep" for robot in sim.robots if robot.alive}
```

```bash
python -m hatsim.cli --list-policies
python -m hatsim.cli --policy reward_per_task --task-sequence 1 --seed 42 --timeout 300 --headless
python -m hatsim.cli --policy my_policy --task-sequence 1 --seed 42 --timeout 300 --headless
```

**Research note:** The current `sim` object exposes globally generated tasks,
true owners and full robot positions. This API is suitable for **full-information
baselines**, but it is **not yet a decentralized/partial-observation API**. For
communication-constrained PACT, introduce a read-only per-agent observation
adapter and prevent policies from accessing ground truth directly.

The greedy policies deliberately retain their original behavior, including
repeated contention for the same attractive task. The engine resolves claims,
but does not immediately offer a losing robot its second choice.
