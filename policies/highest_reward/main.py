"""B0: naive highest-reward greedy with distance and ID tie breakers.

No task stealing: the engine resolves simultaneous claims deterministically.
"""
import numpy as np


def select_actions(sim):
    actions = {}
    tasks = [t for t in sim.tasks if t.status in ("available", "assigned", "servicing")]
    for robot in sim.robots:
        if not robot.alive:
            continue
        eligible = [t for t in tasks if t.eligible(robot.kind)
                    and t.owner in (None, robot.id)]
        if not eligible:
            actions[robot.id] = "keep"
            continue
        chosen = max(eligible, key=lambda t: (
            t.reward, -np.linalg.norm(robot.position - np.asarray(t.position)), t.id))
        actions[robot.id] = "keep" if robot.task_id == chosen.id else f"claim:{chosen.id}"
    return actions
