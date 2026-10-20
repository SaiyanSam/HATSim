"""Example collaborator baseline: reward divided by expected travel + service time.

Uses Euclidean travel time for a *simple baseline*, not a collision-aware ETA.
May reassign current robots if a higher score is available; engine checks ownership.
"""
import numpy as np


def select_actions(sim):
    actions = {}
    for robot in sim.robots:
        if not robot.alive:
            continue
        candidates = [task for task in sim.tasks
                      if task.status in ("available", "assigned", "servicing")
                      and task.owner in (None, robot.id) and task.eligible(robot.kind)]
        if not candidates:
            actions[robot.id] = "keep"
            continue
        def score(task):
            travel = float(np.linalg.norm(robot.position - np.asarray(task.position))) / max(float(robot.spec["cruise_speed"]), 1e-6)
            service = max(0., task.total_work - task.progress) / max(float(task.rates[robot.kind]), 1e-6)
            return (task.reward / max(travel + service, 1e-6), task.id)
        chosen = max(candidates, key=score)
        actions[robot.id] = "keep" if robot.task_id == chosen.id else f"claim:{chosen.id}"
    return actions
