"""B1: keep an existing commitment; otherwise claim highest-reward eligible task."""
from policies.highest_reward.main import select_actions as greedy_actions


def select_actions(sim):
    proposed = greedy_actions(sim)
    for robot in sim.robots:
        if robot.alive and robot.task_id is not None:
            proposed[robot.id] = "keep"
    return proposed
