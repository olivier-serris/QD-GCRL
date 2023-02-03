from xpag.setters import Setter, DefaultSetter
from utils import GoalGrid
import numpy as np
import itertools


class GoalGridSetter(DefaultSetter):
    def __init__(self, goal_grid: GoalGrid):
        super().__init__()
        self.goal_grid = goal_grid
        self.grid_centers = self.goal_grid.get_cells_center()
        self.goal_cycle = itertools.cycle(self.grid_centers)

    def reset(self, env, observation, info, eval_mode=False):
        if not eval_mode:
            goal = self.grid_centers
        else:
            goal = np.expand_dims(next(self.goal_cycle), axis=0)
        env.set_goal(goal)
        observation["desired_goal"] = goal
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        if not eval_mode:
            goal = np.where(done, self.grid_centers, observation["desired_goal"])
        else:
            goal = np.expand_dims(next(self.goal_cycle), axis=0)
        env.set_goal(goal)
        observation["desired_goal"] = goal

        return observation, info, done
