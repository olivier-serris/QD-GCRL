import itertools
import matplotlib.pyplot as plt
import numpy as np
import jax
import gym
from xpag.agents import TD3

#### GRID UTILS ####


class GoalGrid:
    """
    Partition a [-1,1] continuous space into a [width,height] grid.
    """

    # Warning: class is not tested with non-squared grids.

    def __init__(self, space_min, space_max, grid_width, grid_height) -> None:
        self.space_min = np.array(space_min)
        self.space_max = np.array(space_max)
        self.grid_width = grid_width
        self.grid_height = grid_height

    def get_cells_center(self) -> np.ndarray:
        """
        Returns a list containing the center of each cell in the 2D grid.
        """
        cell_width = (self.space_max - self.space_min) / np.array(
            [self.grid_width, self.grid_height]
        )
        start = self.space_min + (cell_width / 2)
        end = self.space_max - (cell_width / 2)
        x = np.linspace(start[0], end[0], self.grid_width)
        y = np.linspace(start[1], end[1], self.grid_height)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx, yy], axis=-1).reshape(-1, 2)

    @property
    def n_cells(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def grid_shape(self) -> int:
        return self.grid_width, self.grid_height

    def cell_coords(self, pos):
        """
        Given a continuous vector pos, returns the cell row and column where this vector is located.
        """
        coords = np.floor(
            np.multiply(
                (pos - self.space_min) / (self.space_max - self.space_min),
                np.array([self.grid_height, self.grid_width]),
            )
        )
        # in case pos is exactly on the border of the grid, we need to clip the coords to be in the grid.
        coords = np.clip(
            coords, 0, np.array([self.grid_height, self.grid_width]) - 1
        ).astype(int)
        return coords

    def get_is_in_cell_function(self):
        """Returns  function that returns true if the agent is in the target cell,false otherwise.
        achieved goal and desired goal are supposed to be 2D continious positions."""

        def is_in_goal_cell(achieved_goal: np.ndarray, desired_goal: np.ndarray, *args):
            in_target_cell = (
                self.cell_coords(achieved_goal) == self.cell_coords(desired_goal)
            ).all(axis=1)

            return in_target_cell

        return is_in_goal_cell

    def get_cell_reward_function(self):
        """Returns a reward function that returns 0 if the agent is in the target cell,-1 otherwise.
        achieved goal and desired goal are supposed to be 2D continious positions."""
        is_in_goal_cell = self.get_is_in_cell_function()

        def reward_fn(achieved_goal: np.ndarray, desired_goal: np.ndarray, *args):
            return is_in_goal_cell(achieved_goal, desired_goal, *args) - 1

        return reward_fn

    def cell_id_to_cell_coords(self, cell_id):
        return np.unravel_index(cell_id, self.grid_shape)

    def cell_coords_to_cell_id(self, cell_coords):
        return np.ravel_multi_index(cell_coords, self.grid_shape)


def save_grid(scores, labels, show=False, path=None):
    fig, ax = plt.subplots()

    # for i in range(len(scores)):
    #     for j in range(len(scores)):
    #         ax.text(j, i, labels[i, j], ha="center", va="center", color="w")
    ax.set_title(path)
    ax.set_xticks(np.arange(len(scores)))
    ax.set_yticks(np.arange(len(scores)))
    ax.set_xticklabels(np.linspace(-1, 1, len(scores)))
    ax.set_yticklabels(np.linspace(-1, 1, len(scores)))
    im = ax.imshow(scores, vmin=0, vmax=1)
    plt.colorbar(
        im,
    )
    # save plot :
    if show:
        plt.show()
    if path:
        plt.savefig(path)
    plt.close()


class EndEpisodeDistanceRewardWrapper(gym.Wrapper):
    def step(self, action):
        def dist(x, y):
            return np.linalg.norm(x - y, axis=-1)

        obs, reward, terminated, truncated, info = super().step(action)

        # override terminated condition :
        terminated = self.steps == self.max_episode_steps
        terminated = np.expand_dims(terminated, axis=-1)
        self.done = np.logical_or(terminated, truncated)
        reward = np.where(
            self.steps == self.max_episode_steps,
            -dist(obs["desired_goal"], obs["achieved_goal"]),
            0,
        )
        reward = np.expand_dims(reward, axis=-1)

        return obs, reward, terminated, truncated, info


#### NN weights operations ####


def get_weights(agent: TD3):
    """Get the NN of the actor of an agent as a 1D array"""
    params_tree = agent.policy_params
    params, unravel_fn = jax.flatten_util.ravel_pytree(params_tree)
    return params


def set_weights(agent: TD3, new_weights):
    """Set the weights of the NN of an agent from a 1D array"""
    params, unravel_fn = jax.flatten_util.ravel_pytree(new_weights)
    agent.policy_params = unravel_fn(new_weights)


#### LOGS ####


def score_matrix_image(scores, title):
    """
    Returns a matplotlib figure containing the plotted scores.
    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(scores, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # Use white text if squares are dark; otherwise black.
    threshold = scores.max() / 2.0
    for i, j in itertools.product(range(scores.shape[0]), range(scores.shape[1])):
        color = "white" if scores[i, j] > threshold else "black"
        plt.text(j, i, scores[i, j], horizontalalignment="center", color=color)

    return figure


def logs(goal_grid, agents, metrics, writer, n_step_done, gen):
    """
    scores shape : [n_agents, n_cells]
    """

    # SR metrics :
    best_sr = metrics["sr"].max(axis=0)
    writer.add_scalar("SR/mean", best_sr.mean(), global_step=n_step_done)
    best_sr_image = np.flipud(
        best_sr.reshape(goal_grid.grid_height, goal_grid.grid_width)
    )  # should be the correct orientation

    writer.add_figure(
        "SR/grid",
        score_matrix_image(best_sr_image, f"SR grid"),
        global_step=n_step_done,
    )
    # CR (cumulated reward) metrics :
    best_cr = metrics["cr"].max(axis=0)
    writer.add_scalar("CR/mean", best_cr.mean(), global_step=n_step_done)
    best_cr_image = np.flipud(
        best_cr.reshape(goal_grid.grid_width, goal_grid.grid_height)
    )
    writer.add_figure(
        "CR/grid",
        score_matrix_image(best_cr_image, f"CR grid"),
        global_step=n_step_done,
    )

    # pop metrics :
    writer.add_scalar("Pop/pop_size", len(agents), global_step=n_step_done)

    best_pol_per_cell = (
        metrics["cr"]
        .argmax(axis=0)
        .reshape(1, goal_grid.grid_width, goal_grid.grid_height)
    )
    best_pol_image = np.flipud(
        best_pol_per_cell.reshape(goal_grid.grid_width, goal_grid.grid_height)
    )
    writer.add_figure(
        "Pop/best_pol_per_cell",
        score_matrix_image(best_pol_image, f"SR grid"),
        global_step=n_step_done,
    )

    print(f"{gen} : mean best grid (sr,cr) : ", best_sr.mean(), best_cr.mean())
