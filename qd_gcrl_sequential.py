import os

# this line prevents XLA from preallocating ~all the memory
# it is needed to run the code on a machine where the GPU memory not big enough
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from xpag.agents.agent import Agent
from xpag.agents import TD3
from xpag.buffers import Buffer, DefaultEpisodicBuffer
from xpag.tools.eval import single_rollout_eval
from xpag.samplers import HER
from xpag.setters.setter import Setter
from xpag.tools.utils import get_datatype, datatype_convert, hstack, logical_or
from xpag.wrappers import gym_vec_env
import math
import numpy as np
import gym_gmazes  # must be imported to register the environment
from typing import Dict, Any, Union, List, Optional, Callable
import hydra
from omegaconf import OmegaConf
import random
from torch.utils.tensorboard import SummaryWriter

# Current project imports
from utils import GoalGrid, logs, get_weights, set_weights
from setters import GoalGridSetter


def single_agent_rollout(
    env,
    env_info,
    master_rng,
    agent,
    noise,
    setter,
    additional_step_keys,
    buffer,
    episodic_buffer,
    goal_grid,
):
    """
    Rollout for a single agent. Save the data in the replay buffer.
    Returns the scores of the agent on all the cells of the grid.
    """

    assert env_info["num_envs"] == goal_grid.n_cells

    # reset envs
    reset_obs, reset_info = env.reset(seed=master_rng.randint(1e9))
    env_datatype = get_datatype(
        reset_obs if not env_info["is_goalenv"] else reset_obs["observation"]
    )
    observation, _ = setter.reset(env, reset_obs, reset_info)

    cumulated_reward = np.zeros(env_info["num_envs"])
    trajectory_metrics = {
        "sr": np.ones(env_info["num_envs"]) * float("nan"),
        "cr": np.ones(env_info["num_envs"]) * float("nan"),
    }

    # rollout
    for i in range(env_info["max_episode_steps"]):
        action_info = {}
        action = agent.select_action(
            hstack(observation["observation"], observation["desired_goal"]),
            eval_mode=not noise,
        )
        if isinstance(action, tuple):
            action_info = action[1]
            action = action[0]

        action = datatype_convert(action, env_datatype)

        next_observation, reward, terminated, truncated, info = setter.step(
            env, observation, action, action_info, *env.step(action)
        )
        cumulated_reward += reward.flatten()

        # truncate unfinished trajecories :
        if i == env_info["max_episode_steps"] - 1:
            truncated = np.ones_like(truncated)

        step = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": next_observation,
            "is_success": info["is_success"],
        }

        if additional_step_keys is not None:
            for a_s_key in additional_step_keys:
                if a_s_key in info:
                    step[a_s_key] = info[a_s_key]

        # gather data in the replay buffer :
        buffer.insert(step)

        observation = next_observation
        done = logical_or(terminated, truncated)

        if done.max():
            # use store_done() if the buffer is an episodic buffer
            if episodic_buffer:
                buffer.store_done(done)
            observation, _, _ = setter.reset_done(
                env,
                *env.reset_done(done, seed=master_rng.randint(1e9)),
                done,
            )
            for goal in np.argwhere(done.flatten()):
                if math.isnan(trajectory_metrics["cr"][goal]):
                    trajectory_metrics["sr"][goal] = info["is_success"][goal]
                    trajectory_metrics["cr"][goal] = cumulated_reward[goal]
            cumulated_reward = np.where(done.flatten(), 0, cumulated_reward.flatten())

    return trajectory_metrics


def evaluate_all_agents(
    env,
    env_info,
    master_rng,
    agents,
    noise,
    setter,
    additional_step_keys,
    buffer,
    episodic_buffer,
    goal_grid,
):
    metric_per_agent = []
    for agent in agents:
        metrics = single_agent_rollout(
            env,
            env_info,
            master_rng,
            agent,
            noise,
            setter,
            additional_step_keys,
            buffer,
            episodic_buffer,
            goal_grid,
        )
        metric_per_agent.append(metrics)
    # from List of dict to  dict of np array :
    # metric per agents has the format [{metric1 : score_agent1, metric2 : score_agent1},{metric1 : score_agent2, metric2 : score_agent2} ...]
    # merge metrics has the format {metric1 : [score_agent1, score_agent2], metric2 :[score_agent1, score_agent2], ...]}
    merged_metrics = {}
    for key in metric_per_agent[0].keys():
        merged_metrics[key] = np.stack(
            metric_per_agent[i][key] for i in range(len(agents))
        )

    return merged_metrics


def update_population(agents, scores):
    """Filter the agents based on their grid scores,
    An agent is kept only if it has a at least one score better than all other on a specific cell of the grid.
    """
    best_scores = scores.max(axis=0)
    filtered_agents = []
    for i, agent in enumerate(agents):
        if np.any(scores[i] >= best_scores):
            filtered_agents.append(agent)

    assert len(filtered_agents) > 0
    return filtered_agents


def update_policies(agents, buffer, batch_size, gd_steps_per_step, n_step):
    for agent in agents:
        for _ in range(max(gd_steps_per_step * n_step, 1)):
            _ = agent.train_on_batch(buffer.sample(batch_size))


def learn(
    env,
    env_info: Dict[str, Any],
    agents: Agent,
    buffer: Buffer,
    setter: Setter,
    goal_grid: GoalGrid,
    *,
    noise=False,
    n_generations=100,
    batch_size: int = 256,
    gd_steps_per_step: int = 1,
    additional_step_keys: Optional[List[str]] = None,
    seed: Optional[int] = None,
):
    writer = SummaryWriter("runs/")

    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    # seed action_space sample
    env_info["action_space"].seed(master_rng.randint(1e9))

    n_step_per_pol = env_info["max_episode_steps"] * env_info["num_envs"]

    episodic_buffer = True if hasattr(buffer, "store_done") else False
    score_by_agent = {id(agent): [] for agent in agents}

    for gen in range(n_generations):
        n_step_done = n_step_per_pol * len(agents) * gen
        metrics = evaluate_all_agents(
            env,
            env_info,
            master_rng,
            agents,
            noise,
            setter,
            additional_step_keys,
            buffer,
            episodic_buffer,
            goal_grid,
        )
        for agent, metric in zip(agents, metrics):
            score_by_agent[id(agent)].append(metric)

        agents = update_population(agents, metrics["cr"])
        update_policies(agents, buffer, batch_size, gd_steps_per_step, n_step_per_pol)
        logs(goal_grid, agents, metrics, writer, n_step_done, gen)


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="large_config.yaml")
def launch(hydra_config):

    cfg = OmegaConf.to_container(hydra_config, resolve=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    grid_size = cfg["env"]["grid_size"]
    goal_grid = GoalGrid([-1.0, -1.0], [1.0, 1.0], grid_size[0], grid_size[1])

    # setup env :
    num_envs = goal_grid.n_cells

    def set_up_env_wrappers(env):
        env.max_episode_steps = cfg["algo"]["step_per_rollout"]
        if cfg["algo"]["reward_type"] == "sparse":
            pass  # this is the default reward system
        elif cfg["algo"]["reward_type"] == "cell":
            env.set_reward_function(goal_grid.get_cell_reward_function())
            env.set_success_function(goal_grid.get_is_in_cell_function())
        else:
            raise Exception("reward type not implemented")
        if "walls" in cfg["env"]:
            env.set_walls(cfg["env"]["walls"])
        # scale action :
        env.action_scale = cfg["env"]["action_scale"]
        return env

    env, eval_env, env_info = gym_vec_env(
        cfg["env"]["name"], num_envs, wrap_function=set_up_env_wrappers
    )

    # create agents :
    def get_agent():
        seed = random.randint(0, 1e9)
        agent = TD3(
            env_info["observation_dim"] + env_info["desired_goal_dim"],
            env_info["action_dim"],
            {**cfg["algo"]["RL"]["params"], **{"seed": seed}},
        )
        return agent

    agents = [get_agent() for _ in range(cfg["algo"]["start_pop_size"])]

    sampler = HER(env.compute_reward)
    buffer = DefaultEpisodicBuffer(
        max_episode_steps=env_info["max_episode_steps"],
        buffer_size=cfg["algo"]["replay_buffer_size"],
        sampler=sampler,
    )
    setter = GoalGridSetter(goal_grid)

    learn(
        env=env,
        env_info=env_info,
        agents=agents,
        buffer=buffer,
        setter=setter,
        goal_grid=goal_grid,
        noise=cfg["algo"]["noise"],
        n_generations=cfg["algo"]["n_generations"],
        batch_size=cfg["algo"]["batch_size"],
        gd_steps_per_step=cfg["algo"]["gd_steps_per_step"],
        additional_step_keys=None,
        seed=cfg["seed"],
    )

    ## save trajectory visualization after learning:
    for agent_i, agent in enumerate(agents):
        for goal_i, pos in enumerate(goal_grid.get_cells_center()):
            single_rollout_eval(
                goal_i,
                eval_env,
                env_info,
                agent,
                setter,
                save_dir=f"trajectory/agent{agent_i}",
                plot_projection=lambda x: x,
                save_episode=True,
            )


if __name__ == "__main__":
    launch()
