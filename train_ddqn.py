import torch
import numpy as np
import argparse

import gymnasium as gym
import speederbikes_sim
from ddqn.DDQNAgent import Agent
from ddqn.DDQNTrainer import Trainer

# setup all components
def run(verbose:bool=False):
    sim_speedup = 10

    env = gym.make('speederbikes/SpeederBikes-v0', render_mode="rgb_array", observation_mode="flatten",
                    lvl_n_lanes=3, lvl_speed= 200 * sim_speedup, lvl_road_width= 350, 
                    agt_speed= 200 * sim_speedup 
                )
    obs, info = env.reset()
    env.metadata["render_fps"] = 60 * sim_speedup

    agent = Agent(env, "train")

    validation_steps = 10000
    validation_interval = 500000 # means to evaluate and check about every 30 minutes if running on local machine (CPU ryzen 7 5800H, 3.2 Ghz) and GeForce RTX 3070
    trainer = Trainer(agent, update_every=8, epochs=100000, 
                    validation_interval=validation_interval, validation_steps=validation_steps,
                    goal_reward=validation_steps * 1,
                    log_dir="./logs", storage_dir="./trained_models"
                    )
    if verbose:
        print("INFO: starting training")
    trainer.start(verbose)

    env.close()

if __name__ == "__main__":
    parser =argparse.ArgumentParser(prog="train_ddqn.py")
    parser.add_argument('-v', '--verbose', action="count", default=0)

    kwargs = parser.parse_args()
    run(kwargs.verbose)