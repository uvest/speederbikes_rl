import os
import torch
import numpy as np
# from torch.utils.tensorboard.writer import SummaryWriter

from resources.agent import Agent

class Trainer():
    def __init__(self, agent:Agent, epochs:int|None=10000, max_steps:int=100000, update_every:int=32,
                 validation_interval:int=500000, validation_steps:int=10000,
                 goal_reward:float|None=None,
                 seed:int=4,
                 log_dir:str|None="./logs",
                 storage_dir:str="./trained_models"
                 ) -> None:
        """_summary_

        Args:
            agent (Agent): The agent to train. Must inherit from agent.Agent
            epochs (int | None, optional):if provided training will stop after epochs. if not goal_reward is required. Defaults to 10000.
            max_steps (int, optional): _description_. Defaults to 100000.
            update_every (int, optional): _description_. Defaults to 32.
            validation_interval (int, optional): _description_. Defaults to 500000.
            validation_steps (int, optional): _description_. Defaults to 10000.
            goal_reward (float | None, optional): If not None, training continues until the goal reward is reached in one validation or epochs is reached if provided. Defaults to None.
            log_dir (str | None, optional): _description_. Defaults to "./logs".
            storage_dir (str, optional): _description_. Defaults to "./trained_models".
        """
        assert (epochs is not None) or (goal_reward is not None)
        assert (storage_dir is not None)

        self.agent = agent

        self.epochs = epochs
        self.max_steps = max_steps
        self.update_every = update_every
        self.validation_interval = validation_interval # validate after (validation_interval * max_steps) steps have been taken
        self.validation_steps = validation_steps
        self.goal_reward = goal_reward

        self.seed = seed

        self.log_dir = log_dir
        self.storage_dir = storage_dir

        # make sure storage and log directory exists:
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)

        # set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # define device to run on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===
        if self.epochs is None:
            self.train_infinitely = True
        else:
            self.train_infinitely = False

        # ...

    def start(self):
        """Start the training (blocking execution).
        """
        if self.train_infinitely:
            self.epochs = 1 # keep self.epochs always one higher than the epoch counter
        # train for epochs epochs
        steps = 0
        epoch = 0
        while epoch < self.epochs:
            epoch_reward = 0
            
            # ...

            state, info = self.agent.env.reset()
            for step in range(self.max_steps):
                steps += 1
                # let agent select action
                action = self.agent.selectAction(state)

                # take step with this action
                next_state, reward, terminated, truncated, info = self.agent.env.step(action)

                # store reward
                epoch_reward += reward

                # calculate learning error
                error = ...
                
                # update agent
                if steps % self.update_every == 0:
                    self.agent.update(error)

                # proceed to next state
                state = next_state

                # end episode if next statae is terminal or environment is truncated (stopped because of time passed)
                if terminated or truncated:
                    break


            if steps % self.validation_interval == 0:
                # evaluate
                score = self.evaluate()
                # store agent
                self.agent.save(self.storage_dir, steps, ...)
                # check if performance requirements are met
                if self.goal_reward is not None:
                    if score >= self.goal_reward:
                        break
            epoch += 1
            if self.train_infinitely:
                self.epochs += 1

                
    def evaluate(self) -> float:
        """Evaluate agent
        """
        eval_reward = 0
        state, info = self.agent.env.reset()

        for step in range(self.max_steps):
            # let agent select action
            action = self.agent.selectAction(state)

            # take step with this action
            next_state, reward, terminated, truncated, info = self.agent.env.step(action)

            # store reward
            eval_reward += reward

            # end episode if needed
            if terminated or truncated:
                break
                
            # proceed
            state = next_state
        
        return eval_reward
