import torch
import numpy as np
import pandas as pd
import datetime as dt
import json
import os
import sys
# from torch.utils.tensorboard.writer import SummaryWriter

from resources.trainer import Trainer as TrainerTemplate
from resources.logger import CSVLogger
from ddqn.DDQNAgent import Agent


class Trainer(TrainerTemplate):
    def __init__(self, agent: Agent, epochs: int | None = 100000, max_steps: int = 100000, 
                 update_every: int = 8, validation_interval: int = 150000, validation_steps: int = 10000, goal_reward: float | None = None, 
                 seed: int = 4, 
                 log_dir: str | None = "./logs/ddqn", storage_dir: str = "./trained_models/ddqn") -> None:
        super().__init__(agent, epochs, max_steps, update_every, validation_interval, validation_steps, goal_reward, seed, log_dir, storage_dir)

        # exploration rate
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995

        # overwrite agent settings regarding updating
        self.agent.update_every = self.update_every

        # logging
        # self.logger = SummaryWriter(self.log_dir)
        self.logger = CSVLogger(self.log_dir)
        self.logging_columns = ["sim_steps", "learning_steps", "reward", "eval_score"]

        # try loading information about past training
        self._prev_train_log_file_names = [f for f in os.listdir(self.log_dir) if str(self.agent.hash) in f]
        # print(self._prev_train_log_file_names)
        if len(self._prev_train_log_file_names) > 0:
            self._load_info()

        pass


    def start(self, verbose:bool=False):
        """Start the training (blocking execution).
        """
        # logging
        self.start_time = dt.datetime.now().strftime("%y-%m-%d_%H.%M.%S")
        file_name = "ddqn_training_" + self.start_time + "_agt_" + str(self.agent.hash)
        self.logger.create_file(file_name, self.logging_columns)
        # meta_info = [
        #         ",".join(["epochs", str(self.epochs)]),
        #         ",".join(["max_steps", str(self.max_steps)]),
        #         ",".join(["update_every", str(self.update_every)]),
        #         ",".join(["validation_interval", str(self.validation_interval)]),
        #         ",".join(["validation_steps", str(self.validation_steps)]),
        #         ",".join(["goal_reward", str(self.goal_reward)])
        #     ]
        meta_info = {
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "update_every": self.update_every,
            "validation_interval": self.validation_interval,
            "validation_steps": self.validation_steps,
            "goal_reward": self.goal_reward,
        }
        self.logger.log_meta_data(meta_info)

        # setup
        if self.train_infinitely:
            self.epochs = 1 # keep self.epochs always one higher than the epoch counter

        total_steps = 0
        epoch = 0
        eval_counter = 0
        eps = self.eps_start

        # run
        while epoch < self.epochs:
            epoch_reward = 0
            eps = max(eps * self.eps_decay, self.eps_end)

            state, info = self.agent.env.reset()
            for step in range(self.max_steps):
                total_steps += 1
                # let agent select action
                action = self.agent.selectAction(state, eps)

                # take step with this action
                next_state, reward, terminated, truncated, info = self.agent.env.step(action)

                # update replay buffer and update agent
                self.agent.step(state, action, reward, next_state, terminated)
                
                # store reward
                epoch_reward += reward

                # proceed to next state
                state = next_state

                # end episode if next statae is terminal or environment is truncated (stopped because of time passed)
                if terminated or truncated:
                    break

            # log current reward/ steps
            #self.logger.add_scalar("Reward/Train", epoch_reward, self.agent.steps)
            self.logger.push([self.agent.steps, self.agent.learnings, epoch_reward, 0])

            # if step % self.validation_interval == 0:
            # evaluate if we have had enough steps in between
            if total_steps - (eval_counter * self.validation_interval) >= self.validation_interval:
                eval_counter += 1
                # evaluate
                eval_score = self.evaluate()

                # log current evaluation score
                # self.logger.add_scalar("Reward/Test", eval_score, self.agent.steps)
                self.logger.push([self.agent.steps, self.agent.learnings, 0, eval_score])

                # store agent
                self.agent.save(self.storage_dir, self.agent.steps, inference=False)

                # check if performance requirements are met and quit if so
                if self.goal_reward is not None:
                    if eval_score >= self.goal_reward:
                        break

            epoch += 1
            if verbose:
                # sys.stdout.write(".")
                if epoch % 100 == 0:
                    sys.stdout.write("\n")
                    sys.stdout.write(f"epoch: {epoch}")
                    # sys.stdout.write("\n")
            if self.train_infinitely:
                self.epochs += 1

                
    def evaluate(self, n:int=1, aggregate:bool=True) -> np.array:
        """Evaluate agent
        """
        eval_rewards = []
        for i in range(n):
            eval_reward = 0
            state, info = self.agent.env.reset()

            for step in range(self.max_steps):
                # let agent select action
                action = self.agent.selectAction(state, eps=0.0)

                # take step with this action
                next_state, reward, terminated, truncated, info = self.agent.env.step(action)

                # store reward
                eval_reward += reward

                # end episode if needed
                if terminated or truncated:
                    break
                    
                # proceed
                state = next_state
            
            eval_rewards.append(eval_reward)

        if aggregate:
            return np.array(eval_rewards).mean()
        else:
            return np.array(eval_rewards)
        
    def _load_meta(self, file:str) -> None:
        "Load meta information about past training from provided file"
        with open(file, "r") as f:
            meta = json.load(f)
        self.epochs = meta["epochs"]
        self.max_steps = meta["max_steps"]
        self.update_every = meta["update_every"]
        self.validation_interval = meta["validation_interval"]
        self.validation_steps = meta["validation_steps"]
        self.goal_reward = meta["goal_reward"]

    def _load_info(self) -> None:
        # load latest training informations
        fns = self._prev_train_log_file_names[-2:]
        fn = fns[0]
        fn_meta = fns[1]

        fn_path = os.path.join(self.log_dir, fn)
        fn_meta_path = os.path.join(self.log_dir, fn_meta)

        self._load_meta(fn_meta_path)

        # load training rewards and scores from file
        self.past_training_rewards = pd.read_csv(fn_path)
        self.past_evaluation_scores = self.past_training_rewards[self.past_training_rewards["reward"] == 0][["sim_steps", "eval_score"]]
        self.past_training_rewards = self.past_training_rewards[self.past_training_rewards["eval_score"] == 0][["sim_steps", "reward"]]

