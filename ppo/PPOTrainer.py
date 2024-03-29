import torch
import numpy as np

import time

from ppo.PPOAgent import PPOAgent
from resources.trainer import Trainer as TrainerTemplate
from ppo.PPOBuffer import PPOBuffer

class PPOTrainer(TrainerTemplate):
    def __init__(self, agent:PPOAgent,
                 epochs:int|None=10000, steps_per_epoch:int=4000, 
                 max_traj_len:int=1000,
                #  gamma:float=0.99, clip_ratio:float=0.2, lam:float=0.97,
                #  pi_lr:float=3e-4, v_lr:float=1e-3,
                 pi_train_max_iters:int=80, v_train_iters:int=80, target_kl:float=0.01, 
                 validation_interval:int=100, validation_steps:int=10000, goal_reward:float|None=None, 
                 seed:int=4,
                 log_dir:str|None="./logs/ppo", storage_dir:str="./trained_models/ppo") -> None:
        """_summary_

        Args:
            agent (Agent): PPOAgent carrying the environment
            epochs (int | None, optional): How many epochs to train for (combined with goal_reward defines abort criterion). Defaults to 10000.
            steps_per_epoch (int, optional): How many steps an epoch has. Can be comprised of multiple episodes/ trajectories. Defaults to 4000.
                gamma (float, optional): discounting factor. Defaults to 0.99.
                clip_ratio (float, optional): clip ratio for clipped PPO. Defaults to 0.2.
                lam (float, optional): lambda for the GAE Q-Value/ Advantage estimate. Defaults to 0.97.
                pi_lr (float, optional): learning rate for the policy network (actor). Defaults to 3e-4.
                v_lr (float, optional): learning rate for the value function (critic). Defaults to 1e-3.
            pi_train_max_iters (int, optional): max number of update steps to take on the policy network each epoch. Defaults to 80.
            v_train_inters (int, optional): number of update steps to take on the v network each epoch. Defaults to 80.
            max_traj_len (int, optional): maximum length of one trajectory/ episode. Defaults to 1000.
            target_kl (float, optional): target KL-Divergence to check. Used for early stopping of pi update. May limit pi update steps. Defaults to 0.01.
            validation_interval (int, optional): validation interval in epochs. Defaults to 100.
            validation_steps (int, optional): maximum number of steps to take in the evaluation trajectory. Defaults to 10000.
            goal_reward (float | None, optional): targeted reward in the evaluation. May stop training earlier than defined by epochs. Defaults to None.
            seed (int, optional): seed for random numbers. Defaults to 4.
            log_dir (str | None, optional): path to the log directory. Defaults to "./logs".
            storage_dir (str, optional): path to store the trained models in. Defaults to "./trained_models".
        """
        # based on https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
        super().__init__(agent, epochs, None, None, validation_interval, validation_steps, goal_reward, seed, log_dir, storage_dir)
        self.agent = agent
        self.steps_per_epoch = steps_per_epoch

        # self.gamma = gamma
        # self.clip_ratio = clip_ratio
        # self.lam = lam
        # self.pi_lr = pi_lr
        # self.v_lr = v_lr
        self.pi_train_max_iters = pi_train_max_iters
        self.v_train_iters = v_train_iters
        self.max_traj_len = max_traj_len
        self.target_kl = target_kl
        
        # ========================
        # create buffer and give it to agent
        # obs_shape = self.agent.env.observation_space.shape[0]
        # act_shape = self.agent.env.action_space.shape[0]
        # buffer = PPOBuffer(obs_dim=obs_shape, act_dim=act_shape, size=steps_per_epoch, gamma=self.gamma, lam=self.lam)
        # self.agent.set_buffer(buffer)
        self.agent.buffer.set_size(steps_per_epoch)


    # def compute_loss_pi(self, experience:dict) -> tuple:
    #     """Implementing the loss function L using importance weighting and clipping for the policy network.
    #     Args:
    #         experience (dict): experience of full trajectories from buffer. Should contain "obs", "act", "adv", "log_prob"
    #     Returns:
    #         tuple: loss, info
    #     """
    #     obs = experience["obs"]
    #     act = experience["act"]
    #     adv = experience["adv"]
    #     log_prob_old = experience["log_prob"]

    #     # policy loss
    #     # 1. get logp of act from current pi
    #     pi, log_prob = self.agent.pi(obs, act)

    #     # 2. calculate importance ratio as prob_new / prob_old
    #     # need to use exponential here, because we were working with log probabilities so far
    #     i_ratio = torch.exp(log_prob - log_prob_old)

    #     # 3. calculate clipped advantage
    #     adv_clipped = torch.clip(i_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

    #     # 4. calculate Advantage_to_be_maximized as average over the minimum of weighted advantage and clipped weighted advantage
    #     L = (torch.min(i_ratio * adv, adv_clipped)).mean()

    #     # 5. because so far this is the Advantage - which we want to maximize - to get the loss, we need to invert it
    #     loss = -L

    #     # more information
    #     approx_kl = (log_prob_old - log_prob).mean().item()
    #     entropy = pi.entropy().mean().item()
    #     clipped = i_ratio.gt(1 + self.clip_ratio) | i_ratio.lt(1 - self.clip_ratio)
    #     cliped_frac = torch.as_tensor(clipped).mean().item() # how many of the transitions were clipped
    #     pi_info = dict(kl=approx_kl, ent=entropy, cf=cliped_frac)

    #     return loss, pi_info
    
    # def compute_loss_v(self, experience:dict) -> torch.tensor:
    #     """Compute MSE loss between current V estimate of the observation and the return to come from the trajectory (noisy estimate).
    #     Args:
    #         experience (dict): experience taken from PPOBuffer. Must contain "obs" and "ret"
    #     Returns:
    #         torch.tensor: MSE loss for updating v function
    #     """
    #     obs = experience["obs"]
    #     ret = experience["ret"]

    #     estimate = self.agent.v(obs)
    #     return ((estimate - ret)**2).mean()
    
    # def update_agent(self):
    #     """Update the policy (actor) and value (critic) function of the agent.
    #     Expects the buffer to be filled.
    #     """
    #     # get data/ experience from full buffer
    #     # experience = self.buffer.get()
    #     experience = self.agent.buffer.get()

    #     # store previour loss for logging reasons
    #     pi_loss_old, pi_info_old = self.compute_loss_pi(experience)
    #     pi_loss_old = pi_loss_old.item()
    #     v_loss_old = self.compute_loss_v(experience).item()

    #     # update policy until it dierges too far to use the current collected data/experience
    #     for i in range(self.pi_train_max_iters):
    #         self.agent.pi_optimizer.zero_grad()
    #         pi_loss, pi_info = self.compute_loss_pi(experience)
    #         kl = pi_info["kl"].mean()
    #         if kl > 1.5 * self.target_kl:
    #             # log that we did early stopping because of too bit KL divergence
    #             break

    #         pi_loss.backward()
    #         self.agent.pi_optimizer.step()

    #     # update value function
    #         for i in range(self.v_train_iters):
    #             self.agent.v_optimizer.zero_grad()
    #             v_loss = self.compute_loss_v(experience)
    #             v_loss.backward()
    #             self.agent.v_optimizer.step()

    #     # log infos about changes during training
    #     kl = pi_info["kl"]
    #     ent = pi_info_old["ent"] # why old? is it the same as the new one?
    #     clipped_frac = pi_info["cf"]
    #     # TODO: log


    def start(self):
        start_time = time.time()

        obs, _info = self.agent.env.reset()
        episode_return = 0
        trajectory_len = 0

        for epoch in range(self.epochs):
            # fill experience buffer until full (=make agent run until buffer full)
            for t in range(self.steps_per_epoch):
                # get action and infos from agent
                action, val, log_prob = self.agent.step(torch.as_tensor(obs, dtype=torch.float32))
                # a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

                # step through environment
                next_obs, reward, terminated, _truncated, _info = self.agent.env.step(action)
                episode_return += reward
                trajectory_len += 1

                # save to buffer
                self.agent.buffer.store(obs, action, reward, val, log_prob)
                # logger.store(VVals=v)
                
                # update observation
                obs = next_obs

                timeout = (trajectory_len == self.max_traj_len) # max_ep_len
                terminal = (terminated or timeout)

                buffer_full = (t==self.steps_per_epoch-1)

                if terminal or buffer_full:
                    if buffer_full and not(terminal):
                        print(f"WARNING: Trajectory cut off by epoch end at {trajectory_len} steps.")

                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or buffer_full:
                        _, val, _ = self.agent.step(torch.as_tensor(obs, dtype=torch.float32))
                    else:
                        val = 0
                    # calculate GAE estimate and discount returns
                    self.agent.buffer.finish_path_estimates(val)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        # logger.store(EpRet=ep_ret, EpLen=ep_len)
                        pass
                    obs, _info = self.agent.env.reset()
                    episode_return = 0
                    trajectory_len = 0

                    # o, ep_ret, ep_len = env.reset(), 0, 0


            # update agent as far as possible with current experience
            print(f"epoch {epoch}: Updating agent...")
            self.agent.update(self.pi_train_max_iters, self.v_train_iters, self.target_kl)
            # log
            ...

            # validate and save
            if epoch % self.validation_interval == 0:
                eval_score = self.evaluate()
                print(f">> Validation score: {eval_score}")
                # log score
                ...
                # save agent
                self.agent.save(self.storage_dir, self.agent.epochs + epoch)
                # check if goal is reached and break
                if self.goal_reward is not None:
                    if eval_score >= self.goal_reward:
                        # log
                        ...
                        print("INFO: Reached target performance. Quitting.")
                        break

        # end for

    
    def evaluate(self) -> float:
        obs, info = self.agent.env.reset()
        self.agent.eval()

        eval_reward = 0

        for t in range(self.validation_steps):
            action = self.agent.selectAction(obs)
            
            # make step
            new_obs, rew, term, trunc, info = self.agent.env.step(action)

            # store reward
            eval_reward += rew

            # update observation
            obs = new_obs

            # end if terminated
            if term or trunc:
                break

        # reset to training mode
        self.agent.train()

        return eval_reward
