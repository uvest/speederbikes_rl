import torch
from torch import nn

import gymnasium as gym
from gymnasium import Env

import numpy as np
from typing import Tuple
import os

from resources.agent import Agent as AgentTemplate
from resources.networks import create_mlp
from ppo.PPOBuffer import PPOBuffer


class MLPGaussianPolicy(torch.nn.Module): # "Actor"
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        raise NotImplementedError
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        raise NotImplementedError
        # mu = self.mu_net(obs)
        # std = torch.exp(self.log_std)
        # return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
        # return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def forward(self, obs):
        raise NotImplementedError


class MLPCategoricalPolicy(torch.nn.Module): # "Actor"
    def __init__(self, input_dim:int, hidden_sizes:list, output_dim:int, activation) -> None:
        super().__init__()
        sizes = [input_dim] + hidden_sizes + [output_dim]
        self.logits_net = create_mlp(sizes, activation) # ??? Why logit here? Why couldn't the output be a probability directly, so using the logistic here?

    def distribution(self, obs:torch.Tensor|np.ndarray) -> torch.distributions.Distribution:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)
    
    def log_prob(self, pi:torch.distributions.Distribution, act) -> torch.Tensor:
        return pi.log_prob(act)
    
    def forward(self, obs:torch.Tensor|np.ndarray, act:torch.Tensor=None) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        pi = self.distribution(obs)
        log_prob_a = None
        if act is not None:
            log_prob_a = self.log_prob(pi, act)
        return pi, log_prob_a
        

class VNetwork(torch.nn.Module): # "Critic"
    def __init__(self, input_dim:int, hidden_sizes:list, activation) -> None:
        """Create Critic using an MLP V-Network
        Args:
            input_dim (int): dimension of observation/ state
            hidden_sizes (list): list of sizes of hidden layers. Can be empty
            activation (_type_): activation function to use for the network. E.g. torch.nn.ReLU/ torch.nn.Tanh
        """
        super().__init__()
        sizes = [input_dim] + hidden_sizes + [1]
        self.v_net = create_mlp(sizes, activation)

    def forward(self, obs:torch.Tensor|np.ndarray) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        v_value = self.v_net(obs)
        return torch.squeeze(v_value, -1)




class PPOAgent(AgentTemplate): # ActorCritic Agent
    def __init__(self, env: Env, hidden_sizes:list=[64, 64],
                 gamma:float=0.99, clip_ratio:float=0.2, lam:float=0.97,
                 pi_lr:float=3e-4, v_lr:float=1e-3, 
                 activation=torch.nn.Tanh):
        super().__init__(env)

        obs_shape = self.env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            act_shape = 1
            num_possible_actions = self.env.action_space.n
        else:
            act_shape = self.env.action_space.shape[0]
            num_possible_actions = ...
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.pi = MLPCategoricalPolicy(obs_shape, hidden_sizes, num_possible_actions, activation=activation)
        # elif isinstance(self.env.action_space, gym.spaces.Box):
        #     self.pi = MLPGaussianPolicy(obs_shape, hidden_sizes, num_possible_actions, activation=activation)
            # not implemented
        else:
            print("ERROR: Wrong action space type.")
            raise NotImplementedError
        
        self.v = VNetwork(obs_shape, hidden_sizes, activation=activation)

        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=v_lr)

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lam = lam

        # create buffer
        self.buffer = PPOBuffer(obs_dim=obs_shape, act_dim=act_shape, size=None, gamma=self.gamma, lam=self.lam)

        # ======
        self.epochs = 0 # number of epochs trained so far
        self.hash = hash(self)
        

    def initialize_buffer(self, size:float):
        self.buffer.set_size(size)


    def step(self, obs:torch.Tensor|np.ndarray) -> tuple: # was called "step" in original code
        """Selects an action given the provided observation.
        Also returns the observations value estimate and the action's log prob

        Args:
            obs (torch.Tensor): obsercation as tensor
        Returns:
            tuple: selected action, value function estimate of provided observation, log prob of a
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            # select action
            pi = self.pi.distribution(obs)
            a = pi.sample()
            # calculate log probability of selected action
            log_prob_a = self.pi.log_prob(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), log_prob_a.numpy()
    

    def selectAction(self, obs:torch.Tensor|np.ndarray) -> np.array:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            # select action
            pi = self.pi.distribution(obs)
            a = pi.sample()
        return a.numpy()


    def eval(self) -> None:
        self.pi.eval()
        self.v.eval()

    def train(self) -> None:
        self.pi.train()
        self.v.train()

    # ==== PPO learning behaviour
    def compute_loss_pi(self, experience:dict) -> tuple:
        """Implementing the loss function L using importance weighting and clipping for the policy network.
        Args:
            experience (dict): experience of full trajectories from buffer. Should contain "obs", "act", "adv", "log_prob"
        Returns:
            tuple: loss, info
        """
        obs = experience["obs"]
        act = experience["act"]
        adv = experience["adv"]
        log_prob_old = experience["log_prob"]

        # policy loss
        # 1. get logp of act from current pi
        pi, log_prob = self.pi(obs, act)

        # 2. calculate importance ratio as prob_new / prob_old
        # need to use exponential here, because we were working with log probabilities so far
        i_ratio = torch.exp(log_prob - log_prob_old)

        # 3. calculate clipped advantage
        adv_clipped = torch.clip(i_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

        # 4. calculate Advantage_to_be_maximized as average over the minimum of weighted advantage and clipped weighted advantage
        L = (torch.min(i_ratio * adv, adv_clipped)).mean()

        # 5. because so far this is the Advantage - which we want to maximize - to get the loss, we need to invert it
        loss = -L

        # more information
        approx_kl = (log_prob_old - log_prob).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = i_ratio.gt(1 + self.clip_ratio) | i_ratio.lt(1 - self.clip_ratio)
        cliped_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item() # how many of the transitions were clipped
        pi_info = dict(kl=approx_kl, ent=entropy, cf=cliped_frac)

        return loss, pi_info
    
    def compute_loss_v(self, experience:dict) -> torch.Tensor:
        """Compute MSE loss between current V estimate of the observation and the return to come from the trajectory (noisy estimate).
        Args:
            experience (dict): experience taken from PPOBuffer. Must contain "obs" and "ret"
        Returns:
            torch.Tensor: MSE loss for updating v function
        """
        obs = experience["obs"]
        ret = experience["ret"]

        estimate = self.v(obs)
        return ((estimate - ret)**2).mean()
    
    def update(self, pi_train_max_iters:int, v_train_iters:int, target_kl:float):
        """Update the policy (actor) and value (critic) function of the agent.
        Expects the buffer to be filled.
        """
        # get data/ experience from full buffer
        # experience = self.buffer.get()
        experience = self.buffer.get()

        # store previour loss for logging reasons
        pi_loss_old, pi_info_old = self.compute_loss_pi(experience)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = self.compute_loss_v(experience).item()

        # update policy until it dierges too far to use the current collected data/experience
        for i in range(pi_train_max_iters):
            self.pi_optimizer.zero_grad()
            pi_loss, pi_info = self.compute_loss_pi(experience)
            kl = pi_info["kl"]#.mean()
            if kl > 1.5 * target_kl:
                # log that we did early stopping because of too bit KL divergence
                break

            pi_loss.backward()
            self.pi_optimizer.step()

        # update value function
            for i in range(v_train_iters):
                self.v_optimizer.zero_grad()
                v_loss = self.compute_loss_v(experience)
                v_loss.backward()
                self.v_optimizer.step()

        # log infos about changes during training
        kl = pi_info["kl"]
        ent = pi_info_old["ent"] # why old? is it the same as the new one?
        clipped_frac = pi_info["cf"]
        # TODO: log


    # =====
    # Housekeeping functions
    def save(self, path:str, epochs:int, inference:bool=False) -> None:
        """Saves the current agent to the specified directory. 
        Epochs, the current number of epochs taken during training, is weaved into the filename.
        Inference determines what is saved and influences the filename, too. Filenameformat: f"ppo_agent_{str(epochs)}.pt" or f"ppo_agent_inf_{str(epochs)}.pt"
        Args:
            path (str): path to directory to save the agent in.
            epochs (int): number of epochs taken so far during training.
            inference (bool, optional): If True, only the parts needed for inference are saved. Defaults to False.
        """
        if inference:
            # save parts needed for inference
            file = os.path.join(path, f"ppo_agent_inf_{str(epochs)}.pt")
            torch.save({
                "pi_state_dict": self.pi.state_dict(),
                "hash": self.hash
            }, file)
                
        else:
            file = os.path.join(path, f"ppo_agent_{str(epochs)}.pt")
            torch.save({
                'epochs': epochs,
                'pi_state_dict': self.pi.state_dict(),
                'v_state_dict': self.v.state_dict(),
                'pi_opt_state_dict': self.pi_optimizer.state_dict(),
                'v_opt_state_dict': self.v_optimizer.state_dict(),
                "hash": self.hash
            }, file)

    def load(self, path:str, inference:bool=False) -> None:
        """load parameters from specified file
        Args:
            file (str): path to file containing agent parameters
        """
        if inference:
            # load parts needed for inferene
            if os.path.isdir(path):
                files = os.listdir(path)
                files = [f for f in files if (f.endswith(".pt") or f.endswith(".pth")) and ("_inf_" in f)]
                epochs = [int(f.split("_")[-1].split(".")[0]) for f in files] # extract step count from filename as int
                most_epochs = np.max(epochs)
                path = os.path.join(path, f"ppo_agent_inf_{str(most_epochs)}.pt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # testing
            checkpoint = torch.load(path, map_location=device)
            self.pi.load_state_dict(checkpoint["pi_state_dict"])
            self.hash = checkpoint["hash"]

            self.pi.eval()

        else:
            # load parts needed to continue training
            if os.path.isdir(path):
                files = os.listdir(path)
                files = [f for f in files if (f.endswith(".pt") or f.endswith(".pth")) and (not ("_inf_" in f))]
                epochs = [int(f.split("_")[-1].split(".")[0]) for f in files] # extract step count from filename as int
                most_epochs = np.max(epochs)
                path = os.path.join(path, f"ppo_agent_{str(most_epochs)}.pt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            self.epochs = checkpoint["epochs"]
            # self.learnings = checkpoint["learnings"]
            self.pi.load_state_dict(checkpoint['pi_state_dict'])
            self.v.load_state_dict(checkpoint['v_state_dict'])
            self.pi_optimizer.load_state_dict(checkpoint["pi_opt_state_dict"])
            self.v_optimizer.load_state_dict(checkpoint["v_opt_state_dict"])
            self.hash = checkpoint["hash"]

            self.pi.train()
            self.v.train()

