from gymnasium import Env
from gymnasium import spaces
import torch
import numpy as np
import os

from resources.agent import Agent as AgentTemplate
from resources.networks import QNetwork
from ddqn.replayBuffer import ReplayBuffer

class Agent(AgentTemplate):
    def __init__(self, env:Env, seed:int=99835,
                 learning_rate:float = 0.01, discount_factor:float = 0.99, tau:float=0.001, 
                 update_every:int = 8, batch_size:int=64,
                 buffer_capacity:int=10000, n_hidden:int=64):
        super().__init__(env) # mode is not used

        self.state_size = self.env.observation_space.shape[0] # CHECK

        if type(env.action_space) == spaces.Discrete:
            self.action_size = self.env.action_space.n
        else:
            print("ERROR: inappropriate action space")
            raise TypeError
        
        self.seed = seed
        self.buffer_capacity = buffer_capacity

        # learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        self.n_hidden =n_hidden

        # ========
        self.steps = 0
        self.learnings = 0
        self.qnet_local = QNetwork(n_input=self.state_size, n_output=self.action_size, n_hidden=self.n_hidden) # also called online network
        self.qnet_target = QNetwork(n_input=self.state_size, n_output=self.action_size, n_hidden=self.n_hidden)

        self.optimizer = torch.optim.Adam(params=self.qnet_local.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        # initially set target network parameters to the same as local network parameters
        self.update_target_network()

        # =========
        self.hash = hash(self)

    def step(self, state, action, reward, next_state, terminated) -> None:
        """Save step-tuple to replay buffer
        and learn if sufficint experience was collected
        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            terminated (_type_): _description_
        """
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, terminated)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.update(experiences)

    def selectAction(self, state:np.array, eps=0.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set to evaluation mode
        self.qnet_local.eval()

        # forward pass
        with torch.no_grad():
            action_values = self.qnet_local(state)
        
        # set back to training mode
        self.qnet_local.train()

        # epsilon-greedy action selection (exploration VS exploitation)
        if np.random.random() < eps:
            return np.random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())
        
    def update(self, experiences):
        states, actions, rewards, next_states, terminateds = experiences

        ## Get predicted Q-Values for next state from target network for actions selected based on local network
        #pseudo-code: next_state_action = np.argmax(self.qnet_local(next_states))
        q_next_action_selection = self.qnet_local(next_states) # = q_value_next_state_for_action_selection
        a_next_state = q_next_action_selection.detach().argmax(1)
        # adjust shape to be (batch_size, 1)
        a_next_state = a_next_state.unsqueeze(1)

        #pseudo-code: q_target_next = self.qnet_target(next_states)[next_state_action]
        q_next_value_estimation = self.qnet_target(next_states) # = q_value_next_state_for_value_estimaion
        q_target_next = q_next_value_estimation.gather(1, a_next_state)


        ## Compute Q targets for current states = discout q of next state plus current reward
        rewards = rewards.reshape(rewards.shape[0], 1)
        terminateds = terminateds.reshape(terminateds.shape[0], 1)
        q_target = rewards + self.discount_factor * q_target_next * (1 - terminateds)


        # Get expected Q values from local network
        ## pseudo-code: q_expected = self.qnet_local(states)[actions]
        q_expected = self.qnet_local(states).gather(1, actions.unsqueeze(1)) # actions.view(-1, 1)


        # Compute loss
        loss = torch.nn.functional.mse_loss(q_target, q_expected)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learnings += 1

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        # update the target network using Polyak averaging
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path:str, steps:int, inference:bool=False) -> None:
        """Saves the current agent to the specified directory. 
        Steps, is the current number of steps taken during training, is weaved into the filename.
        Inference determines what is saved and influences the filename, too. Filenameformat: f"ddqn_agent_{str(steps)}.pt" or f"ddqn_agent_inf_{str(steps)}.pt"

        Args:
            path (str): path to directory to save the agent in.
            steps (int): number of simulation steps taken so far during training.
            inference (bool, optional): If True, only the parts needed for inference are saved. Defaults to False.
        """
        if inference:
            # save parts needed for inference
            file = os.path.join(path, f"ddqn_agent_inf_{str(steps)}.pt")
            torch.save({
                "qnet_local_state_dict": self.qnet_local.state_dict(),
                "hash": self.hash
            }, file)
                
        else:
            file = os.path.join(path, f"ddqn_agent_{str(steps)}.pt")
            torch.save({
                'steps': steps,
                'learnings': self.learnings,
                'qnet_local_state_dict': self.qnet_local.state_dict(),
                'qnet_target_state_dict': self.qnet_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'buffer': self.replay_buffer.serialize(),
                "hash": self.hash
            }, file)


    def load(self, path:str, inference:bool=False) -> None:
        """Loads model from provided file or directory. If path is a specific .pt/.pth file, this one is loaded. 
        If it is a path to a directory, the latest model with the most training steps is loaded.
        The latest model is determined based on the name which is expected to be of format f"ddqn_agent_{str(steps)}.pt" or f"ddqn_agent_inf_{str(steps)}.pt"

        Args:
            path (str): path to file or directory
            inference (bool, optional): if True, only the part needed for inference is laoded from the file or directory. Defaults to False.
        """
        if inference:
            # load parts needed for inferene
            if os.path.isdir(path):
                files = os.listdir(path)
                files = [f for f in files if (f.endswith(".pt") or f.endswith(".pth")) and ("_inf_" in f)]
                steps = [int(f.split("_")[-1].split(".")[0]) for f in files] # extract step count from filename as int
                most_steps = np.max(steps)
                path = os.path.join(path, f"ddqn_agent_inf_{str(most_steps)}.pt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # testing
            checkpoint = torch.load(path, map_location=device)
            self.qnet_local.load_state_dict(checkpoint["qnet_local_state_dict"])
            self.hash = checkpoint["hash"]

            self.qnet_local.eval()

        else:
            # load parts needed to continue training
            if os.path.isdir(path):
                files = os.listdir(path)
                files = [f for f in files if (f.endswith(".pt") or f.endswith(".pth")) and (not ("_inf_" in f))]
                steps = [int(f.split("_")[-1].split(".")[0]) for f in files] # extract step count from filename as int
                most_steps = np.max(steps)
                path = os.path.join(path, f"ddqn_agent_{str(most_steps)}.pt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            self.steps = checkpoint["steps"]
            # self.learnings = checkpoint["learnings"]
            self.qnet_local.load_state_dict(checkpoint['qnet_local_state_dict'])
            self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.replay_buffer.deserialize(checkpoint["buffer"])
            self.hash = checkpoint["hash"]

            self.qnet_local.train()
            self.qnet_target.train()
