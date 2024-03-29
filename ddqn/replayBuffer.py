import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, capacity:int) -> None:
        self.capacity = capacity
        self.buffer = []
        self.index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, terminated) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # add new element to fill if not all indeces are initialized, yet.
        self.buffer[self.index] = (state, action, reward, next_state, terminated)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size:int) -> tuple:
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, terminateds = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, terminated = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)

        # explicit typecasting
        ## action castin needed for gather(...) later
        actions = torch.tensor(np.array(actions), device=self.device).int()
        actions = actions.type(torch.int64)
        
        return (
            torch.tensor(np.array(states), device=self.device).float(),
            actions,
            torch.tensor(np.array(rewards), device=self.device).float(), # might need unsqueeze(1).float
            torch.tensor(np.array(next_states), device=self.device).float(),
            torch.tensor(np.array(terminateds), device=self.device).int(), # might need unsqueeze(1).int
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def serialize(self) -> dict:
        """Serialize current state to a dictionary
        Returns:
            dict: _description_
        """
        return {
            "capacity": self.capacity,
            "index": self.index,
            "buffer": self.buffer
        }
    
    def deserialize(self, dict:dict) -> None:
        """Load state from dictionary
        Args:
            dict (dict): dict created with self.serialize
        """
        self.capacity = dict["capacity"]
        self.index = dict["index"]
        self.buffer = dict["buffer"]
    