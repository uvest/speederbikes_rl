import numpy as np

import gymnasium as gym
from gymnasium import Env
from resources.agent import Agent as AgentTemplate

class Agent(AgentTemplate):
    def __init__(self, env: Env):
        super().__init__(env)

        
    def selectAction(self, state) -> int:
        """Select one action index from the environments action_space for the current state. 
        May apply exploration depending on current mode.
        Args:
            state (_type_): state to evaluate
        """
        return 2