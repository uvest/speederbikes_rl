from gymnasium import Env

class Agent():
    def __init__(self, env:Env):
        self.env = env
        # self.mode = mode # "train" or "eval"

    def selectAction(self, state) -> int:
        """Select one action index from the environments action_space for the current state. 
        May apply exploration depending on current mode.
        Args:
            state (_type_): state to evaluate
        """
        pass

    def update(self, error) -> None:
        """Upate parameters based on provided error.
        Args:
            error (_type_): error signal
        """
        pass


    # =====
    # Housekeeping functions
    def load(self, path:str) -> None:
        """load parameters from specified file
        Args:
            file (str): path to file containing agent parameters
        """
        pass

    def save(self, path:str) -> None:
        """Write parameters to specified file
        Args:
            file (str): path to file to write parameters to
        """
        pass

