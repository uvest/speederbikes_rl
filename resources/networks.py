import torch
from torch import nn

def create_mlp(sizes:list, activation, output_activation=nn.Identity, device=None):
    """Create an MLP with layer number and sizes defined by list sizes

    Args:
        sizes (list): list containing the size for each layer. 
            Must at least have one entry for the input and one for the output layer (len(sizes) >= 2)
        activation (_type_): class generating the activation function, e.g. torch.nn.ReLU/ torch.nn.Tanh
        output_activation (_type_, optional): same as activation. Defaults to nn.Identity.

    Returns:
        _type_: nn.Sequential
    """
    layers = []
    for i in range(len(sizes)-1):
        act_fun = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(in_features=sizes[i], out_features=sizes[i+1], device=device), act_fun()]
    return nn.Sequential(*layers)


class QNetwork(torch.nn.Module):
    def __init__(self, n_input:int, n_output:int, n_hidden:int, n_hidden_layers:int=2, activation_function:callable=None) -> None:
        super(QNetwork, self).__init__()
        assert (n_input > 0) and (n_output > 0) and (n_hidden > 0) and (n_hidden_layers > 0)
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers= n_hidden_layers

        # could think of introducing bottleneck

        self.activation_function = activation_function
        if self.activation_function is None:
            self.activation_function = torch.nn.ReLU() # torch.relu
        
        # ======
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_layer = torch.nn.Linear(self.n_input, self.n_hidden, bias=True, device=device)
        self.hidden_layers = []
        for _ in range(self.n_hidden_layers - 1):
            self.hidden_layers.append(
                torch.nn.Linear(self.n_hidden, self.n_hidden, device=device)
            )
        self.output_layer = torch.nn.Linear(n_hidden, n_output, device=device)

    def forward(self, x):
        x = self.activation_function(self.input_layer(x))

        for hl in self.hidden_layers:
            x = self.activation_function(hl(x))
        x = self.output_layer(x)

        return x
    

class QNetwork_Alternative(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_sizes:list, activation) -> None:
        super().__init__()
        assert (input_dim > 0) and (output_dim > 0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sizes = [input_dim] + hidden_sizes + [output_dim]
        self.q_net = create_mlp(sizes, activation, device=device)

    def forward(self, obs:torch.tensor):
        return self.q_net(obs)

