import torch
import torch.nn as nn

class LinModel(nn.Module):
    def __init__(self, n_inputs, nodes_per_layer, use_dropout=False, dropout_prob=0.1, use_batchnorm=False, output_size=1): 
        """
        Initialize the LinModel with given parameters.
        
        Parameters:
        n_inputs (int): Number of input features 
        nodes_per_layer (list of int): List containing the number of nodes in each hidden layer e.g. [10,10,10]
        use_dropout (bool): Use dropout layers. Default is False.
        dropout_prob (float): Dropout probability. Default is 0.1.
        use_batchnorm (bool): Use batch normalization. Default is False.
        output_size (int): Number of output features. Default is 1.

        To access first layer weight:    first_linear_weights = model.linear_relu_stack[0].weight
        """
        super(LinModel, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.output_size = output_size
        
        layers = []
        num_layers = len(nodes_per_layer)

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(n_inputs, nodes_per_layer[i]))
            else:
                layers.append(nn.Linear(nodes_per_layer[i-1], nodes_per_layer[i]))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(nodes_per_layer[i]))
            
            layers.append(nn.Tanh())


            if use_dropout and i != num_layers-2:
                layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(nodes_per_layer[-1], output_size))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1)
        x = self.linear_relu_stack(x)
        return x          
