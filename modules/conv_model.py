import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, n_inputs, conv_layers_params, nodes_per_layer, use_dropout=False, dropout_prob=0.1, use_batchnorm=False, output_size=1, pooling_type='max'):
        """
        Initialize the ConvModel with given parameters.
        
        Parameters:
        n_inputs (tuple): Tuple representing the shape of input features (channels, height, width) - is always (1,45,90).
        conv_layers_params (list of tuples): List of tuples, each containing the parameters for a conv layer
                                             in the format (out_channels, kernel_size, stride, padding, pool_kernel).
        nodes_per_layer (list of int): List containing the number of nodes in each hidden layer e.g. [10, 10, 10].
        use_dropout (bool): Use dropout layers. Default is False.
        dropout_prob (float): Dropout probability. Default is 0.1.
        use_batchnorm (bool): Use batch normalization. Default is False.
        output_size (int): Number of output features. Default is 1.
        pooling_type (str): Type of pooling layer to use ('max' or 'avg'). Default is 'max'.

        To access first layer weight:    first_linear_weights = model.conv_layers[0].weight
        """
        super(ConvModel, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        conv_layers = []
        in_channels = n_inputs[0]
        
        for out_channels, kernel_size, stride, padding, pool_kernel in conv_layers_params:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.Tanh())
            if pooling_type == 'max':
                conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernel))
            elif pooling_type == 'avg':
                conv_layers.append(nn.AvgPool2d(kernel_size=pool_kernel))
            if use_dropout:
                conv_layers.append(nn.Dropout2d(dropout_prob))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)

        conv_output_size = self._get_conv_output(n_inputs)

        layers = []
        num_layers = len(nodes_per_layer)

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(conv_output_size, nodes_per_layer[i]))
            else:
                layers.append(nn.Linear(nodes_per_layer[i-1], nodes_per_layer[i]))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(nodes_per_layer[i]))
            
            layers.append(nn.Tanh())

            if use_dropout and i != num_layers - 1:  # Apply dropout to all but the last hidden layer
                layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(nodes_per_layer[-1], output_size))
        self.linear_relu_stack = nn.Sequential(*layers)

    def _get_conv_output(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size()[1:])))  # Exclude batch size

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Output tensor.
        """
        
        x = x.reshape(-1, 1, 45, 90)  # Reshape input to (batch_size, channels, height, width) (1,2,0)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_relu_stack(x)
        
        return x

import torch

# Define input shape: (channels, height, width)
n_inputs = (1, 45, 90)  # Example input with 1 channels, height 45, width 90

# Define convolutional layers parameters: (out_channels, kernel_size, stride, padding, pool_kernel)
conv_layers_params = [
    (16, (4,8), 1, "same", 8),  # 16 output channels, 4,8 kernel, stride 1, padding same, 8x8 pooling
    (32, (4,8), 1, "same", 4)   # 32 output channels, 3x3 kernel, stride 1, padding same, 8x8 pooling
]

# Define the number of nodes in each hidden layer
nodes_per_layer = [64, 32]

# Initialize the model
model = ConvModel(
    n_inputs=n_inputs,
    conv_layers_params=conv_layers_params,
    nodes_per_layer=nodes_per_layer,
    use_dropout=True,
    dropout_prob=0.5,
    use_batchnorm=True,
    output_size=1,  # Example for classification with 10 classes
    pooling_type='max'  # Use max pooling
)

