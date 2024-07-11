import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt            
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))

# Training loop from exercises 
def train_model(model, train_loader, val_loader, device, checkpoint_path, learning_rate=1e-4, patience=75, min_delta=0.01, min_epochs=250, n_epochs=1000, l2_lambda=1e-4, use_l2_reg=False, info=False):
    """
    Train the model with optional L2 regularization and early stopping.
    
    Parameters:
    model (nn.Module): The neural network model to be trained.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    checkpoint_path (str): Path to save the best model checkpoint.
    learning_rate (float): Learning rate for the optimizer. Default is 1e-4.
    patience (int): Number of epochs with no improvement after which training will be stopped. Default is 75.
    min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.01.
    min_epochs (int): Minimum number of epochs to train before considering early stopping. Default is 250.
    n_epochs (int): Total number of epochs to train the model. Default is 1000.
    l2_lambda (float): Regularization strength for L2 regularization. Default is 1e-4.
    use_l2_reg (bool): Flag to use L2 regularization. Default is False.
    info (bool): Flag to plot training and validation loss after training. Default is False.
    
    Returns:
    model (nn.Module): The trained model.
    dict: Dictionary containing loss and validation loss values.
    """
    model = model.to(device)
    
    # Define the loss function and optimizer
    loss_func = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    loss_vals = []
    val_loss_vals = []
    
    training_step = 0
    
    # Training loop with early stopping
    for epoch in tqdm(range(n_epochs)):
        # Training loop
        model.train()
        
        for x_batch, y_batch in train_loader:
            training_step += 1

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # reshape y_batch to match target
            y_batch = y_batch.reshape(-1, 1)

            optimizer.zero_grad()
            y_hat_batch = model(x_batch)
            
            loss = loss_func(y_batch, y_hat_batch)
            
            if use_l2_reg:
                l2_reg = l2_lambda * torch.norm(model.linear_relu_stack[0].weight, 2)
                total_loss = loss + l2_reg
            else:
                total_loss = loss
            
            total_loss.backward()
            loss_vals.append((training_step, total_loss.item()))
            optimizer.step()
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss, total_val_samples = 0.0, 0
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                # reshape y_val_batch to match target
                y_val_batch = y_val_batch.reshape(-1, 1)
                
                y_hat_val_batch = model(x_val_batch)

                total_val_samples += x_val_batch.shape[0]

                val_loss_batch = loss_func(y_val_batch, y_hat_val_batch)

                total_val_loss += x_val_batch.shape[0] * val_loss_batch.item() # add sum over batch to running total
            val_loss_vals.append((training_step, total_val_loss / total_val_samples))
        
        # Early stopping and best model saving
        avg_val_loss = total_val_loss / total_val_samples
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch >= min_epochs:
            print(f'Early stopping at epoch {epoch}')
            break

    # Save the best model state after training
    torch.save({
        'vis_model_state_dict': best_model_state,
        'training_loss': loss_vals,
        'validation_loss': val_loss_vals
    }, checkpoint_path)
    print('Best model saved to', checkpoint_path)
        
    if info:
        plt.plot(*zip(*loss_vals), label='train')
        plt.plot(*zip(*val_loss_vals), label='val')
        plt.legend()
        plt.show()

    model.load_state_dict(best_model_state)

    return model, loss_vals, val_loss_vals

# Example usage:
# model = SimpleModel()
# train_model(model, train_loader, val_loader, torch.device('cuda'), 'model_checkpoint.pth')


def plot_predictions(model, device, loader, data_stats, title):
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            yhat_batch = model(x_batch)

            # Unnormalize predictions and targets
            y_mean = data_stats[0]
            y_std = data_stats[1]

            yhat_batch = (yhat_batch * y_std) + y_mean
            y_batch = (y_batch * y_std) + y_mean

            # Copy tensors to host memory
            yhat_batch = yhat_batch.cpu()
            y_batch = y_batch.cpu()

            all_y_true.extend(y_batch.numpy().flatten())
            all_y_pred.extend(yhat_batch.numpy().flatten())

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.scatter(all_y_true, all_y_pred, alpha=0.8, color='black', label='Predictions')
    plt.plot([1920, 2100], [1920, 2100], color='red', linestyle='--')  # Add diagonal line
    plt.ylabel("Predicted Years")
    plt.xlabel("Actual Years")
    plt.title(title)
    plt.xlim(1900, 2120)
    plt.ylim(1900, 2120)
    plt.grid(True)
    plt.legend()

    plt.show()
            
# Example usage:
# plot_predictions(model, device, train_loader, train_y_stats, "Predicted vs Actual Values (Training Set)")
# plot_predictions(model, device, val_loader, val_y_stats, "Predicted vs Actual Values (Validation Set)")
# plot_predictions(model, device, test_loader, test_y_stats, "Predicted vs Actual Values (Testing Set)")