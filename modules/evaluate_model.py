import torch
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
            

def evaluate_model(model, data_loader, data_stats,  device, info=False):
    model = model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_batch = y_batch.reshape(-1, 1)
             
            y_hat_batch = model(x_batch)
            y_true.extend(y_batch.cpu().numpy().flatten())
            y_pred.extend(y_hat_batch.cpu().numpy().flatten())

    y_mean = data_stats[0].cpu().numpy()
    y_std = data_stats[1].cpu().numpy()
    
    # Transform back to original scale
    y_true_original = (np.array(y_true) * y_std) + y_mean
    y_pred_original = (np.array(y_pred) * y_std) + y_mean

    # calculate metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr = pearsonr(y_true, y_pred)[0]
    pearson_pval = pearsonr(y_true, y_pred)[1]
    spearmanr_corr = spearmanr(y_true, y_pred)[0]
    spearmanr_pval = spearmanr(y_true, y_pred)[1]
    # calculate metrics in years
    mse_years = mean_squared_error(y_true_original, y_pred_original)
    rmse_years = np.sqrt(mse_years)
    
    if info: 
        print("R2: ", r2)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("MSR years: ", mse_years)
        print("RMSE years: ", rmse_years)
        print("Pearson correlation and p-vale: ", pearson_corr, pearson_pval)
        print("Spearman correlation and p-value: ", spearmanr_corr, spearmanr_pval)

    return r2, mse, pearson_corr
            
