import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt 


def prepare_datasets_L(file_path, batch_size=128, train_ratio=0.7, val_ratio=0.15, shuffle=True, info=False):
    """
    Prepare datasets for training, validation, and testing from a NetCDF file.
    Normalise using testing set

    Args:
        file_path (str): Path to the NetCDF file.
        batch_size (int): Batch size for DataLoader.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        shuffle (bool): Whether to shuffle the datasets.

    Returns:
        tuple: DataLoaders and normalization statistics for train, validation, and test sets.
    """
    # Load the netCDF4 dataset
    nc = netCDF4.Dataset(file_path)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
    
    # Drop rows with NaN values in 'tas' column
    ds = ds.dropna(dim='source', subset=['tas'])
    
    # Extract the dimensions and data
    sources = ds['source'].values
    years = ds['year'].values
    lats = ds['lat'].values
    lons = ds['lon'].values
    tas_data = ds['tas'].values  # shape: (source, year, lat, lon)
    
    # Repeat years for each source, stack differnet models together
    repeated_years = np.tile(years, len(sources))
    
    # Create a new dimension for combined source-year
    combined_dim = len(sources) * len(years)
    
    # Reshape tas_data to align with the new combined dimension
    reshaped_tas = tas_data.reshape(combined_dim, len(lats), len(lons))
    
    # Create a new xarray DataArray with the combined dimension
    new_ds = xr.DataArray(
        reshaped_tas,
        dims=['year', 'lat', 'lon'],
        coords={'year': repeated_years, 'lat': lats, 'lon': lons}
    )
    
    # Create a dataset to hold the new DataArray
    new_ds = new_ds.to_dataset(name='tas')
    
    num_years = len(new_ds['year'])
    
    # Determine split indices
    train_end = int(train_ratio * num_years)
    val_end = int((train_ratio + val_ratio) * num_years)
    
    # Split the dataset
    train_ds = new_ds.isel(year=slice(0, train_end))
    val_ds = new_ds.isel(year=slice(train_end, val_end))
    test_ds = new_ds.isel(year=slice(val_end, num_years))
    
    # Shuffle the year dimension for train and val datasets if required
    if shuffle:
        train_ds = train_ds.isel(year=np.random.permutation(train_ds['year'].size))
        val_ds = val_ds.isel(year=np.random.permutation(val_ds['year'].size))


    reshaped_data = train_ds["tas"].transpose('year', 'lat', 'lon').values
    years = train_ds['year'].values
    
    # Convert to PyTorch tensors
    x = torch.tensor(reshaped_data, dtype=torch.float32)
    y = torch.tensor(years, dtype=torch.float32)
    
    # Normalize the data with the training set
    x_mean, x_sd = x.mean(), x.std()
    x = (x - x_mean) / x_sd
    
    y_mean, y_sd = y.mean(), y.std()
    y = (y - y_mean) / y_sd
    
    # Function to create PyTorch datasets
    def create_torch_dataset(xr_dataset):
        data = xr_dataset['tas']
        reshaped_data = data.transpose('year', 'lat', 'lon').values
        years = data['year'].values
        
        # Convert to PyTorch tensors
        x = torch.tensor(reshaped_data, dtype=torch.float32)
        y = torch.tensor(years, dtype=torch.float32)
        
        # Normalize the data
        x = (x - x_mean) / x_sd
        
        y = (y - y_mean) / y_sd
        
        # Create a TensorDataset
        dataset = TensorDataset(x, y)
        
        return dataset, (x_mean, x_sd), (y_mean, y_sd)
    
    # Create datasets
    train_dataset, train_x_stats, train_y_stats = create_torch_dataset(train_ds)
    val_dataset, val_x_stats, val_y_stats = create_torch_dataset(val_ds)
    test_dataset, test_x_stats, test_y_stats = create_torch_dataset(test_ds)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if info:
        # Print information about the dataset and loaders
        print("Original data shape:", tas_data.shape)
        print("Reshaped Data shape:", reshaped_tas.shape)
        print("Repeated years shape:", repeated_years.shape)
        print("Train DataLoader size:", len(train_loader.dataset))
        print("Validation DataLoader size:", len(val_loader.dataset))
        print("Test DataLoader size:", len(test_loader.dataset))
        print("Train x mean, x std:", train_x_stats[0], train_x_stats[1])
        print("Train y mean, y std:", train_y_stats[0], train_y_stats[1])
        print("Validation x mean, x std:", val_x_stats[0], val_x_stats[1])
        print("Validation y mean, y std:", val_y_stats[0], val_y_stats[1])
        print("Test x mean, x std:", test_x_stats[0], test_x_stats[1])
        print("Test y mean, y std:", test_y_stats[0], test_y_stats[1])
        print("Y min, max:", train_ds['year'].min().values, train_ds['year'].max().values)
        for input, targets in train_loader:
            print("Input shape:", input.shape)
            print("Target shape:", targets.reshape(-1, 1).shape)
            break
        
        print(targets[0])
        data_array = xr.DataArray(input[0, :, :])
        data_array.plot()
        plt.show()
        print(targets[0]*train_y_stats[1]+ train_y_stats[0])
        data_array = xr.DataArray(input[0, :, :]*train_x_stats[1]+ train_x_stats[0])
        data_array.plot()
        plt.show()
        
    return train_loader, val_loader, test_loader, train_x_stats, train_y_stats, val_x_stats, val_y_stats, test_x_stats, test_y_stats




# Example usage:
#file_path = 'path_to_your_netCDF_file.nc'
#train_loader, val_loader, test_loader, train_x_stats, train_y_stats, val_x_stats, val_y_stats, test_x_stats, test_y_stats = prepare_datasets(file_path)

