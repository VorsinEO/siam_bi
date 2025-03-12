import os
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

def save_scaler(scaler, file_path):
    """
    Save an sklearn scaler to a pickle file.

    Args:
        scaler: The scaler object (e.g., StandardScaler, MinMaxScaler).
        file_path (str): Path to save the scaler.

    Returns:
        None
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(scaler, file)
        print(f"Scaler saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving scaler: {e}")
        
def load_scaler(file_path):
    """
    Load an sklearn scaler from a pickle file.

    Args:
        file_path (str): Path to the scaler file.

    Returns:
        scaler: The loaded scaler object.
    """
    try:
        with open(file_path, 'rb') as file:
            scaler = pickle.load(file)
        print(f"Scaler loaded successfully from {file_path}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def combine_csv_files_to_dataframe(folder_path):
    """
    Reads all CSV files from the specified folder, combines them into a single DataFrame,
    and adds a 'file_name' column indicating the source file for each row.

    Args:
        folder_path (str): Path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: A DataFrame with columns ['file_name', 'time', 'atm', 'der_atm'].
    """
    # Initialize an empty list to store DataFrames
    dataframes = []
    error_files = []
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the current item is a file and ends with .csv
        if os.path.isfile(file_path):
            try:
                # Read the CSV file without a header and with tab as the separator
                df = pd.read_csv(file_path, sep='\t', header=None)

                # Ensure the file has exactly 3 columns (time, atm, der_atm)
                if df.shape[1] != 3:
                    print(f"Skipping file {file_name}: Expected 3 columns but found {df.shape[1]}.")
                    continue

                # Assign column names to the DataFrame
                df.columns = ['time', 'atm', 'der_atm']

                # Add a new column for the file name
                df['file_name'] = file_name

                # Reorder columns to match the desired output format
                df = df[['file_name', 'time', 'atm', 'der_atm']]

                # Append the processed DataFrame to the list
                dataframes.append(df)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                error_files.append(file_name)

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        for file in error_files:
            combined_df = pd.concat([combined_df,pd.DataFrame({'file_name': {0: file},
                'time': {0: 96.9374591655535},
                'atm': {0: 35.5767},
                'der_atm': {0: 6.169643}})], ignore_index=True)
        return combined_df
    else:
        print("No valid CSV files found or all files failed to process.")
        return pd.DataFrame(columns=['file_name', 'time', 'atm', 'der_atm'])

# Example usage:
# folder_path = r"C:\path\to\your\folder"
# result_df = combine_csv_files_to_dataframe(folder_path)
# print(result_df)


def create_directory(base_path: str, dir_name: str) -> str:
    """
    Create a directory inside the given base path.

    :param base_path: The base directory where the new folder should be created.
    :param dir_name: The name of the new directory.
    :return: The absolute path of the created directory.
    """
    new_dir_path = Path(base_path) / dir_name
    new_dir_path.mkdir(parents=True, exist_ok=True)
    #return str(new_dir_path)



def smooth_and_plot(df, save_path, pressure_type='ma', derivative_type='fd', save_plot=True):
    """
    Function to smooth the pressure and derivative pressure time-series data,
    create log-log plots of both pressure and derivative, and optionally save the plot as an image.
    
    Parameters:
        time (pd.Series): The time series data.
        pressure (pd.Series): The pressure time series data.
        derivative_pressure (pd.Series): The derivative of pressure.
        save_path (str): Path where the plot image will be saved (if save_plot=True).
        pressure_type (str): Type of smoothing for pressure ('ma', 'gaussian', 'sg', 'lowess').
        derivative_type (str): Type of derivative computation ('fd', 'sd', 'cd_log').
        save_plot (bool): Whether to save the plot as an image (default=True). If False, it will just display the plot.
    """
    time = df["time"]
    pressure = df["atm"]
    derivative_pressure = df["der_atm"]
    file_name = df.file_name.head(1).item()
    
    # Smoothing Techniques for Pressure
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def gaussian_smoothing(data, sigma=1):
        return gaussian_filter1d(data, sigma=sigma)

    def savitzky_golay_smoothing(data, window_length=51, polyorder=3):
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    def lowess_smoothing(time, data, frac=0.1):
        smoothed = lowess(data, time, frac=frac)
        return smoothed[:, 1]

    # Apply smoothing to pressure
    if pressure_type == 'ma':
        smoothed_pressure = moving_average(pressure, window_size=5)
    elif pressure_type == 'gaussian':
        smoothed_pressure = gaussian_smoothing(pressure, sigma=2)
    elif pressure_type == 'sg':
        smoothed_pressure = savitzky_golay_smoothing(pressure)
    elif pressure_type == 'lowess':
        smoothed_pressure = lowess_smoothing(time, pressure)
    elif pressure_type == 'origin':
        smoothed_pressure = pressure 
    else:
        raise ValueError("Invalid pressure smoothing type. Choose from 'ma', 'gaussian', 'sg', 'lowess'.")

    # Derivative Calculation Methods
    def finite_difference_derivative(time, pressure):
        return np.gradient(pressure, time)

    def smoothed_derivative(time, pressure, window_length=51, polyorder=3):
        smoothed_pressure = savitzky_golay_smoothing(pressure, window_length, polyorder)
        return np.gradient(smoothed_pressure, time)

    def central_difference_logscale_derivative(time, pressure):
        dt = np.gradient(np.log(time))
        dp = np.gradient(pressure)
        return dp * time / dt

    # Apply chosen derivative method
    if derivative_type == 'fd':
        derivative = finite_difference_derivative(time, pressure)
    elif derivative_type == 'sd':
        derivative = smoothed_derivative(time, pressure)
    elif derivative_type == 'cd_log':
        derivative = central_difference_logscale_derivative(time, pressure)
    elif derivative_type == 'origin':
        derivative = derivative_pressure 
    else:
        raise ValueError("Invalid derivative type. Choose from 'fd', 'sd', 'cd_log'.")

    # Plotting both pressure and its derivative on the same plot (log-log scale)
    plt.figure(figsize=(10, 6))

    # Plot original and smoothed pressure
    #plt.loglog(time, pressure, label='Original Pressure', color='blue', alpha=0.5)
    plt.loglog(time[:len(smoothed_pressure)], smoothed_pressure, label=f'{pressure_type.capitalize()} Smoothing', color='green')

    # Plot original and computed derivative
    #plt.loglog(time, derivative_pressure, label='Original Derivative of Pressure', color='orange', alpha=0.5)
    plt.loglog(time, derivative, label=f'{derivative_type.capitalize()} Derivative', color='red')
    
    plt.title("Smoothed Pressure and Derivative of Pressure (Log-Log Scale)")
    plt.xlabel("Time")
    plt.ylabel("Pressure / Derivative of Pressure")
    plt.legend()
    plt.grid(True)

    # If save_plot is True, save the plot as an image
    if save_plot:
        plt.tight_layout()
        plt.savefig(f'{save_path}/{file_name}.png')
        #print(f"Plot saved to {save_path}")
    else:
        # Just display the plot if save_plot is False
        plt.show()

    # Close the plot after saving or showing
    plt.close()

# Example usage:
# Assuming you have 'time', 'pressure', 'derivative_pressure' data ready and a save path
# smooth_and_plot(time, pressure, derivative_pressure, "/path/to/save/directory", pressure_type='sg', derivative_type='sd', save_plot=True)

# Example to just display the plot without saving
# smooth_and_plot(time, pressure, derivative_pressure, "/path/to/save/directory", pressure_type='ma', derivative_type='fd', save_plot=False)

def smooth_and_compute_derivative(df,window_length =51, polyorder =3):
    """
    Function to smooth the pressure and derivative pressure time-series data using all smoothing and derivative methods, 
    and add new columns to the DataFrame with the smoothed data and derivatives.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing 'time', 'pressure', and 'derivative_pressure' columns.
    
    Returns:
        pd.DataFrame: The DataFrame with added columns for smoothed pressure and derivatives for all methods.
    """
    
    time = df['time']
    pressure = df['atm']
    derivative_pressure = df['der_atm']
    if len(df)<51:
        window_length = len(df)
    if polyorder>=window_length:
        polyorder=window_length-1
        
    
    # Smoothing Techniques for Pressure
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def gaussian_smoothing(data, sigma=1):
        return gaussian_filter1d(data, sigma=sigma)

    def savitzky_golay_smoothing(data, window_length=window_length, polyorder=polyorder):
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    def lowess_smoothing(time, data, frac=0.1):
        smoothed = lowess(data, time, frac=frac)
        return smoothed[:, 1]

    # Derivative Calculation Methods
    def finite_difference_derivative(time, pressure):
        return np.gradient(pressure, time)

    def smoothed_derivative(time, pressure, window_length=window_length, polyorder=polyorder):
        smoothed_pressure = savitzky_golay_smoothing(pressure, window_length, polyorder)
        return np.gradient(smoothed_pressure, time)

    def central_difference_logscale_derivative(time, pressure):
        dt = np.gradient(np.log(time))
        dp = np.gradient(pressure)
        return dp * time / dt
    if len(df)>1:
        # Apply smoothing to pressure (all types)
        smoothed_pressure_ma = moving_average(pressure, window_size=5)
        smoothed_pressure_gaussian = gaussian_smoothing(pressure, sigma=2)
        #print(window_length,polyorder)
        smoothed_pressure_sg = savitzky_golay_smoothing(pressure, window_length=window_length, polyorder=polyorder)
        smoothed_pressure_lowess = lowess_smoothing(time, pressure)
    
        # Apply derivative methods (all types)
        derivative_fd = finite_difference_derivative(time, pressure)
        derivative_sd = smoothed_derivative(time, pressure, window_length=window_length, polyorder=polyorder)
        derivative_cd_log = central_difference_logscale_derivative(time, pressure)
        

        # Add new columns to the DataFrame for all smoothed pressure and derivatives
        df['smoothed_pressure_ma'] = np.pad(smoothed_pressure_ma, (0, len(df) - len(smoothed_pressure_ma)), mode='edge')
        df['smoothed_pressure_gaussian'] = np.pad(smoothed_pressure_gaussian, (0, len(df) - len(smoothed_pressure_gaussian)), mode='edge')
        df['smoothed_pressure_sg'] = np.pad(smoothed_pressure_sg, (0, len(df) - len(smoothed_pressure_sg)), mode='edge')
        df['smoothed_pressure_lowess'] = np.pad(smoothed_pressure_lowess, (0, len(df) - len(smoothed_pressure_lowess)), mode='edge')
    
        df['derivative_fd'] = np.pad(derivative_fd, (0, len(df) - len(derivative_fd)), mode='edge')
        df['derivative_sd'] = np.pad(derivative_sd, (0, len(df) - len(derivative_sd)), mode='edge')
        df['derivative_cd_log'] = np.pad(derivative_cd_log, (0, len(df) - len(derivative_cd_log)), mode='edge')
    else:
        df['smoothed_pressure_ma']=0
        df['smoothed_pressure_gaussian']=0
        df['smoothed_pressure_sg']=0
        df['smoothed_pressure_lowess']=0
        df['derivative_fd']=0
        df['derivative_sd']=0
        df['derivative_cd_log']=0
    return df