import os
import pandas as pd

import pickle

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

    # Concatenate all DataFrames into a single DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        print("No valid CSV files found or all files failed to process.")
        return pd.DataFrame(columns=['file_name', 'time', 'atm', 'der_atm'])

# Example usage:
# folder_path = r"C:\path\to\your\folder"
# result_df = combine_csv_files_to_dataframe(folder_path)
# print(result_df)