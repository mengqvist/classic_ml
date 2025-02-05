import pandas as pd
import os


def prepare_romero_data(file_path):
    """
    Read the Romero TSV file and prepare the dataframe.
    
    Args:
    file_path (str): Path to the TSV file. Defaults to 'data/raw/romero_T50.tsv'.
    
    Returns:
    pd.DataFrame: Prepared dataframe with 'tm' and 'sequence' columns.
    """
    # Check if the file exists    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the file
    df = pd.read_csv(file_path, sep='\t')
    df.columns = ['tm', 'sequence']
    
    return df[['tm', 'sequence']]


def sample_library(df, seed, frac=0.8):
    """
    Sample the library to create train and test sets.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    seed (int): Random seed for reproducibility.
    frac (float): Fraction of data to include in the training set. Defaults to 0.8.
    
    Returns:
    tuple: (df_train, df_test) - Training and test dataframes.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1.")
    
    if df.empty:
        raise ValueError("The input dataframe is empty.")
    
    df_train = df.sample(frac=frac, random_state=seed)
    df_test = df.drop(df_train.index)
    return df_train, df_test
