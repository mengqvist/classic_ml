import pytest
import pandas as pd
import os
from pathlib import Path
from data_handling import prepare_romero_data, sample_library

# Fixture to create a temporary CSV file for testing
@pytest.fixture
def temp_tsv(tmp_path):
    # Create a mock project structure
    mock_project_root = tmp_path / "classic_ml"
    mock_data_dir = mock_project_root / "data" / "raw"
    mock_data_dir.mkdir(parents=True)

    df = pd.DataFrame({
        'tm': [50.0, 55.0, 60.0],
        'sequence': ['ACGT', 'TGCA', 'GACT']
    })
    file_path = mock_data_dir / "romero_T50.tsv"
    df.to_csv(file_path, sep='\t', index=False)
    
    return str(file_path)

def test_prepare_romero_data(temp_tsv, monkeypatch):
    # Monkeypatch os.path.exists and os.path.join to use our temp file
    def mock_exists(path):
        return path == temp_tsv
    
    def mock_join(*args):
        return temp_tsv
    
    monkeypatch.setattr(os.path, 'exists', mock_exists)
    monkeypatch.setattr(os.path, 'join', mock_join)
    
    df = prepare_romero_data(temp_tsv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['tm', 'sequence']
    assert len(df) == 3

def test_prepare_romero_data_file_not_found(monkeypatch):
    monkeypatch.setattr(os.path, 'exists', lambda _: False)
    with pytest.raises(FileNotFoundError):
        prepare_romero_data('non_existent_file.tsv')

def test_sample_library():
    df = pd.DataFrame({
        'tm': range(100),
        'sequence': ['A' * i for i in range(100)]
    })
    
    df_train, df_test = sample_library(df, seed=42)
    
    assert len(df_train) == 80
    assert len(df_test) == 20
    assert set(df_train.index).isdisjoint(set(df_test.index))
    assert set(df_train.index) | set(df_test.index) == set(df.index)

def test_sample_library_invalid_frac():
    df = pd.DataFrame({'tm': [1, 2], 'sequence': ['A', 'B']})
    
    with pytest.raises(ValueError, match="frac must be between 0 and 1"):
        sample_library(df, seed=42, frac=1.5)
    
    with pytest.raises(ValueError, match="frac must be between 0 and 1"):
        sample_library(df, seed=42, frac=0)

def test_sample_library_empty_df():
    df = pd.DataFrame({'tm': [], 'sequence': []})
    
    with pytest.raises(ValueError, match="The input dataframe is empty"):
        sample_library(df, seed=42)

def test_sample_library_reproducibility():
    df = pd.DataFrame({
        'tm': range(100),
        'sequence': ['A' * i for i in range(100)]
    })
    
    df_train1, _ = sample_library(df, seed=42)
    df_train2, _ = sample_library(df, seed=42)
    
    assert df_train1.equals(df_train2)

    df_train3, _ = sample_library(df, seed=43)
    assert not df_train1.equals(df_train3)
