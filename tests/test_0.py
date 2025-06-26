import pytest
from definition_43cc3edb5a24492498957b2b88daa60f import load_dataset
import pandas as pd
import os

@pytest.fixture
def sample_file(tmp_path):
    # Create a sample CSV file for testing
    file_path = tmp_path / "dataset.csv"
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.0, 5.5, 6.1],
        'C': ['x', 'y', 'z']
    })
    data.to_csv(file_path, index=False)
    return str(file_path)

def test_load_dataset_valid_file(sample_file):
    df = load_dataset(sample_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['A', 'B', 'C']

def test_load_dataset_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent_file.csv")

def test_load_dataset_invalid_format(tmp_path):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("Just some text, not a CSV")
    with pytest.raises(pd.errors.EmptyDataError):
        load_dataset(str(invalid_file))

def test_load_dataset_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_dataset(str(empty_file))

def test_load_dataset_with_no_extension(tmp_path):
    no_ext_file = tmp_path / "no_extension"
    no_ext_file.write_text("A,B,C\n1,2,3")
    df = load_dataset(str(no_ext_file))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['A', 'B', 'C']

def test_load_dataset_with_extra_whitespace(tmp_path):
    file_path = tmp_path / "whitespace.csv"
    content = " A , B , C \n 1 , 2 , 3 \n 4 , 5 , 6 "
    file_path.write_text(content)
    df = load_dataset(str(file_path))
    assert 'A' in df.columns
    assert df.iloc[0]['A'] == 1

def test_load_dataset_with_only_headers(tmp_path):
    header_file = tmp_path / "headers.csv"
    header_file.write_text("A,B,C\n")
    df = load_dataset(str(header_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 0

def test_load_dataset_with_malformed_csv(tmp_path):
    malformed_file = tmp_path / "malformed.csv"
    malformed_file.write_text("A,B,C\n1,2\n3,4,5,6")
    # Should load without error, but with NaNs
    df = load_dataset(str(malformed_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 0
