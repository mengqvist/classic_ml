import numpy as np
import pytest
from typing import List, Callable

# Now import from src.encode
from encode import encode_sequences, one_hot, heil_short, bork, tscale, property_encoding, aa_index, is_valid_sequence

# Test sequences
test_sequences = [
    "ACDEFGHIKLMNPQRSTVWY",  # All 20 standard amino acids
    "ACDEF",                 # Short sequence
    "A" * 100,               # Long sequence with repeats
    "ACDEFGHIKL",           # Sequence with a gap
    "ACDEFXGHIKL",           # Sequence with an unknown amino acid
    "acdefghiklmnpqrstvwy",  # Lowercase sequence
    "AC DE FG HI KL",        # Sequence with spaces (invalid)
    "",                      # Empty sequence
    "BZJOU",                 # Sequence with invalid characters
    "ARNDCQEGHILKMFPSTWYV-X"  # All standard AAs, gap, and unknown
]

# Non-string inputs for testing
non_string_inputs = [
    42,                      # Integer
    3.14,                    # Float
    True,                    # Boolean
    ["a", 2, 'C', 4],            # List of ints and strings
    {'A': 1, 'C': 2},        # Dictionary
    None,                    # None
    b"ACGT",                 # Bytes
    np.array(["ACGT"]),      # Numpy array
    set("ACGT"),             # Set
]

# Test functions
def test_is_valid_sequence():
    """Test is_valid_sequence function"""
    assert is_valid_sequence("ACDEFGHIKLMNPQRSTVWY")  # Valid sequence
    assert is_valid_sequence("ACDEFXGHIKL")  # Sequence with X is valid
    assert not is_valid_sequence("ACDEF-GHIKL")  # Sequence with gap is invalid
    assert not is_valid_sequence("BZJOU")  # Sequence with invalid characters
    assert not is_valid_sequence("")  # Empty sequence is invalid

def test_encode_sequences_valid_modes():
    """Test encode_sequences with all valid modes"""
    modes = ["identity", "heil_short", "bork", "t-scale", "property", "aa_index"]
    for mode in modes:
        result = encode_sequences(test_sequences[:5], mode)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5  # 5 valid sequences
        assert all(len(seq) == len(result[0]) for seq in result)  # All sequences should have the same length

def test_encode_sequences_invalid_mode():
    """Test encode_sequences with an invalid mode"""
    with pytest.raises(ValueError):
        encode_sequences(test_sequences[:5], "invalid_mode")

def test_encode_sequences_invalid_sequence():
    """Test encode_sequences with an invalid sequence"""
    with pytest.raises(ValueError):
        encode_sequences(["ACDEF-GHIKL"], "identity")  # Sequence with a gap

def test_encode_sequences_empty_sequence():
    """Test encode_sequences with an empty sequence"""
    with pytest.raises(ValueError):
        encode_sequences(test_sequences[7:8], "identity")  # Empty sequence

def test_encode_sequences_lowercase():
    """Test encode_sequences with lowercase sequence"""
    result = encode_sequences(test_sequences[5:6], "identity")
    assert result.shape[0] == 1  # Should handle lowercase

def test_encoding_functions_output():
    """Test output shape and type for all encoding functions"""
    encoding_funcs = [one_hot, heil_short, bork, tscale, property_encoding, aa_index]
    for func in encoding_funcs:
        result = func()
        assert isinstance(result, dict)
        assert set(result.keys()) >= set("ACDEFGHIKLMNPQRSTVWY-X")
        assert all(isinstance(v, np.ndarray) for v in result.values())

def test_encoding_functions_consistency():
    """Test consistency of encoding across different calls"""
    encoding_funcs = [one_hot, heil_short, bork, tscale, property_encoding, aa_index]
    for func in encoding_funcs:
        result1 = func()
        result2 = func()
        assert all(np.array_equal(result1[k], result2[k]) for k in result1)

def test_unknown_and_gap_encoding():
    """Test that unknown (X) and gap (-) are encoded consistently"""
    encoding_funcs = [one_hot, heil_short, bork, tscale, property_encoding, aa_index]
    for func in encoding_funcs:
        result = func()
        assert np.array_equal(result['X'], np.zeros_like(result['A']))
        assert np.array_equal(result['-'], np.zeros_like(result['A']))

def test_encoding_dimensions():
    """Test that each encoding method produces the expected number of dimensions"""
    expected_dims = {
        "identity": 20,
        "heil_short": 9,
        "bork": 11,
        "t-scale": 5,
        "property": 12,
        "aa_index": 5
    }
    for mode, dims in expected_dims.items():
        result = encode_sequences(["A"], mode)
        assert result.shape[1] == dims

def test_long_sequence_encoding():
    """Test encoding of a long sequence"""
    long_seq = ["A" * 1000]
    for mode in ["identity", "heil_short", "bork", "t-scale", "property", "aa_index"]:
        result = encode_sequences(long_seq, mode)
        expected_dims = {"identity": 20, "heil_short": 9, "bork": 11, "t-scale": 5, "property": 12, "aa_index": 5}
        assert result.shape == (1, 1000 * expected_dims[mode])

def test_encode_sequences_non_string_input():
    """Test encode_sequences with non-string input in the list"""
    for non_string in non_string_inputs:
        with pytest.raises(TypeError):
            encode_sequences([non_string], "identity")

def test_encode_sequences_mixed_input():
    """Test encode_sequences with a mix of valid strings and non-strings"""
    mixed_input = ["ACDEF", 42, "GHIKL", ['A', 'C'], "MNPQR"]
    with pytest.raises(TypeError):
        encode_sequences(mixed_input, "identity")

def test_is_valid_sequence_non_string():
    """Test is_valid_sequence function with non-string inputs"""
    for non_string in non_string_inputs:
        with pytest.raises(TypeError):
            is_valid_sequence(non_string)

def test_encode_sequences_single_non_string():
    """Test encode_sequences with a single non-string input (not in a list)"""
    for non_string in non_string_inputs:
        with pytest.raises(TypeError):
            encode_sequences(non_string, "identity")

def test_encode_sequences_nested_list():
    """Test encode_sequences with a nested list input"""
    nested_input = [["ACDEF"], ["GHIKL"]]
    with pytest.raises(TypeError):
        encode_sequences(nested_input, "identity")

def test_sequences_with_x():
    """Test that sequences with 'X' are considered valid"""
    valid_sequences = ["AXCEF", "XGHIK", "LMNPX"]
    result = encode_sequences(valid_sequences, "identity")
    assert result.shape[0] == 3  # All three sequences should be valid


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])