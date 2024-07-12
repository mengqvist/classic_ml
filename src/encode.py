import numpy as np
import re

from typing import List, Dict, Any

def is_valid_sequence(sequence: Any, allow_gap: bool = False) -> bool:
    """
    Check if the input is a valid amino acid sequence.

    :param sequence: The input to validate
    :param allow_gap: Whether to allow gap character '-' in the sequence (default: True)
    :return: True if the sequence is valid, False otherwise
    """
    if not isinstance(sequence, str):
        raise TypeError(f"Input must be a string, not {type(sequence).__name__}")

    valid_aa = set('ACDEFGHIKLMNPQRSTVWYX')  # Added 'X' as valid
    if allow_gap:
        valid_aa.add('-')
    
    sequence = sequence.upper()
    
    if not sequence:
        return False
    
    return set(sequence).issubset(valid_aa)


def create_encoding(properties: Dict[str, List[Any]], num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create an encoding dictionary based on given properties.

    Args:
        properties (Dict[str, List[Any]]): Dictionary of amino acid properties
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of encoded amino acids
    """
    codes = {}
    for aa, props in properties.items():
        codes[aa] = np.array(props, dtype=num_fmt)
    
    # Add default encodings for gap and unknown
    codes['-'] = np.zeros_like(list(codes.values())[0])
    codes['X'] = np.zeros_like(list(codes.values())[0])
    
    return codes


def one_hot(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a one-hot encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of one-hot encoded amino acids
    """
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    properties = {aa: [1 if i == j else 0 for j in range(len(AA_ORDER))] for i, aa in enumerate(AA_ORDER)}
    return create_encoding(properties, num_fmt)


def heil_short(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a Heil short encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of Heil short encoded amino acids
    """
    properties = {
        'A': [1, 0, 1, 0, 1, 1, 0, 0, 0],
        'C': [0, 0, 0, 1, 0, 1, 0, 0, 0],
        'D': [0, 1, 0, 0, 0, 0, 1, 0, 0],
        'E': [0, 1, 0, 0, 0, 0, 0, 1, 0],
        'F': [0, 0, 1, 0, 0, 0, 0, 0, 1],
        'G': [0, 0, 0, 1, 1, 0, 0, 0, 0],
        'H': [0, 0, 0, 1, 0, 0, 0, 1, 0],
        'I': [0, 0, 1, 0, 0, 0, 1, 0, 0],
        'K': [1, 0, 0, 0, 0, 0, 0, 1, 0],
        'L': [0, 0, 1, 0, 0, 0, 1, 0, 0],
        'M': [0, 0, 1, 0, 0, 0, 0, 1, 0],
        'N': [0, 0, 0, 1, 0, 0, 1, 0, 0],
        'P': [0, 0, 1, 0, 0, 1, 0, 0, 0],
        'Q': [0, 0, 0, 1, 0, 0, 0, 1, 0],
        'R': [1, 0, 0, 0, 0, 0, 0, 0, 1],
        'S': [0, 0, 0, 1, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 1, 0, 1, 0, 0, 0],
        'V': [0, 0, 1, 0, 0, 1, 0, 0, 0],
        'W': [0, 0, 1, 0, 0, 0, 0, 0, 1],
        'Y': [0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    return create_encoding(properties, num_fmt)


def bork(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a Bork encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of Bork encoded amino acids
    """
    properties = {
        'A': [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        'C': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'D': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'E': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'F': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'G': [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        'H': [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        'I': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'K': [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        'L': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'M': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        'Q': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'R': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'T': [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        'V': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        'W': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'Y': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    }
    return create_encoding(properties, num_fmt)


def tscale(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a T-scale encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of T-scale encoded amino acids
    """
    properties = {
        'A': [-9.11, -1.63, 0.63, 1.04, 2.26],
        'R': [0.23, 3.89, -1.16, -0.39, -0.06],
        'N': [-4.62, 0.66, 1.16, -0.22, 0.93],
        'D': [-4.65, 0.75, 1.39, -0.40, 1.05],
        'C': [-7.35, -0.86, -0.33, 0.80, 0.98],
        'Q': [-3.00, 1.72, 0.28, -0.39, 0.33],
        'E': [-3.03, 1.82, 0.51, -0.58, 0.43],
        'G': [-10.61, -1.21, -0.12, 0.75, 3.25],
        'H': [-1.01, -1.31, 0.01, -1.81, -0.21],
        'I': [-4.25, -0.28, -0.15, 1.40, -0.21],
        'L': [-4.38, 0.28, -0.49, 1.45, 0.02],
        'K': [-2.59, 2.34, -1.69, 0.41, -0.21],
        'M': [-4.08, 0.98, -2.34, 1.64, -0.79],
        'F': [0.49, -0.94, -0.63, -1.27, -0.44],
        'P': [-5.11, -3.54, -0.53, -0.36, -0.29],
        'S': [-7.44, -0.65, 0.68, -0.17, 1.58],
        'T': [-5.97, -0.62, 1.11, 0.31, 0.95],
        'W': [5.73, -2.67, -0.07, -1.96, -0.54],
        'Y': [2.08, -0.47, 0.07, -1.67, -0.35],
        'V': [-5.87, -0.94, 0.28, 1.10, 0.48]
    }
    return create_encoding(properties, num_fmt)


def property_encoding(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a property-based encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of property-encoded amino acids
    """
    # Properties for amino acids
    # "Molar mass", "Van der Waals volume", "Vr A3", "Nonpolar", "Polar", "Neutral", "Acidic", "Basic", "Basic (weak)", "Basic (strong)", "Hydropathy index", "Isoelectric point (pI)"
    properties = {
        'A': [89.09, 67, 92, 1, 0, 1, 0, 0, 0, 0, 1.8, 6.01],
        'C': [121.15, 86, 106, 0, 1, 1, 0, 0, 0, 0, 2.5, 5.05],
        'D': [133.1, 91, 125, 0, 1, 0, 1, 0, 0, 0, -3.5, 2.85],
        'E': [147.13, 109, 161, 0, 1, 0, 1, 0, 0, 0, -3.5, 3.15],
        'F': [165.19, 135, 203, 1, 0, 1, 0, 0, 0, 0, 2.8, 5.49],
        'G': [75.06, 48, 66, 1, 0, 1, 0, 0, 0, 0, -0.4, 6.06],
        'H': [155.15, 118, 167, 0, 1, 0, 0, 1, 1, 0, -3.2, 7.6],
        'I': [131.17, 124, 169, 1, 0, 1, 0, 0, 0, 0, 4.5, 6.05],
        'K': [146.18, 135, 171, 0, 1, 0, 0, 1, 0, 1, -3.9, 9.6],
        'L': [131.17, 124, 168, 1, 0, 1, 0, 0, 0, 0, 3.8, 6.01],
        'M': [149.2, 124, 171, 1, 0, 1, 0, 0, 0, 0, 1.9, 5.74],
        'N': [132.11, 96, 135, 0, 1, 1, 0, 0, 0, 0, -3.5, 5.41],
        'P': [115.13, 90, 129, 1, 0, 1, 0, 0, 0, 0, -1.6, 6.3],
        'Q': [146.15, 114, 155, 0, 1, 1, 0, 0, 0, 0, -3.5, 5.65],
        'R': [174.2, 148, 225, 0, 1, 0, 0, 1, 0, 1, -4.5, 10.76],
        'S': [105.09, 73, 99, 0, 1, 1, 0, 0, 0, 0, -0.8, 5.68],
        'T': [119.12, 93, 122, 0, 1, 1, 0, 0, 0, 0, -0.7, 5.6],
        'V': [117.14, 105, 142, 1, 0, 1, 0, 0, 0, 0, 4.2, 6],
        'W': [204.22, 163, 240, 1, 0, 1, 0, 0, 0, 0, -0.9, 5.89],
        'Y': [181.19, 141, 203, 0, 1, 1, 0, 0, 0, 0, -1.3, 5.64]
    }
    return create_encoding(properties, num_fmt)


def aa_index(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create an AA-index encoding dictionary for amino acids.

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of AA-index encoded amino acids
    """
    # AA-index properties (shortened for brevity)
    properties = {
        'A': [-0.12035015, 5.4258137, 17.016745, 2.2042143, 2.8196585],
        'C': [8.227066, 8.151966, -6.015521, -14.363759, 12.445896],
        'D': [-18.502157, -3.0412152, -0.074919604, -1.6854501, 5.738219],
        'E': [-12.165897, -11.911348, 10.555523, 4.395205, 7.926744],
        'F': [19.688152, -0.37279683, -3.9456587, 1.698545, -1.9570603],
        'G': [-16.831188, 21.629177, 3.1235836, -7.335674, -9.4261465],
        'H': [-0.3357358, -8.323935, -4.663859, -4.9724045, 1.0845268],
        'I': [21.014881, 5.8575606, 2.4039905, 4.249276, -2.4844866],
        'K': [-11.812772, -13.541781, 5.459664, 1.9290336, -5.95619],
        'L': [17.919273, 3.0868933, 11.307291, 6.8597193, -2.107819],
        'M': [16.168451, -4.852773, 3.4575717, -2.0003242, 6.5592566],
        'N': [-15.427823, -0.0959064, -3.612455, -6.311836, -1.5995593],
        'P': [-16.50458, 11.8824, -16.928919, 19.22078, 6.0852327],
        'Q': [-7.977834, -9.140128, 1.7335126, -0.06425645, 1.7158138],
        'R': [-8.324453, -15.8011, -1.2679517, 0.054631874, -8.1227865],
        'S': [-12.768331, 7.992347, 1.6343912, -3.085485, -1.3099675],
        'T': [-4.8389106, 5.9711475, -0.06308909, -0.5623075, -1.2759036],
        'V': [16.193203, 8.5665655, 5.880625, 2.6137846, -1.8640387],
        'W': [17.825146, -8.438242, -13.067144, -1.6442173, -0.6060697],
        'Y': [8.573883, -3.044647, -12.933385, -1.1994684, -7.665342]
    }
    return create_encoding(properties, num_fmt)


def georgiev(num_fmt: str = 'float16') -> Dict[str, np.ndarray]:
    """
    Create a Georgiev encoding dictionary for amino acids.

    This encoding is based on the work by Georgiev (DOI: 10.1089/cmb.2008.0173),
    as implemented in ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345).

    Args:
        num_fmt (str): Numeric format for the encoding arrays

    Returns:
        Dict[str, np.ndarray]: Dictionary of Georgiev encoded amino acids
    """
    properties = {
        'A': [0.57, 3.37, -3.66, 2.34, -1.07, -0.4, 1.23, -2.32, -2.01, 1.31, -1.14, 0.19, 1.66, 4.39, 0.18, -2.6, 1.49, 0.46, -4.22],
        'C': [2.66, -1.52, -3.29, -3.77, 2.96, -2.23, 0.44, -3.49, 2.22, -3.78, 1.98, -0.43, -1.03, 0.93, 1.43, 1.45, -1.15, -1.64, -1.05],
        'D': [-2.46, -0.66, -0.57, 0.14, 0.75, 0.24, -5.15, -1.17, 0.73, 1.5, 1.51, 5.61, -3.85, 1.28, -1.98, 0.05, 0.9, 1.38, -0.03],
        'E': [-3.08, 3.45, 0.05, 0.62, -0.49, 0, -5.66, -0.11, 1.49, -2.26, -1.62, -3.97, 2.3, -0.06, -0.35, 1.51, -2.29, -1.47, 0.15],
        'F': [3.12, 0.68, 2.4, -0.35, -0.88, 1.62, -0.15, -0.41, 4.2, 0.73, -0.56, 3.54, 5.25, 1.73, 2.14, 1.1, 0.68, 1.46, 2.33],
        'G': [0.15, -3.49, -2.97, 2.06, 0.7, 7.47, 0.41, 1.62, -0.47, -2.9, -0.98, -0.62, -0.11, 0.15, -0.53, 0.35, 0.3, 0.32, 0.05],
        'H': [-0.39, 1, -0.63, -3.49, 0.05, 0.41, 1.61, -0.6, 3.55, 1.52, -2.28, -3.12, -1.45, -0.77, -4.18, -2.91, 3.37, 1.87, 2.17],
        'I': [3.1, 0.37, 0.26, 1.04, -0.05, -1.18, -0.21, 3.45, 0.86, 1.98, 0.89, -1.67, -1.02, -1.21, -1.78, 5.71, 1.54, 2.11, -4.18],
        'K': [-3.89, 1.47, 1.95, 1.17, 0.53, 0.1, 4.01, -0.01, -0.26, -1.66, 5.86, -0.06, 1.38, 1.78, -2.71, 1.62, 0.96, -1.09, 1.36],
        'L': [2.72, 1.88, 1.92, 5.33, 0.08, 0.09, 0.27, -4.06, 0.43, -1.2, 0.67, -0.29, -2.47, -4.79, 0.8, -1.43, 0.63, -0.24, 1.01],
        'M': [1.89, 3.88, -1.57, -3.58, -2.55, 2.07, 0.84, 1.85, -2.05, 0.78, 1.53, 2.44, -0.26, -3.09, -1.39, -1.02, -4.32, -1.34, 0.09],
        'N': [-2.02, -1.92, 0.04, -0.65, 1.61, 2.08, 0.4, -2.47, -0.07, 7.02, 1.32, -2.44, 0.37, -0.89, 3.13, 0.79, -1.54, -1.71, -0.25],
        'P': [-0.58, -4.33, -0.02, -0.21, -8.31, -1.82, -0.12, -1.18, 0, -0.66, 0.64, -0.92, -0.37, 0.17, 0.36, 0.08, 0.16, -0.34, 0.04],
        'Q': [-2.54, 1.82, -0.82, -1.85, 0.09, 0.6, 0.25, 2.11, -1.92, -1.67, 0.7, -0.27, -0.99, -1.56, 6.22, -0.18, 2.72, 4.35, 0.92],
        'R': [-2.8, 0.31, 2.84, 0.25, 0.2, -0.37, 3.81, 0.98, 2.43, -0.99, -4.9, 2.09, -3.08, 0.82, 1.32, 0.69, -2.62, -1.49, -2.57],
        'S': [-1.1, -2.05, -2.19, 1.36, 1.78, -3.36, 1.39, -1.21, -2.83, 0.39, -2.92, 1.27, 2.86, -1.88, -2.42, 1.75, -2.77, 3.36, 2.67],
        'T': [-0.65, -1.6, -1.39, 0.63, 1.35, -2.45, -0.65, 3.43, 0.34, 0.24, -0.53, 1.91, 2.66, -3.07, 0.2, -2.2, 3.73, -5.46, -0.73],
        'V': [2.64, 0.03, -0.67, 2.34, 0.64, -2.01, -0.33, 3.93, -0.21, 1.27, 0.43, -1.71, -2.93, 4.22, 1.06, -1.31, -1.97, -1.21, 4.77],
        'W': [1.89, -0.09, 4.21, -2.77, 0.72, 0.86, -1.07, -1.66, -5.87, -0.66, -2.49, -0.3, -0.5, 1.64, -0.72, 1.75, 2.73, -2.2, 0.9],
        'Y': [0.79, -2.62, 4.11, -0.63, 1.89, -0.53, -1.3, 1.31, -0.56, -0.95, 1.91, -1.26, 1.57, 0.2, -0.76, -5.19, -2.56, 2.87, -3.43],
    }
    
    return create_encoding(properties, num_fmt)


def encode_sequences(seq_list: List[str], mode: str, num_fmt: str = 'float16') -> np.ndarray:
    """
    Encode a list of protein sequences using the specified encoding method.

    Args:
        seq_list (List[str]): List of protein sequences to encode
        mode (str): Encoding method to use
        num_fmt (str, optional): Numeric format for the encoding arrays. Defaults to 'float16'.

    Returns:
        np.ndarray: Numpy array of encoded sequences

    Raises:
        TypeError: If seq_list is not a list or contains non-string elements
        ValueError: If mode is invalid or if any sequence is invalid
    """
    if not isinstance(seq_list, (list, tuple)):
        raise TypeError(f"Input must be a list or tuple of strings, got {type(seq_list).__name__} instead")

    if not all(isinstance(seq, str) for seq in seq_list):
        raise TypeError(f"All elements in the input must be strings, the first non-string element is of type {type(seq_list[next(i for i, x in enumerate(seq_list) if not isinstance(x, str))]).__name__}")

    available_modes = {
        "identity": one_hot,
        "heil_short": heil_short,
        "bork": bork,
        "t-scale": tscale,
        "property": property_encoding,
        "aa_index": aa_index,
        "georgiev": georgiev  # Add this line
    }

    if mode not in available_modes:
        raise ValueError(f"Invalid mode '{mode}'. Available modes are: {', '.join(available_modes.keys())}")

    codes = available_modes[mode](num_fmt)

    encoded_sequences = []
    max_length = 0
    for sequence in seq_list:
        if not is_valid_sequence(sequence):
            raise ValueError(f"Invalid sequence encountered: {sequence}")

        encoded_sequence = []
        for aa in sequence.upper():
            encoded_sequence.extend(codes.get(aa, codes['X']))
        
        encoded_sequences.append(encoded_sequence)
        max_length = max(max_length, len(encoded_sequence))

    # Pad sequences to the same length
    padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in encoded_sequences]

    return np.array(padded_sequences, dtype=num_fmt)
