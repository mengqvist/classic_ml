# Protein Sequence Encoding Analysis

This project explores different methods of encoding protein sequences and tests various models using scikit-learn. It's designed as a learning project to understand the impact of different encoding techniques on model performance in protein analysis.

## Project Structure

```
project_root/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── encode.py
│   └── model.py
│
├── results/
│
├── tests/
│   └── __init__.py
│
├── environment.yml
├── README.md
└── .gitignore
```

## Encoding Methods

This project implements several protein sequence encoding methods:

1. **One-hot encoding**: A simple binary encoding where each amino acid is represented by a binary vector.

2. **Property encoding**: A comprehensive encoding based on various physicochemical properties of amino acids.

3. **Heil short encoding**: Based on Heil et al. 2006 (DOI: 10.1093/bioinformatics/btl132). It encodes amino acids based on five categories of properties.

4. **Bork encoding**: Based on Bork et al. 1990 (DOI: 10.1111/j.1432-1033.1990.tb19129.x). It uses eleven properties for encoding.

5. **T-scale encoding**: Based on Tian et al. 2006 (DOI: 10.1016/j.molstruc.2006.07.004). It uses five descriptors for each amino acid.

6. **AA-index encoding**: Based on a reduced set of AAIndex properties, as described in Gelman et al. 2021 (DOI: 10.1073/pnas.2104878118).

7. **Georgiev encoding**: Based on Georgiev et al. 2008 (DOI: 10.1089/cmb.2008.0173), as implemented in ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345). It uses 19 parameters for each amino acid.

## Setup

1. Clone this repository:
   ```
   git clone git@github.com:mengqvist/classic_ml.git
   cd classic_ml
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate classic_ml
   ```

## Usage

To run the analysis:

1. Navigate to the `notebooks/` directory.
2. Open the Jupyter notebook(s) using Jupyter Lab or Jupyter Notebook.
3. Run the cells in the notebook to perform the analysis.

The notebooks contain step-by-step processes for encoding protein sequences, applying different models, and analyzing the results.

## Contributing

This is primarily a learning project, but feedback and suggestions are greatly appreciated. If you have ideas for improvements or find any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.