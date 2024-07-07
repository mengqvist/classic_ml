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
│   ├── feature_engineering.py
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