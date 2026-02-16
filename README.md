# Polymarket Binary Option Classifier

## Description
This project demonstrates a classifier model for predicting the outcome of a over/under options contract based on various features from a Polymarket dataset. I am specifically aiming to see if there is a particular market bias for the option based on the size of the betting line (e.g. over/under 240+) even when the teams and sports are abstracted away. This program was written by Evan Redden (ID: 012248650) for the Advanced AI and ML (D683) course offered by Western Governors University.

## Usage
Make sure to install the required dependencies from the `requirements.txt` file at the root of this repository! *Note that this command assumes your present working directory is the root of this repository.*

```bash
pip install -r requirements.txt
```

The Polymarket classification model can executed using the following commands in your local shell, with the `--data` argument requiring the path to the data CSV which contains the Polymarket data. *Note that this assumes your present working directory is the src directory of this repository: if this is not the case, simply adjust the path supplied to the `--data` argument as needed.*

```bash
# Pre-process the raw polymarket_markets.csv dataset to get over/under bets into a separate dataset file.
python data-preprocessing.py --data "..\data\polymarket_markets.csv" --output "..\data\binary_polymarket_markets.csv"

# OPTIONAL: Get optimal hyperparameters for the XGBoost model.
python hyperparameter-optimization.py --data "..\data\binary_polymarket_markets.csv"

# Train the binary classification model and see the test results.
python main.py --data "..\data\binary_polymarket_markets.csv"

# Generate images showing the relationships between select features and the target feature "over."
python data-analysis.py --data "..\data\binary_polymarket_markets.csv"
```