# Polymarket Binary Option Classifier

## Description
This project demonstrates a classifier model for predicting the outcome of a binary option contract based on various features from a Polymarket dataset. This program was written by Evan Redden (ID: 012248650) for the Advanced AI and ML (D683) course offered by Western Governors University.

## Usage
Make sure to install the required dependencies from the `requirements.txt` file at the root of this repository! *Note that this command assumes your present working directory is the root of this repository.*

```bash
pip install -r requirements.txt
```

The Polymarket classification model can executed using the following command in your local shell, with the `--data` argument requiring the path to the data CSV which contains the Polymarket data. *Note that this assumes your present working directory is the root of this repository: if this is not the case, simply adjust the path supplied to the `--data` argument as needed.*

```bash
python .\src\main.py --data ".\data\data.csv"
```