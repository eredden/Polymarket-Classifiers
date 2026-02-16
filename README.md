# Polymarket Binary Option Classifier

## Description
This project demonstrates a classifier model for predicting the probability of a "over" bet for over/under options contracts based on various features from a Polymarket dataset. I am specifically aiming to see if there is a particular market bias for the "over" option based on the size of the betting line (e.g. over/under 240+) even when the teams and players are abstracted away. This program was written by Evan Redden (ID: 012248650) for the Advanced AI and ML (D683) course offered by Western Governors University.

## Usage
Make sure to install the required dependencies from the `requirements.txt` file at the root of this repository! *Note that this command assumes your present working directory is the root of this repository.*

```bash
pip install -r requirements.txt
```

The Polymarket bet option classification model can executed using the following commands in your local shell, with the `--data` argument requiring the path to the data CSV which contains the Polymarket data. *Note that this assumes your present working directory is the bet-option-classifier directory of this repository: if this is not the case, simply adjust the path supplied to the `--data` argument as needed.*

```bash
# Pre-process the raw polymarket_markets.csv dataset to get over/under bets into a separate dataset file.
python data-preprocessing.py --data "..\data\polymarket_markets.csv" --output "..\data\binary_polymarket_markets.csv"

# OPTIONAL: Get optimal hyperparameters for the XGBoost model.
python hyperparameter-optimization.py --data "..\data\binary_polymarket_markets.csv"

# Runs the model and shows you the AUC-ROC score, correlation heatmap, and SHAP dependence graphs.
# Note that images will be placed in the directory where this script is executed. You may want to 
# change your working directory to an images folder when running this.
python data-analysis.py --data "..\data\binary_polymarket_markets.csv"
```

The Polymarket line bias model can be executed using the following commands . . . 

**TODO:** Get this documented.

## Findings

I originally intended for this project to be used for predicting the outcomes of over/under bets on Polymarket. This model is capable of accurately predicting whether a given bet will be over or under the line 81% of the time. However, this is because the model is working with a dataset that was generated after the events had already ended. Most explicit data leaks have been removed, but the general sentiment from betting participants is still priced into the `lastTradePrice`, `liquidityIndex`, `spread`, and the various volume features.

```
[MODEL EVALUATION METRICS]
AUC-ROC SCORE: 0.8099730458221024
AUC-ROC SCORE (AS A PERCENTAGE): 81%
```

As a result, this model would likely be functionally useless when used against metrics from ongoing betting events through Polymarket. However, it may be useful for post-mortems of bets to see what features tend to correlate with particular betting options, e.g. seeing if there is a bias towards "over" bets for lines exceeding 240 . . .

**TODO:** Finish this section.