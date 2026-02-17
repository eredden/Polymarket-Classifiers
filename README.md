# Polymarket Binary Option Classifier

## Description
This project demonstrates a classifier model for predicting the probability of a "over" bet for over/under options contracts based on various features from a Polymarket dataset. I am specifically aiming to see if there is a particular market bias for the "over" option based on the size of the betting line (e.g. over/under 240+) even when the teams and players are abstracted away. This program was written by Evan Redden (ID: 012248650) for the Advanced AI and ML (D683) course offered by Western Governors University.

## Usage
Make sure to install the required dependencies from the `requirements.txt` file at the root of this repository! *Note that this command assumes your present working directory is the root of this repository.*

```bash
pip install -r requirements.txt
```

The Polymarket bet option classification model can executed using the following commands in your local shell, with the `--data` argument requiring the path to the data CSV which contains the Polymarket data. *Note that this assumes your present working directory is the .\src\bet-option-classifier directory of this repository: if this is not the case, simply adjust the path supplied to the `--data` argument as needed.*

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

The Polymarket line bias classification model can executed using the following commands in your local shell, with the `--data` argument requiring the path to the data CSV which contains the Polymarket data. *Note that this assumes your present working directory is the .\src\line-bias-classifier directory of this repository: if this is not the case, simply adjust the path supplied to the `--data` argument as needed.*

```bash
# Pre-process the raw polymarket_markets.csv dataset to get over/under bets into a separate dataset file.
python data-preprocessing.py --data "..\..\data\polymarket_markets.csv" --output "..\..\data\binary_polymarket_markets.csv"

# OPTIONAL: Get optimal hyperparameters for the XGBoost model.
python hyperparameter-optimization.py --data "..\..\data\binary_polymarket_markets.csv"

# Runs the model and shows you the AUC-ROC score, correlation heatmap, and SHAP dependence graphs.
# Note that images will be placed in the directory where this script is executed. You may want to 
# change your working directory to an images folder when running this.
python data-analysis.py --data "..\..\data\binary_polymarket_markets.csv"
```

## Findings

I originally intended for this project to be used for predicting the outcomes of over/under bets on Polymarket. This model is capable of accurately predicting whether a given bet will be over or under the line 81% of the time. However, this is because the model is working with a dataset that was generated after the events had already ended. Most explicit data leaks have been removed, but the general sentiment from betting participants is still priced into the `lastTradePrice`, `liquidityIndex`, `spread`, and the various volume features.

```
[MODEL EVALUATION METRICS]
AUC-ROC SCORE: 0.8099730458221024
AUC-ROC SCORE (AS A PERCENTAGE): 81%
```

As a result, this model would likely be functionally useless when used against metrics from ongoing betting events through Polymarket. However, it may be useful for post-mortems of bets to see what features tend to correlate with particular betting options. To see if there is any statisically significant bias from gamblers based on the line of the bet, I created a new line bias classification model which exclusively uses the `line` feature, `daysElapsed` feature, and one-hot encoded sports categories to determine the AUC-ROC score when predicting the `over` target feature. 

```
[MODEL EVALUATION METRICS]
AUC-ROC SCORE: 0.5624034872424196
AUC-ROC SCORE (AS A PERCENTAGE): 56%
```

With an AUC-ROC score of 56% when reviewing over 10,000 binary options bets with over/under options, it is clear that there is a statistically significant bias caused exclusively by the line, duration, and category of the bet in question. In the context of sports betting, where markets are generally efficient, it is incredibly interesting to see that knowledge of these features alone results in a 6% edge over random guessing in predicting the result of a bet. 

Moreover, reviewing the SHAP dependency graphs show that these edges tend to be stratified as well! In the SHAP values summary graph, you can see that the `sport_football` and `sport_other` features have a significant amount of sway on the ultimate SHAP prediction compared to other category features. I split the categories into buckets to determine the edge (accuracy - 0.50) that the model has for each category, finding the following results:

```
[MODEL EDGE BY SPORT]
                        bucket  sample_size  accuracy      edge edge_percentage
7                  sport_other          146  0.624246  0.124246             12%
8                 sport_soccer          160  0.594952  0.094952              9%
4               sport_football          853  0.586573  0.086573              9%
1             sport_basketball          900  0.533091  0.033091              3%
0               sport_baseball            3  0.500000  0.000000              0%
6                 sport_hockey           73  0.500000  0.000000              0%
3                sport_esports          154  0.500000  0.000000              0%
2  sport_basketball_first_half          167  0.489638 -0.010362             -1%
5   sport_football_player_prop           46  0.441520 -0.058480             -6%
```

The edges range from approximately -6% for bets on `sport_football_player_prop` category rows to +12% for the `sport_other` category rows! `sport_soccer` and `sport_football` both boast a significant edge of +9% as well! As a point of reference, the break-even point for standard sports betting with -110 odds is a 52.38% win rate due to sportsbooks charging "juice" or "vig" fees on bets. The `sport_football` accuracy (i.e. win rate) of 58.65% would be more than enough to generate a significant profit over time. While comparing the `sport_other` and `sport_soccer` edges is tempting, it should be kept in mind that both of these have sample sizes under 200 bets for our dataset and may not be a representative sample of their actual edges.