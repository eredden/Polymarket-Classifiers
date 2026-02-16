# data-analysis.py - Used for generating graphs for the Polymarket dataset we
# use in the other Python scripts, written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
from pathlib import Path
import seaborn
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys
from xgboost import XGBClassifier

# This function validates that the CSV exists and can be read.
def load_data(file: str) -> pandas.DataFrame:
    if not file.exists():
        print(f"{file} does not exist.")
        sys.exit(1)
        
    if not file.is_file():
        print(f"{file} is a directory.")
        sys.exit(1)

    try:
        data = pandas.read_csv(
            filepath_or_buffer=file,
            index_col=False
        )
    except:
        print(f"{file} failed to be read by Pandas.")
        sys.exit(1)
    
    return data

if __name__ == "__main__":
    # Makes this Python script accept an argument for the data path rather than
    # hard-coding it in. Using relative paths gets hairy depending on the PWD,
    # and absolute paths would make this program unusable without the directory 
    # hierarchy being perfectly replicated.
    # e.g. python main.py --data "data.csv"
    parser = ArgumentParser(
        description="Predicts outcomes of Polymarket binary options contracts."
    )
    parser.add_argument(
        "--data", 
        help="Path to the CSV data file.", 
        type=Path
    )
    args = parser.parse_args()

    # Splits data into features that we use for prediction (i.e. X) and the 
    # target metric "over" (i.e. Y) for comparing predictions to actual
    # results.
    data = load_data(args.data)

    y = data["over"]
    X = data.drop(labels="over", axis=1)

    # Instantitates training and testing splits, which prevent overfitting by
    # using some of the data exclusively for training, and the rest exclusively
    # for testing. I arbitrarily chose to use 25% of the data for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.25, # Percentage of the data to include in the test split.
        random_state=42 # A seed used for RNG, which ensures reproducibility.
    )

    # Instantiate the XGBoost classifier and a hyperparameter search space.
    # This is the search space that we will randomly search through to find the
    # best hyperparameters for the XGBoost classifier.
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc", # ROC-AUC score used for determining model effectiveness.
        learning_rate=0.1, # The rate at which the gradient can be adjusted.
        n_estimators=35, # The number of sequential trees to build.
        max_depth=7 # The amount of layers in a given tree.
    )

    # Fit the classifier to the training data.
    model.fit(X_train, y_train)

    # Get the probability for the "1" class (e.g., over wins the bet).
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate the AUC-ROC score. It is a good metric for determining 
    # classification accuracy with a skewed dataset like ours where the ratio of 
    # under to over bets is lopsided.
    auc_score = roc_auc_score(y_test, y_probs)

    # Output the model evaluation metrics!
    print("[MODEL EVALUATION METRICS]")
    print(f"AUC-ROC SCORE: {auc_score}")

    # This correlation heatmap code was ripped directly from: 
    # https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/
    plt.figure(figsize=(20, 20))

    seaborn.heatmap(
        data.corr(), 
        annot=True, 
        fmt=".2f", 
        cmap="rocket", 
        linewidths=2
    )

    plt.title("Correlation Heatmap")
    plt.savefig("correlation-heatmap.png", transparent=False)
    plt.close()

    # Initialize the SHAP Explainer to get Shapley values for the features from 
    # the X_test dataset, then generate a plot and save it as an image.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values, 
        X_test, 
        show=False,
        cmap="rocket"
    )

    plt.savefig("shapley-values.png", transparent=False)
    plt.close()

    # Generate SHAP dependence graphs for these features.
    shap_dependence_features = [
        "lastTradePrice",
        "spread"
    ]

    for feature in shap_dependence_features:
        shap.dependence_plot(
            feature, 
            shap_values, 
            X_test,
            show=False,
            cmap="rocket"
        )

        plt.savefig(f"shapley-{feature}-dependence.png", transparent=False)
        plt.close()