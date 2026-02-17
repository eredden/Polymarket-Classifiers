# data-analysis.py - Used for generating graphs for the Polymarket dataset we
# use in the other Python scripts, written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
from pathlib import Path
import seaborn
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys

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

    # Instantiate the LogisticRegression classifier.
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    # Fit the classifier to the training data.
    model.fit(X_train, y_train)

    # Get the probability for the "1" class (e.g., over wins the bet).
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate the AUC-ROC score. It is a good metric for determining 
    # classification accuracy with a skewed dataset like ours where the ratio of 
    # under to over bets is lopsided.
    auc_score = roc_auc_score(y_test, y_probs)

    # Output the global model evaluation metrics!
    print("[GLOBAL MODEL EVALUATION METRICS]")
    print(f"AUC-ROC SCORE: {auc_score}")

    # Combine test features, actual results, and predictions back into one 
    # DataFrame for analysis.
    analysis = X_test.copy()
    analysis["actual"] = y_test
    analysis["predicted"] = model.predict(X_test)
    analysis["is_correct"] = (analysis["actual"] == analysis["predicted"]).astype(int)

    # Create a bucket of rows for each sport category and evaluate the model"s 
    # prediction accuracy for each.
    sport_cols = [col for col in X.columns if col.startswith("sport_")]
    sport_edge_results = []

    for sport in sport_cols:
        # Filter for rows where this sport category is True (1).
        subset = analysis[analysis[sport] == 1]

        if len(subset) > 0:
            # The edge is simply the accuracy minus 0.50, effectively the
            # advantage over predicting with random chance.
            accuracy = roc_auc_score(
                subset["actual"], 
                subset["predicted"]
            )
            edge = accuracy - 0.50
        
            sport_edge_results.append({
                "bucket": sport,
                "sample_size": len(subset),
                "accuracy": accuracy,
                "edge": edge,
                "edge_percentage": f"{edge:.0%}"
            })

    sport_df = pandas.DataFrame(sport_edge_results).sort_values(
        by="edge", 
        ascending=False
    )

    print("\n[MODEL EDGE BY SPORT]")
    print(sport_df)

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
        "line",
        "days_elapsed"
    ]

    cmap = plt.colormaps["rocket"]

    for feature in shap_dependence_features:
        shap.dependence_plot(
            feature, 
            shap_values, 
            X_test,
            show=False,
            cmap=cmap
        )

        plt.savefig(f"shapley-{feature}-dependence.png", transparent=False)
        plt.close()