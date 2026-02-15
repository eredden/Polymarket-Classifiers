# hyperparameter-optimization.py - Tunes the hyperparameters for the model using 
# a randomized search with cross-validation to see which values work best.
# Written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
import pandas
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import sys
from xgboost import XGBRegressor

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
    # target metric "under" (i.e. Y) for comparing predictions to actual
    # results.
    data = load_data(args.data)

    y = data["under"]
    X = data.drop(labels="under", axis=1)

    # Instantitates training and testing splits, which prevent overfitting by
    # using some of the data exclusively for training, and the rest exclusively
    # for testing. I arbitrarily chose to use 25% of the data for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.25, # Percentage of the data to include in the test split.
        random_state=42 # A seed used for RNG, which ensures reproducibility.
    )

    # Instantiate the XGBoost regressor and a hyperparameter search space.
    # This is the search space that we will randomly search through to find the
    # best hyperparameters for the XGBoost regressor.
    regressor = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.1 # The rate at which the gradient can be adjusted.
    )

    model_params = {
        "n_estimators": randint(10, 150), # The number of sequential trees to build.
        "max_depth": randint(8, 9) # The amount of layers in a given tree.
    }

    model = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=model_params,
        n_iter=100, # The number of models to generate and test.
        cv=5, # The number of slices of data to supply to each individual model.
        random_state=42 # A seed used for RNG, which ensures reproducibility.
    )

    # Fit the regressor to the training data.
    model.fit(X_train, y_train)

    # Make predictions on the test data.
    y_pred = model.predict(X_test)

    # Output the model evaluation metrics!
    # R^2 is the variation of the dependent variable (e.g. under) 
    # explained by the variation in the independent variables (i.e. features).
    # RMSE is the average of the differences between the predicted and actual 
    # values using the same units as the target variable.
    mse = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("[MODEL EVALUATION METRICS]")
    print(f"R^2: {mse}")
    print(f"R^2 as a percentage: {mse * 100}%")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Output the winning hyperparameters.
    best_params = model.best_estimator_.get_params()

    print("\n[BEST HYPERPARAMETER VALUES]")
    print(f"n_estimators: {best_params["n_estimators"]}")
    print(f"max_depth: {best_params["max_depth"]}")