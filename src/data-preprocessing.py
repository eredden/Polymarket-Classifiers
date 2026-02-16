# data-preprocessing.py - Preprocesses the Polymarket markets dataset so that
# it can be parsed more effectively by the Polymarket classifier model.
# Written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
import ast
import pandas
from pathlib import Path
import sys

EXCLUDED_COLUMNS = [
    "acceptingOrders", 
    "acceptingOrdersTimestamp", 
    "active", 
    "approved", 
    "archived", 
    "automaticallyActive", 
    "automaticallyResolved", 
    "bestAsk",
    "bestBid",
    "category", 
    "categoryMailchimpTag", 
    "clearBookOnStart", 
    "clobRewards", 
    "clobTokenIds", 
    "closed",
    "closedTime",
    "commentsEnabled", 
    "competitive", 
    "conditionId",
    "createdAt", 
    "createdBy", 
    "creator", 
    "customLiveness", 
    "cyom", 
    "denominationToken", 
    "deploying", 
    "deployingTimestamp",
    "description", 
    "disqusThread", 
    "enableOrderBook",
    "endDate",
    "endDateIso",
    "event_id",
    "event_slug",
    "event_title",
    "eventStartTime",
    "featured",
    "fee", 
    "feesEnabled", 
    "formatType",
    "fpmmLive", 
    "funded", 
    "gameId", 
    "gameStartTime", 
    "groupItemRange", 
    "groupItemThreshold", 
    "groupItemTitle",
    "hasReviewedDates", 
    "holdingRewardsEnabled", 
    "icon", 
    "id", 
    "image",
    "liquidity",
    "liquidityAmm",
    "liquidityClob",
    "liquidityNum",
    "lowerBound",
    "mailchimpTag", 
    "makerBaseFee", 
    "manualActivation", 
    "marketGroup", 
    "marketMakerAddress", 
    "marketType", 
    "negRisk",
    "negRiskOther",
    "negRiskMarketID", 
    "negRiskRequestID", 
    "new",
    "notificationsEnabled", 
    "oneDayPriceChange",
    "oneHourPriceChange",
    "oneMonthPriceChange",
    "oneWeekPriceChange",
    "oneYearPriceChange",
    "orderMinSize",
    "pagerDutyNotificationEnabled", 
    "pendingDeployment", 
    "question",
    "questionID", 
    "ready", 
    "readyForCron", 
    "resolutionSource",
    "resolvedBy",
    "restricted",
    "umaResolutionStatuses",
    "rfqEnabled", 
    "secondsDelay", 
    "sentDiscord", 
    "seriesColor", 
    "showGmpOutcome", 
    "showGmpSeries", 
    "slug", 
    "sponsorImage", 
    "sportsMarketType", 
    "startDate",
    "subcategory", 
    "submitted_by",
    "takerBaseFee",
    "teamAID",
    "teamBID",
    "twitterCardImage",
    "twitterCardLastRefreshed",
    "twitterCardLastValidated",
    "twitterCardLocation",
    "umaBond",
    "umaEndDateIso",
    "umaReward",
    "updatedAt",
    "updatedBy",
    "upperBound",
    "volume1moAmm",
    "volume1moClob",
    "volume1wkAmm",
    "volume1yrAmm",
    "volume24hrAmm",
    "volumeAmm",
    "wideFormat"
]

FLOAT_COLUMNS = [
    "lastTradePrice",
    "orderPriceMinTickSize",
    "volume",
    "volume1mo",
    "volume1wk",
    "volume1wkClob",
    "volume1yr",
    "volume1yrClob",
    "volume24hr",
    "volume24hrClob",
    "volumeClob",
    "volumeNum"
]

CALC_COLUMNS = [
    "endDate",
    "outcomes",
    "outcomePrices",
    "startDate",
    "startDateIso",
    "umaEndDate",
    "umaResolutionStatus"
]

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

# Determine the winning result by checking the prices of the outcomes.
def determine_result(row):
    try:
        return row["outcomePrices"].index("0")
    except (ValueError, IndexError):
        return pandas.NA

# This function pre-processes the data.
def preprocess_data(data: pandas.DataFrame) -> pandas.DataFrame:
    # Drop redundant, irrelevant, and/or mostly blank feature columns.
    data.drop(
        columns=EXCLUDED_COLUMNS, 
        axis=1, 
        inplace=True
    )

    # Fill in null values for important feature columns.
    data[FLOAT_COLUMNS] = data[FLOAT_COLUMNS].astype("float64").fillna(0.0)

    # Filter for resolved market contracts that have over/under outcomes.
    # Note that the Pandas dataframe stores lists as objects rather than as 
    # Python lists, so we have to check against string types as a result.
    mask_closed = data["umaResolutionStatus"] == "resolved"
    mask_binary = data["outcomePrices"].astype(str).isin([
        "[\"0\", \"1\"]", 
        "[\"1\", \"0\"]"
    ])
    mask_over_under = data["outcomes"].astype(str) == "[\"Over\", \"Under\"]"

    # Apply the filtering masks and drop any rows that still have null values.
    data = data.loc[mask_closed & mask_binary & mask_over_under].dropna()

    # Remove settled markets to prevent extreme target leakage.
    data = data[(data["lastTradePrice"] > 0.01) & (data["lastTradePrice"] < 0.99)]

    # Convert the outcomes and outcome prices strings into lists.
    data["outcomePrices"] = data["outcomePrices"].apply(ast.literal_eval)
    data["outcomes"] = data["outcomes"].apply(ast.literal_eval)

    # Generate over column to show whether the "Over" option won or not.
    # This will be the target feature for our classification model.
    data["over"] = data.apply(determine_result, axis=1)

    # Convert dates to datetime objects and calculate a daysElapsed value.
    data["endDate"] = pandas.to_datetime(
        data["umaEndDate"], 
        format="mixed",
        utc=True
    )

    data["startDate"] = pandas.to_datetime(
        data["startDateIso"],
        format="mixed",
        utc=True
    )

    data["daysElapsed"] = (data["endDate"] - data["startDate"]).dt.days

    # Drop the columns we used for calculations since they are no longer needed.
    data.drop(
        columns=CALC_COLUMNS, 
        axis=1, 
        inplace=True
    )

    # Create a relative volume feature for 24 hour volume v.s. all volume to
    # show how much is changing RIGHT NOW!
    # 1e-9 added to volume to make sure we never divide by zero.
    data["relativeVolume"] = data["volume24hr"] / (data["volume"] + 1e-9)

    # Liquidity index to see how much volume is distributed over the bet spread.
    # Tighter spreads with more volume are generally better.
    data["liquidityIndex"] = data["volume"] / (data["spread"] + 1e-9)

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

    parser.add_argument(
        "--output",
        help="Output path for the processed CSV data file.",
        type=Path
    )
    
    args = parser.parse_args()

    # Load in the market dataset and process it.
    data = load_data(args.data)
    processed_data = preprocess_data(data)

    # Save the filtered data to a separate CSV in the current directory.
    processed_data.to_csv(
        path_or_buf=args.output,
        index=False
    )

    # For debugging purposes, output info about the new filtered_data frame.
    print(processed_data.info())