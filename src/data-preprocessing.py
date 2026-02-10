# data-preprocessing.py - Preprocesses the Polymarket markets dataset so that
# it can be parsed more effectively by the Polymarket classifier model.
# Written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
import pandas
from pathlib import Path
import sys

# This function is dedicated to validating that:
# 1. The data path actually exists.
# 2. The data path points to a file.
# 3. That Pandas can successfully parse this as a CSV-formatted file.
def load_data(file: str) -> pandas.DataFrame:
    if not file.exists():
        print(f"{file} does not exist.")
        sys.exit(1)
        
    if not file.is_file():
        print(f"{file} is a directory.")
        sys.exit(1)

    try:
        data = pandas.read_csv(filepath_or_buffer=file)
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

    # Load in the market dataset.
    data = load_data(args.data)

    # These features were excluded due to large amounts of null entries,
    # entries that all had the same values, and due to perceived redundancy or 
    # irrelevance.
    excluded_columns = [
        "acceptingOrders", 
        "acceptingOrdersTimestamp", 
        "active", 
        "approved", 
        "archived", 
        "automaticallyActive", 
        "automaticallyResolved", 
        "category", 
        "categoryMailchimpTag", 
        "clearBookOnStart", 
        "clobRewards", 
        "clobTokenIds", 
        "commentsEnabled", 
        "competitive", 
        "conditionId", 
        "createdBy", 
        "creator", 
        "customLiveness", 
        "cyom", 
        "denominationToken", 
        "deploying", 
        "deployingTimestamp", 
        "disqusThread", 
        "endDateIso",
        "eventStartTime",
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
        "lastTradePrice", 
        "lowerBound",
        "line", 
        "mailchimpTag", 
        "makerBaseFee", 
        "manualActivation", 
        "marketGroup", 
        "marketMakerAddress", 
        "marketType", 
        "negRiskMarketID", 
        "negRiskRequestID", 
        "notificationsEnabled", 
        "pagerDutyNotificationEnabled", 
        "pendingDeployment", 
        "questionID", 
        "ready", 
        "readyForCron", 
        "rfqEnabled", 
        "secondsDelay", 
        "sentDiscord", 
        "seriesColor", 
        "showGmpOutcome", 
        "showGmpSeries", 
        "slug", 
        "sponsorImage", 
        "sportsMarketType", 
        "subcategory", 
        "submitted_by",
        "takerBaseFee",
        "teamAID",
        "teamBID",
        "twitterCardImage",
        "twitterCardLastRefreshed",
        "twitterCardLastValidated",
        "twitterCardLocation",
        "umaEndDateIso",
        "updatedBy",
        "upperBound",
        "wideFormat"
    ]

    data.drop(
        columns=excluded_columns, 
        axis=1, 
        inplace=True
    )

    # These features need to have null values filled in with intuitive 
    # substitute values. I generally interpret blanks here as false or zero.
    data["enableOrderBook"] = data["enableOrderBook"].astype(bool)
    data["enableOrderBook"] = data["enableOrderBook"].fillna(False)

    data["featured"] = data["featured"].astype(bool)
    data["featured"] = data["featured"].fillna(False)

    data["negRisk"] = data["negRisk"].astype(bool)
    data["negRisk"] = data["negRisk"].fillna(False)

    data["new"] = data["new"].astype(bool)
    data["new"] = data["new"].fillna(False)

    data["orderMinSize"] = data["orderMinSize"].astype("float64")
    data["orderMinSize"] = data["orderMinSize"].fillna(0)

    data["orderPriceMinTickSize"] = data["orderPriceMinTickSize"].astype("float64")
    data["orderPriceMinTickSize"] = data["orderPriceMinTickSize"].fillna(0)

    # Filter for closed market contracts that have boolean outcomes.
    # Note that the CSV stores lists as string objects rather than as Python
    # lists, so we have to check against strings as a result.
    mask_closed = (data["closed"] == True)
    mask_01 = (data["outcomePrices"].astype(str) == "[\"0\", \"1\"]")
    mask_10 = (data["outcomePrices"].astype(str) == "[\"1\", \"0\"]")

    filtered_data = data.loc[mask_closed & (mask_01 | mask_10)]

    # Save the filtered data to a separate CSV in the current directory.
    filtered_data.to_csv(path_or_buf="binary_polymarket_markets.csv")

    print(filtered_data.info())