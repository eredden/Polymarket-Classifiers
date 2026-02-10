# data-preprocessing.py - Preprocesses the Polymarket markets dataset so that
# it can be parsed more effectively by the Polymarket classifier model.
# Written by Evan Redden (ID: 012248650).

from argparse import ArgumentParser
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
    "category", 
    "categoryMailchimpTag", 
    "clearBookOnStart", 
    "clobRewards", 
    "clobTokenIds", 
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
    "disqusThread", 
    "endDate",
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
    "umaEndDateIso",
    "updatedBy",
    "upperBound",
    "wideFormat"
]

BOOL_COLUMNS = [
    "enableOrderBook",
    "featured",
    "negRisk",
    "new"
]

FLOAT_COLUMNS = [
    "bestBid",
    "liquidity",
    "liquidityAmm",
    "liquidityClob",
    "liquidityNum",
    "oneDayPriceChange",
    "oneHourPriceChange",
    "oneMonthPriceChange",
    "oneWeekPriceChange",
    "oneYearPriceChange",
    "orderMinSize",
    "orderPriceMinTickSize",
    "umaBond",
    "umaReward",
    "volume",
    "volume1mo",
    "volume1moAmm",
    "volume1moClob",
    "volume1wk",
    "volume1wkAmm",
    "volume1wkClob",
    "volume1yr",
    "volume1yrAmm",
    "volume1yrClob",
    "volume24hr",
    "volume24hrAmm",
    "volume24hrClob",
    "volumeAmm",
    "volumeClob",
    "volumeNum"
]

STR_COLUMNS = [
    "resolutionSource",
    "resolvedBy"
]

TIME_COLUMNS = [
    "endDate",
    "startDate",
    "startDateIso",
    "umaEndDate"
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

# This function pre-processes the data.
def preprocess_data(data: pandas.DataFrame) -> pandas.DataFrame:
    # Drop redundant, irrelevant, and/or mostly blank feature columns.
    data.drop(
        columns=EXCLUDED_COLUMNS, 
        axis=1, 
        inplace=True
    )

    # Fill in null values for important feature columns.
    data[BOOL_COLUMNS] = data[BOOL_COLUMNS].astype(bool).fillna(False)
    data[FLOAT_COLUMNS] = data[FLOAT_COLUMNS].astype("float64").fillna(0.0)
    data[STR_COLUMNS] = data[STR_COLUMNS].astype(str).fillna("")

    # Filter for resolved market contracts that have boolean outcomes.
    # Note that the Pandas dataframe stores lists as objects rather than as 
    # Python lists, so we have to check against string types as a result.
    mask_closed = data["umaResolutionStatus"] == "resolved"
    mask_binary = data["outcomePrices"].astype(str).isin([
        "[\"0\", \"1\"]", 
        "[\"1\", \"0\"]"
    ])

    # Apply the filtering masks and drop any rows that still have null values.
    data = data.loc[mask_closed & mask_binary].dropna()

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

    # Drop the time columns since they are no longer needed.
    data.drop(
        columns=TIME_COLUMNS, 
        axis=1, 
        inplace=True
    )

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