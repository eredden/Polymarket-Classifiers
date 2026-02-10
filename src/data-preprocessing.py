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
    "startDateIso",
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

# This function is dedicated to validating that the CSV exists and can be read.
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

    parser.add_argument(
        "--output",
        help="Output path for the processed CSV data file.",
        type=Path
    )
    
    args = parser.parse_args()

    # Load in the market dataset.
    data = load_data(args.data)

    # Drop the columns we do not care for. These are removed because they have
    # little relevance to the betting odds or have far too few values filled in.
    data.drop(
        columns=EXCLUDED_COLUMNS, 
        axis=1, 
        inplace=True
    )

    # These features need to have null values filled in with intuitive 
    # substitute values. I generally interpret blanks here as false or zero.
    for bool_column in BOOL_COLUMNS:
        data[bool_column] = data[bool_column].astype(bool)
        data[bool_column] = data[bool_column].fillna(False)
    
    for float_column in FLOAT_COLUMNS:
        data[float_column] = data[float_column].astype("float64")
        data[float_column] = data[float_column].fillna(0.0)

    for str_column in STR_COLUMNS:
        data[str_column] = data[str_column].astype(str)
        data[str_column] = data[str_column].fillna("")

    # Filter for resolved market contracts that have boolean outcomes.
    # Note that the Pandas dataframe stores lists as  objects rather than as 
    # Python lists, so we have to check against string types as a result.
    mask_closed = (data["umaResolutionStatus"] == "resolved")
    mask_01 = (data["outcomePrices"].astype(str) == "[\"0\", \"1\"]")
    mask_10 = (data["outcomePrices"].astype(str) == "[\"1\", \"0\"]")

    filtered_data = data.loc[mask_closed & (mask_01 | mask_10)]

    # Drop all rows that still contain null values.
    filtered_data = filtered_data.dropna()

    # Save the filtered data to a separate CSV in the current directory.
    filtered_data.to_csv(
        path_or_buf=args.output,
        index=False
    )

    # For debugging purposes, output info about the new filtered_data frame.
    print(filtered_data.info())