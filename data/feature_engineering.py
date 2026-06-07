import pandas as pd
import numpy as np


# =========================
# 1. Load cleaned data
# =========================

sold = pd.read_csv("sold_cleaned.csv")
listing = pd.read_csv("listing_cleaned.csv")

print("Sold data shape:", sold.shape)
print("Listing data shape:", listing.shape)


# =========================
# 2. Helper functions
# =========================

def safe_divide(numerator, denominator):
    """
    Avoid division by zero or missing values.
    """
    return np.where(
        (denominator.notna()) & (denominator != 0),
        numerator / denominator,
        np.nan
    )


def parse_date_column(df, col):
    """
    Convert a column to datetime if it exists.
    """
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# =========================
# 3. Parse date columns
# =========================

date_cols = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate"
]

for col in date_cols:
    sold = parse_date_column(sold, col)


# =========================
# 4. Feature engineering
# =========================

# Price Ratio = ClosePrice / OriginalListPrice
if {"ClosePrice", "OriginalListPrice"}.issubset(sold.columns):
    sold["PriceRatio"] = safe_divide(
        sold["ClosePrice"],
        sold["OriginalListPrice"]
    )

# Close to Original List Ratio
# Same formula as Price Ratio, but kept as a separate business metric
if {"ClosePrice", "OriginalListPrice"}.issubset(sold.columns):
    sold["CloseToOriginalListRatio"] = safe_divide(
        sold["ClosePrice"],
        sold["OriginalListPrice"]
    )

# Price Per Sq Ft = ClosePrice / LivingArea
if {"ClosePrice", "LivingArea"}.issubset(sold.columns):
    sold["PricePerSqFt"] = safe_divide(
        sold["ClosePrice"],
        sold["LivingArea"]
    )

# Days on Market = raw DaysOnMarket field
if "DaysOnMarket" in sold.columns:
    sold["DaysOnMarket_Engineered"] = sold["DaysOnMarket"]

# Year / Month / YrMo derived from CloseDate
if "CloseDate" in sold.columns:
    sold["CloseYear"] = sold["CloseDate"].dt.year
    sold["CloseMonth"] = sold["CloseDate"].dt.month
    sold["YrMo"] = sold["CloseDate"].dt.to_period("M").astype(str)

# Listing to Contract Days = PurchaseContractDate - ListingContractDate
if {"PurchaseContractDate", "ListingContractDate"}.issubset(sold.columns):
    sold["ListingToContractDays"] = (
        sold["PurchaseContractDate"] - sold["ListingContractDate"]
    ).dt.days

# Contract to Close Days = CloseDate - PurchaseContractDate
if {"CloseDate", "PurchaseContractDate"}.issubset(sold.columns):
    sold["ContractToCloseDays"] = (
        sold["CloseDate"] - sold["PurchaseContractDate"]
    ).dt.days


# =========================
# 5. Sample output table
# =========================

sample_cols = [
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "PriceRatio",
    "CloseToOriginalListRatio",
    "PricePerSqFt",
    "DaysOnMarket_Engineered",
    "CloseDate",
    "CloseYear",
    "CloseMonth",
    "YrMo",
    "ListingContractDate",
    "PurchaseContractDate",
    "ListingToContractDays",
    "ContractToCloseDays"
]

sample_cols = [col for col in sample_cols if col in sold.columns]

sample_output = sold[sample_cols].head(10)

print("\nSample output table with engineered metrics:")
print(sample_output)


# =========================
# 6. Segment analysis
# =========================

# Option A: Group by PropertyType
if "PropertyType" in sold.columns:
    property_type_summary = sold.groupby("PropertyType").agg(
        NumberOfSales=("ClosePrice", "count"),
        AvgClosePrice=("ClosePrice", "mean"),
        MedianClosePrice=("ClosePrice", "median"),
        AvgPriceRatio=("PriceRatio", "mean"),
        AvgCloseToOriginalListRatio=("CloseToOriginalListRatio", "mean"),
        AvgPricePerSqFt=("PricePerSqFt", "mean"),
        AvgDaysOnMarket=("DaysOnMarket_Engineered", "mean"),
        AvgListingToContractDays=("ListingToContractDays", "mean"),
        AvgContractToCloseDays=("ContractToCloseDays", "mean")
    ).reset_index()

    print("\nSegmented summary by PropertyType:")
    print(property_type_summary)

else:
    property_type_summary = None
    print("\nPropertyType column not found. Skipping PropertyType summary.")


# Option B: Group by CountyOrParish
if "CountyOrParish" in sold.columns:
    county_summary = sold.groupby("CountyOrParish").agg(
        NumberOfSales=("ClosePrice", "count"),
        AvgClosePrice=("ClosePrice", "mean"),
        MedianClosePrice=("ClosePrice", "median"),
        AvgPriceRatio=("PriceRatio", "mean"),
        AvgCloseToOriginalListRatio=("CloseToOriginalListRatio", "mean"),
        AvgPricePerSqFt=("PricePerSqFt", "mean"),
        AvgDaysOnMarket=("DaysOnMarket_Engineered", "mean"),
        AvgListingToContractDays=("ListingToContractDays", "mean"),
        AvgContractToCloseDays=("ContractToCloseDays", "mean")
    ).reset_index()

    print("\nSegmented summary by CountyOrParish:")
    print(county_summary)

else:
    county_summary = None
    print("\nCountyOrParish column not found. Skipping County summary.")


# =========================
# 7. Optional competitive intelligence summaries
# =========================

if "ListOfficeName" in sold.columns:
    list_office_summary = sold.groupby("ListOfficeName").agg(
        NumberOfSales=("ClosePrice", "count"),
        AvgClosePrice=("ClosePrice", "mean"),
        AvgPriceRatio=("PriceRatio", "mean"),
        AvgDaysOnMarket=("DaysOnMarket_Engineered", "mean")
    ).reset_index()

    list_office_summary = list_office_summary.sort_values(
        by="NumberOfSales",
        ascending=False
    )

    print("\nTop listing offices summary:")
    print(list_office_summary.head(10))


if "BuyerOfficeName" in sold.columns:
    buyer_office_summary = sold.groupby("BuyerOfficeName").agg(
        NumberOfPurchases=("ClosePrice", "count"),
        AvgClosePrice=("ClosePrice", "mean"),
        AvgPriceRatio=("PriceRatio", "mean"),
        AvgDaysOnMarket=("DaysOnMarket_Engineered", "mean")
    ).reset_index()

    buyer_office_summary = buyer_office_summary.sort_values(
        by="NumberOfPurchases",
        ascending=False
    )

    print("\nTop buyer offices summary:")
    print(buyer_office_summary.head(10))


# =========================
# 8. Save outputs
# =========================

sold.to_csv("sold_week6_engineered.csv", index=False)
sample_output.to_csv("week6_sample_output.csv", index=False)

if property_type_summary is not None:
    property_type_summary.to_csv("week6_property_type_summary.csv", index=False)

if county_summary is not None:
    county_summary.to_csv("week6_county_summary.csv", index=False)

print("\nFiles saved:")
print("- sold_week6_engineered.csv")
print("- week6_sample_output.csv")
print("- week6_property_type_summary.csv, if PropertyType exists")
print("- week6_county_summary.csv, if CountyOrParish exists")