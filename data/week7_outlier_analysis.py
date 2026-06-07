import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================

# Use Week 6 engineered dataset if available
# If not, change this to "sold_cleaned.csv"
df = pd.read_csv("sold_week6_engineered.csv")

print("Original dataset shape:", df.shape)


# =========================
# 2. Helper function: IQR flagging
# =========================

def add_iqr_outlier_flags(data, column):
    """
    Add IQR-based outlier flag columns for a numeric variable.
    This function does NOT delete records.
    It creates:
    - column_IQR_Lower
    - column_IQR_Upper
    - column_IsOutlier
    """

    if column not in data.columns:
        print(f"{column} not found. Skipping.")
        return data

    data[column] = pd.to_numeric(data[column], errors="coerce")

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data[f"{column}_IQR_Lower"] = lower
    data[f"{column}_IQR_Upper"] = upper

    data[f"{column}_IsOutlier"] = (
            (data[column] < lower) |
            (data[column] > upper)
    )

    print(f"\n{column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower bound:", lower)
    print("Upper bound:", upper)
    print("Number of IQR outliers:", data[f"{column}_IsOutlier"].sum())

    return data


# =========================
# 3. Add IQR outlier flags
# =========================

numeric_fields = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

for field in numeric_fields:
    df = add_iqr_outlier_flags(df, field)

# =========================
# 4. Add business rule flags
# =========================

# Invalid prices
if "ClosePrice" in df.columns:
    df["ClosePrice_Invalid"] = (
            df["ClosePrice"].isna() |
            (df["ClosePrice"] <= 0)
    )

# Invalid living area
if "LivingArea" in df.columns:
    df["LivingArea_Invalid"] = (
            df["LivingArea"].isna() |
            (df["LivingArea"] <= 0)
    )

# Invalid days on market
if "DaysOnMarket" in df.columns:
    df["DaysOnMarket_Invalid"] = (
            df["DaysOnMarket"].isna() |
            (df["DaysOnMarket"] < 0)
    )

# =========================
# 5. Add percentile guide columns
# =========================

# These are not used to delete records.
# They help guide business decisions.

for field in numeric_fields:
    if field in df.columns:
        p01 = df[field].quantile(0.01)
        p99 = df[field].quantile(0.99)

        df[f"{field}_P01"] = p01
        df[f"{field}_P99"] = p99

        df[f"{field}_Outside_1_99_Percentile"] = (
                (df[field] < p01) |
                (df[field] > p99)
        )

# =========================
# 6. Create final combined outlier flag
# =========================

flag_columns = [
    col for col in df.columns
    if col.endswith("_IsOutlier") or col.endswith("_Invalid")
]

df["AnyOutlierOrInvalid"] = df[flag_columns].any(axis=1)

# =========================
# 7. Create clean filtered dataset
# =========================

clean_df = df[df["AnyOutlierOrInvalid"] == False].copy()

print("\nFull flagged dataset shape:", df.shape)
print("Clean filtered dataset shape:", clean_df.shape)

# =========================
# 8. Before vs after comparison
# =========================

comparison_rows = []

for field in numeric_fields:
    if field in df.columns:
        before_count = df[field].notna().sum()
        after_count = clean_df[field].notna().sum()

        before_median = df[field].median()
        after_median = clean_df[field].median()

        before_mean = df[field].mean()
        after_mean = clean_df[field].mean()

        comparison_rows.append({
            "Metric": field,
            "Before_Count": before_count,
            "After_Count": after_count,
            "Before_Median": before_median,
            "After_Median": after_median,
            "Before_Mean": before_mean,
            "After_Mean": after_mean,
            "Median_Change": after_median - before_median,
            "Mean_Change": after_mean - before_mean
        })

comparison_table = pd.DataFrame(comparison_rows)

print("\nBefore vs After Comparison:")
print(comparison_table)

# =========================
# 9. Save outputs
# =========================

df.to_csv("week7_full_flagged_dataset.csv", index=False)
clean_df.to_csv("week7_clean_filtered_dataset.csv", index=False)
comparison_table.to_csv("week7_before_after_comparison.csv", index=False)

print("\nFiles saved:")
print("- week7_full_flagged_dataset.csv")
print("- week7_clean_filtered_dataset.csv")
print("- week7_before_after_comparison.csv")

# =========================
# 10. Sample output
# =========================

sample_cols = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "ClosePrice_IsOutlier",
    "LivingArea_IsOutlier",
    "DaysOnMarket_IsOutlier",
    "ClosePrice_Invalid",
    "LivingArea_Invalid",
    "DaysOnMarket_Invalid",
    "AnyOutlierOrInvalid"
]

sample_cols = [col for col in sample_cols if col in df.columns]

print("\nSample flagged records:")
print(df[sample_cols].head(10))