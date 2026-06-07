import pandas as pd

df = pd.read_csv("week7_clean_filtered_dataset.csv")

# Parse CloseDate
df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")

# Keep Jan 2024 through latest month available
df = df[df["CloseDate"] >= "2024-01-01"]

# Residential filter
# Check your actual PropertyType values first:
print(df["PropertyType"].value_counts(dropna=False))

df = df[df["PropertyType"].str.contains("Residential", case=False, na=False)]

# Save Tableau-ready file
df.to_csv("tableau_residential_clean.csv", index=False)

print("Saved tableau_residential_clean.csv")
print("Rows:", len(df))
print("Date range:", df["CloseDate"].min(), "to", df["CloseDate"].max())