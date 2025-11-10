import pandas as pd
import re

# ================================
# 1) Load data
# ================================

df = pd.read_csv("features_all.csv")

# Keep the top 30% activity for each gesture
top_percent = 0.3

def keep_top30(group):
    n_keep = int(len(group) * top_percent)
    return group.sort_values("activity", ascending=False).iloc[:n_keep]

# Keep only the top 30%
results = df.groupby("src", group_keys=False).apply(keep_top30)

"""
# Manual way of doing stuff

# Remove the stem (.csv) and the numerical digit
labels = {
    "Stationary.csv" : "Stationary",
    "Stationary1.csv" : "Stationary",
    "Stationary2.csv" : "Stationary",
    "Stationary3.csv" : "Stationary",
    "Waving.csv" : "Waving",
    "Waving1.csv" : "Waving",
    "Waving2.csv" : "Waving",
    "Waving3.csv" : "Waving",
    "Lift Upward.csv" : "Lift Upward",
    "Lift Upward1.csv" : "Lift Upward",
    "Lift Upward2.csv" : "Lift Upward",
    "Lift Upward3.csv" : "Lift Upward",
    "Punch Forward.csv" : "Punch Forward",
    "Punch Forward1.csv" : "Punch Forward",
    "Punch Forward2.csv" : "Punch Forward",
    "Punch Forward3.csv" : "Punch Forward",
    "Striking.csv" : "Striking",
    "Striking1.csv" : "Striking",
    "Striking2.csv" : "Striking",
    "Striking3.csv" : "Striking"
}

results["labels"] = results["src"].map(labels) 
"""


# More modern way (I learned this from chatgpt)
results["motion"] = (
    results["src"]
    .str.replace(r'\d+', '', regex=True)  # remove digits
    .str.replace('.csv', '', regex=False) # remove file extension
    .str.strip()                          # clean up spaces
)

results = results.drop(columns=["src"])

# Save to the original csv
results.to_csv("motion_dataset.csv", index=False)
