import pandas as pd

# Read both CSVs
df1 = pd.read_csv("Striking.csv")
df2 = pd.read_csv("Striking1.csv")

# Append (stack rows)
combined = pd.concat([df1, df2], ignore_index=True)

# Save to the original csv
combined.to_csv("Striking.csv", index=False)