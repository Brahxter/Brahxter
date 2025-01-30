import pandas as pd
from sklearn.model_selection import train_test_split

# Load your SPY data
# Adjust filename as needed
df = pd.read_csv('data/spy_5min_data_2021_2024.csv')

# Split into train/val
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

# Save splits
train_df.to_csv('data/spy_train.csv', index=False)
val_df.to_csv('data/spy_val.csv', index=False)
