import pandas as pd

#Loadding the CSV files
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

#Adding labels
fake_df["label"] = 0
real_df["label"] = 1

#Combining both datasets into one
combined_df = pd.concat([fake_df, real_df])

#Shuffling the data so it's mixed well
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

#Saved the combined dataset
combined_df.to_csv("combined_news.csv", index=False)

print("Combined dataset saved as 'combined_news.csv'")
