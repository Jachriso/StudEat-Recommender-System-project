import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/processed/{split}.csv", usecols=["user_id", "recipe_id", "rating"])
    neg = (df["rating"] <= 2).sum()
    print(f"{split:5s} {len(df):7d} notes | <=2 : {neg:6d} ({neg / len(df):.1%})")
    print(df["rating"].value_counts().sort_index().to_dict())