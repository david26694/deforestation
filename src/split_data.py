# %%
"""
Splits data in a stratified manner for cross-validation
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# %%
data = Path("data") / "train.csv"
test_data = Path("data") / "test.csv"
split_data_path = Path("split_data")

# %%
df = pd.read_csv(data)
df_test = pd.read_csv(test_data)

# %%
full_df = pd.concat([df.assign(train="train"), df_test.assign(train="test")], axis=0)

# %% Add fold index in df
for index, (x, y) in enumerate(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, df["label"])
):
    df.loc[y, "fold"] = int(index)

# %%
df["fold"] = df["fold"].astype(int)

df.groupby("label").size()
# %%
df.groupby(["fold", "label"]).size()
# %%
df.loc[:, ["example_path", "fold"]].to_csv(
    split_data_path / "train_folds.csv", index=False
)

# %%
