# %%

"""
EDA of the tabular data

We see that tabular data matters, class 1 only appears after 2010

Different islands have different distributions of classes, latitude and longitude are super important
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

data_path = Path("data")

df_train = pd.read_csv(data_path / "train.csv")
df_test = pd.read_csv(data_path / "test.csv")
# %%
full_df = pd.concat(
    [
        df_test.assign(is_train=False),
        df_train.assign(is_train=True),
    ]
).fillna(-1)
sns.scatterplot(data=df_train, x="longitude", y="latitude", hue="label", size=1)

# %%
sns.pairplot(df_train, hue="label")

# %%

labels = {
    0: "Plantation (0)",
    1: "Grassland (1)",
    2: "Smallholder Agriculture (2)",
}
df_train["Label"] = df_train["label"].map(labels)

# %%
plt.figure(figsize=(10, 10))
sns.displot(df_train, x="year", hue="Label", common_norm=False, kind="kde", fill=True)
plt.title("Grasslands are captured from 2010 onwards")
plt.xlabel("Year")
plt.ylabel("Density")

# %%
fig = px.scatter_mapbox(
    df_train,
    lat="latitude",
    lon="longitude",
    color="Label",
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=3,
    mapbox_style="carto-positron",
)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

# %%
