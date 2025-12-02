import pandas as pd

df = pd.read_csv("mapping.csv")


def clean_to_int(x):
    if pd.isna(x):
        return None
    try:
        # works for "67391582.0", 67391582.0, "67391582"
        return int(float(x))
    except:  # noqa: E722
        return None


df["population_country"] = df["population_country"].apply(clean_to_int)

df.to_csv("mapping.csv", index=False)
print("population_country column cleaned and cast to int.")
