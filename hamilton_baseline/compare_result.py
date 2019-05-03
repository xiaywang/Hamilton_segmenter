import pandas as pd

df_true = pd.read_csv("./True_DetectionTime100.csv")
DT_true = df_true['DetectionTime'].values

df = pd.read_csv("./to_plot/DetectionTime100.csv")
DT = df['DetectionTime'].values

if (DT_true == DT).all():
    print("\nCORRECT! The detected QRS onsets are correct\n")
else:
    print("\nWRONG! The values are wrong!")
