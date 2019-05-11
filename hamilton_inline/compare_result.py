import numpy as np
import pandas as pd

df_true = pd.read_csv("./True_DetectionTime100.csv")
DT_true = df_true['DetectionTime'].values

df = pd.read_csv("./to_plot/DetectionTime100.csv")
DT = df['DetectionTime'].values

if (DT_true == DT).all():
    print("\nCORRECT! The detected QRS onsets are correct\n")
else:
    print("\nWRONG! The values are wrong! For details see what follows:\n")
    print("true label\t-\tpredicted label\t=\tdifference\n")
    diff=np.zeros(np.size(DT), dtype=int)
    for i in range(np.size(DT)):
        diff[i] = DT_true[i]-DT[i]
        print("{}\t-\t{}\t=\t{}\n".format(DT_true[i], DT[i], diff[i]))
    print("There are in total {} different values\n".format(np.size(np.nonzero(diff))))
