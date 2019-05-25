import os
import sys
import subprocess
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import csv


opt_list = []
perf_dict = {}

if len(sys.argv) != 2:
    sys.exit('Error: call the script like this: python3 plot_qrsfilt_perf.py "filename.txt"')

print("plotting for {}".format(sys.argv[1]))

with open(sys.argv[1], 'r') as f:
    for line in f:
        line_split = line.split()
        if len(line_split) == 0:
            continue
        #print(line_split[0])
        if line_split[0] == 'Running:':
            if line_split[1] == 'Slow':
                for i in range(2,len(line_split)):
                    line_split[1] += line_split[i]
            
            if line_split[1] == 'Blocking':
                for i in range(2,len(line_split)):
                    line_split[1] += line_split[i]

            if line_split[1] == 'Precomp':
                for i in range(2,len(line_split)):
                    line_split[1] += line_split[i]
            #print(line_split[1])
            opt_list.append(line_split[1])
        elif line_split[0] == 'average':

            perf_dict[opt_list[-1]] = float(line_split[1])

#print(opt_list)
print(perf_dict)

# save to csv
perf_dataframe=pd.DataFrame.from_dict([perf_dict])
print(perf_dataframe)
perf_dataframe.to_csv(sys.argv[1][:-4]+'.csv', index=False)




