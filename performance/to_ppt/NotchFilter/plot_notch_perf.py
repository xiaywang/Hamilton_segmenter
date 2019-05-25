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


fct_list = []
N = []
perf_dict = {}

if len(sys.argv) != 2:
    sys.exit('Error: call the script like this: python3 plot_notch_perf.py "filename.txt"')

print("plotting for {}".format(sys.argv[1]))

with open(sys.argv[1], 'r') as f:
    for line in f:
        line_split = line.split()
        if len(line_split) == 0:
            continue

        if line_split[0] == 'Running:':
            if line_split[1] == 'Slow':
                line_split[1] = line_split[1]+line_split[2]
            fct_list.append(line_split[1])
        elif line_split[0] == 'N:':
            if int(line_split[1][:-1]) not in N:
                N.append(int(line_split[1][:-1]))
            if fct_list[-1] not in perf_dict.keys():
                perf_dict[fct_list[-1]] = [float(line_split[3])]
            else:
                perf_dict[fct_list[-1]].append(float(line_split[3]))


#print(type(perf_dict[fct_list[-1]][0]))
perf_dict['N']=N
#print(perf_dict)
plt.figure()
for fct in fct_list:
    plt.plot(N, perf_dict[fct])
    #print(perf_dict[fct])

#bottom, top = plt.ylim()  # return the current ylim
#plt.ylim(bottom, top)     # set the ylim to bottom, top
#plt.yticks(np.linspace(bottom,top))
plt.legend()
plt.xlabel("Input size N")
plt.ylabel("Performance (flops/cycle)")
plt.title(sys.argv[1][:-4])
plt.savefig(sys.argv[1][:-4]+'.png')
#plt.show()

# save to csv
perf_dataframe=pd.DataFrame.from_dict(perf_dict)
print(perf_dataframe)
perf_dataframe.to_csv(sys.argv[1][:-4]+'.csv', index=False)




