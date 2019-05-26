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


filetime=sys.argv[1]

optimizatino_flag=[" ",
					"#define MAIN_BLOCK_SIZE 1000",
					"#define INIT_INLINE 1",
					"#define FASTNOTCH 	1", 
					"#define QRSFILT_OPT 1 \n#define BLOCKING_SIZE_QRSFILT 5",
					"#define BDAC_OPT 1",
					"#define AVX_OPT 1",
					"#define NOISECHK_OPT 1"]
				

#define INIT_INLINE 1
#define MAIN_BLOCK_SIZE 1000
#define FASTNOTCH 	1
#define QRSFILT_OPT 1
#define BLOCKING_SIZE_QRSFILT 5 
#define BDAC_OPT 1
#define AVX_OPT 1
#define NOISECHK_OPT 1

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    # print (proc_stdout)

def Average(lst): 
    return sum(lst) / len(lst) 

#incremental tests

dir_results={}
for filename in glob.glob(os.path.join("./", 'step?_%s')%(filetime)):
	with open(filename, "r") as file:
		print("filename :%s"%(filename))
		step_index=filename.split("/")[1][4:5]
		result = {}
		for line in file:
			if line.lstrip():
				name, count=line.split(":")
				count=count.lstrip()
				if name in result.keys():
					result[name].append(float(count))
				else:
					result[name]=[float(count)]

		dir_results[step_index] = result

# print("dir_results:%s"%(dir_results))

metrics_list=["performance","performance (w/ comp)","float total only math","total runtime"]
plot_titles=["Performance in flops per cycle", "Performance with Compare Operations",
			"Total Float Operations", "Total Runtime in cycles"]

for plot_index, metrics in enumerate(metrics_list):
	mean_metrics={}
	stddev_metrics ={}
	for key, value in dir_results.items():
		mean_metrics[key] = Average(value[metrics])
		stddev_metrics[key]=":.2f".format(statistics.stdev(value[metrics]))
		# save to csv
		perf_dataframe=pd.DataFrame.from_dict(value)
		# print(perf_dataframe)
		perf_dataframe.to_csv(('step%s_%s.csv')%(key, filetime), index=False)

optimizatino_flag=[" ",
					"#define MAIN_BLOCK_SIZE 1000",
					"#define INIT_INLINE 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define FASTNOTCH 	1 \n#define MAIN_BLOCK_SIZE 1", 
					"#define QRSFILT_OPT 1 \n#define BLOCKING_SIZE_QRSFILT 5 \n#define MAIN_BLOCK_SIZE 1",
					"#define BDAC_OPT 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define AVX_OPT 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define NOISECHK_OPT 1 \n#define MAIN_BLOCK_SIZE 1"]


dir_results={}
for filename in glob.glob(os.path.join("./", 'only_step*_%s')%(filetime)):
	with open(filename, "r") as file:
		#file name is different only_step1
		step_index=filename.split("/")[1][9]
		result = {}
		for line in file:
			if line.lstrip():
				name, count=line.split(":")
				count=count.lstrip()
				if name in result.keys():
					result[name].append(float(count))
				else:
					result[name]=[float(count)]

		dir_results[step_index] = result


for plot_index, metrics in enumerate(metrics_list):
	mean_metrics={}
	stddev_metrics ={}

	for key, value in dir_results.items():
		mean_metrics[key] = Average(value[metrics])
		stddev_metrics[key]="{:.2f}".format(statistics.stdev(value[metrics]))
		# save to csv
		perf_dataframe=pd.DataFrame.from_dict(value)
		# print(perf_dataframe)
		perf_dataframe.to_csv(('only_step%s_%s.csv')%(key, filetime), index=False)


	

	




