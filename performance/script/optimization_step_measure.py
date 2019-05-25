import os
import subprocess
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import statistics


NUM_RUN=2

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


def autolabel(rects,compute_delta):
    """
    Attach a text label above each bar displaying its height
    """
    for counter, rect in enumerate(rects):
        text = compute_delta[counter]
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % (text),color='green',
                ha='center', va='bottom')

target_dir = "../../hamilton_inline/"
performance_header_path = target_dir+"performance.h"
ourtest_path = target_dir+"ourtest"

now = datetime.datetime.now()

nowtime=now.strftime("%Y-%m-%d_%H:%M:%S")

#incremental tests

for step, opt in enumerate(optimizatino_flag):
	outputfile="step"+str(step)+"_"+nowtime
	if step != 0:
		with open("../../hamilton_inline/performance.h","w") as config_file:
			config_file.write("#define OPERATION_COUNTER \n#define RUNTIME_MEASURE\n#define PRINT \n")
			for i in optimizatino_flag[:step+1]:
				config_file.write("%s\n"%(i))
		# recompile the program
		subprocess_cmd("cd ../../hamilton_inline/; make clean all")
		#flush disk cache
		subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

	for num in range(NUM_RUN):
		if step == 0:
			# run baseline 
			subprocess_cmd("../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
		else:
			subprocess_cmd("../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")


dir_results={}
for filename in glob.glob(os.path.join("./", 'step?_%s')%(nowtime)):
	with open(filename, "r") as file:
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
	# print("mean_metrics:%s"%(mean_metrics))

	# print("sorted mean_metrics: %s"%(sorted(mean_metrics.items(), key=lambda s: s[0])))
	sorted_mean_metrics=sorted(mean_metrics.items(), key=lambda s: s[0])
	sorted_stddev_metrics=sorted(stddev_metrics.items(),key=lambda s: s[0])
	# print("sorted_mean_metrics:%s and standard deviation %s"%(sorted_mean_metrics, sorted_stddev_metrics))

	step_names=[]
	y_values=[]
	delta=[]
	for v in sorted_mean_metrics:
		name= "step"+v[0]
		step_names.append(name)
		y_values.append(v[1])
		d="{:.2f}".format((v[1]-y_values[0])/y_values[0])
		delta.append(d)

	errors = []
	for s in sorted_stddev_metrics:	
		errors.append(v[1])

	fig, ax = plt.subplots()
	x = np.arange(len(step_names))
	p1 = plt.bar(x, y_values,0.35, yerr=errors)
	autolabel(p1,delta)
	ind = np.arange(len(step_names))
	ax.set_xticks(ind)
	# block_size_small = [ "{:03.1f}".format(x/1000) for x in block_size]
	ax.set_xticklabels(step_names, {'fontsize': 10})
	# ax.set_ylabel('Flops/cycle')
	ax.set_xlabel('Optimization Steps')
	ax.set_title('%s Comparison'%(plot_titles[plot_index]))
	ax.plot(x, y_values)
	savefig_filename=plot_titles[plot_index]+nowtime
	plt.savefig(savefig_filename)
	plt.show(block=False)

#individual test (one optimization applied at any time)
optimizatino_flag=[" ",
					"#define MAIN_BLOCK_SIZE 1000",
					"#define INIT_INLINE 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define FASTNOTCH 	1 \n#define MAIN_BLOCK_SIZE 1", 
					"#define QRSFILT_OPT 1 \n#define BLOCKING_SIZE_QRSFILT 5 \n#define MAIN_BLOCK_SIZE 1",
					"#define BDAC_OPT 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define AVX_OPT 1 \n#define MAIN_BLOCK_SIZE 1",
					"#define NOISECHK_OPT 1 \n#define MAIN_BLOCK_SIZE 1"]
for step, opt in enumerate(optimizatino_flag):
	outputfile="only_step"+str(step)+"_"+nowtime
	if step != 0:
		with open("../../hamilton_inline/performance.h","w") as config_file:
			config_file.write("#define OPERATION_COUNTER \n#define RUNTIME_MEASURE\n#define PRINT \n")
			config_file.write("%s\n"%(opt))
		# recompile the program
		subprocess_cmd("cd ../../hamilton_inline/; make clean all")
		#flush disk cache
		subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

	for num in range(NUM_RUN):
		if step == 0:
			# run baseline 
			subprocess_cmd("../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
		else:
			subprocess_cmd("../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

dir_results={}
for filename in glob.glob(os.path.join("./", 'only_step*_%s')%(nowtime)):
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

	# print("sorted mean_metrics: %s"%(sorted(mean_metrics.items(), key=lambda s: s[0])))
	sorted_mean_metrics=sorted(mean_metrics.items(), key=lambda s: s[0])
	sorted_stddev_metrics=sorted(stddev_metrics.items(),key=lambda s: s[0])

	step_names=[]
	y_values=[]
	delta=[]
	for v in sorted_mean_metrics:
		name= "step"+v[0]
		step_names.append(name)
		y_values.append(v[1])
		d="{:.2f}".format((v[1]-y_values[0])/y_values[0])
		delta.append(d)

	errors = []
	for s in sorted_stddev_metrics:	
		errors.append(v[1])


	fig, ax = plt.subplots()
	x = np.arange(len(step_names))
	p1 = plt.bar(x, y_values,0.35,yerr=errors)
	autolabel(p1,delta)
	ind = np.arange(len(step_names))
	ax.set_xticks(ind)
	# block_size_small = [ "{:03.1f}".format(x/1000) for x in block_size]
	ax.set_xticklabels(step_names, {'fontsize': 10})
	# ax.set_ylabel('Flops/cycle')
	ax.set_xlabel('Individual Optimization Applied')
	ax.set_title('%s Comparison'%(plot_titles[plot_index]))
	ax.plot(x, y_values)
	savefig_filename="individual_"+plot_titles[plot_index]+nowtime
	plt.savefig(savefig_filename)
	plt.show(block=False)

			

