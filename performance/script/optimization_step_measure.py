import os
import subprocess
import datetime

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

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    # print (proc_stdout)

target_dir = "../../hamilton_inline/"
performance_header_path = target_dir+"performance.h"
ourtest_path = target_dir+"ourtest"

now = datetime.datetime.now()

nowtime=now.strftime("%Y-%m-%d %H:%M:%S")

for step, opt in enumerate(optimizatino_flag):
	outputfile="step"+str(step)+"_"+nowtime
	if step != 0:
		with open("../../hamilton_inline/performance.h","w") as config_file:
			config_file.write("#define OPERATION_COUNTER \n#define RUNTIME_MEASURE \n#define RUNTIME_QRSDET \n#define RUNTIME_CLASSIFY \n#define PRINT \n")
			for i in optimizatino_flag[:step+1]:
				config_file.write("%s\n"%(i))
		# recompile the program
		subprocess_cmd("cd ../../hamilton_inline/; make clean all")

	for num in range(NUM_RUN):
		if step == 0:
			# run baseline 
			subprocess_cmd("../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
		else:
			subprocess_cmd("../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))







			




