import os
import subprocess
import datetime
NUM_RUN=1
# cache measurement
def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    # print (proc_stdout)

optimizatino_flag=[" ",
					"#define MAIN_BLOCK_SIZE 1000",
					"#define INIT_INLINE 1",
					"#define FASTNOTCH 	1", 
					"#define QRSFILT_OPT 1 \n#define BLOCKING_SIZE_QRSFILT 5",
					"#define BDAC_OPT 1",
					"#define AVX_OPT 1",
					"#define NOISECHK_OPT 1"]

# blocksizes = [1,2,5,10,20,50,100,200,500,1000,2000,4000,7200]
now = datetime.datetime.now()
nowtime=now.strftime("%Y-%m-%d_%H:%M:%S")
outputfile = "cache_"+nowtime
# for bs in blocksizes:
for step, opt in enumerate(optimizatino_flag):
	if step != 0:
		with open("../../hamilton_inline/performance.h","w") as config_file:
			config_file.write("#define OPERATION_COUNTER \n#define RUNTIME_MEASURE\n#define PRINT \n")
			for i in optimizatino_flag[:step+1]:
				config_file.write("%s\n"%(i))
		# recompile the program
		subprocess_cmd("cd ../../hamilton_inline/; make clean all")
		#flush disk cache
		subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

	# subprocess_cmd("echo Block Size: %s >> %s" %(bs,outputfile))
	subprocess_cmd("echo optimization step: %s >> %s" %(step,outputfile))
	for ct in range(NUM_RUN):
		if step != 0:
			subprocess_cmd("perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e dTLB-loads,dTLB-load-misses,dTLB-prefetch-misses ../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches ../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
			
			subprocess_cmd("perf stat -e cycles,instructions,cache-references,cache-misses,bus-cycles ../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e icache.hit,icache.misses ../../hamilton_inline/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
		else: 
			subprocess_cmd("perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e dTLB-loads,dTLB-load-misses,dTLB-prefetch-misses ../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e LLC-loads,LLC-load-misses,LLC-stores,LLC-prefetches ../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
			
			subprocess_cmd("perf stat -e cycles,instructions,cache-references,cache-misses,bus-cycles ../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")

			subprocess_cmd("perf stat -e icache.hit,icache.misses ../../hamilton_float/ourtest 2>&1 | tee -a %s"%(outputfile))
			#flush disk cache
			subprocess_cmd("sync; echo 1 > /proc/sys/vm/drop_caches")
