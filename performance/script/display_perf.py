import sys
import colors
import matplotlib.pyplot as plt
import numpy as np
result={}
if sys.argv[1] == "-p":
	with open(sys.argv[2], "r") as file:
		for line in file:
			if line.lstrip() and line[0] != "S" and line[0] != "D":
				name, count=line.split(":")
				count=count.lstrip()
				if name in result.keys():
					result[name].append(float(count))
					baseline=result[name][1]
					test=result[name][0]
					if test != 0:
						delta = "{0:.0%}".format((test-baseline)/baseline)
					else:
						delta = "infinity"
					result[name].append(delta)
				else:
					result[name]=[float(count)]
			# else:
			# 	print("exception: %s" %line)
	with open(sys.argv[2], "a") as file:
		print(colors.title("Comparing to floating point baseline:"))
		file.write("Comparing to floating point baseline: \n")
		print(colors.title("For detailed report, please refer to %s"%(sys.argv[2])))
		file.write("For detailed report, please refer to %s \n"%(sys.argv[2]))

		for key, value in result.items():
			if len(value) == 3:
				if value[2][0] == "-":
					print("%s: %s" %(colors.yellow(key), colors.green(value[2])))
				else:
					print("%s: %s " %(colors.yellow(key), colors.red(value[2])))
				file.write("%s: %s\n" %(key,value[2]))
			else:
				print(colors.red("%s misses data, please refer to report for detail"%(key)))

if sys.argv[1] == "-i":
	block_size=[]
	performance=[]
	with open(sys.argv[2], "r") as file:
		for line in file:
			if line.lstrip():
				name, count=line.split(":")
				count = count.lstrip()
				if name == "Block Size":
					block_size.append(int(count))
				if name == "performance":
					performance.append(float(count))

	print("block_size:%s and performance:%s"%(block_size, performance))

	fig, ax = plt.subplots()
	x = np.arange(len(block_size))
	plt.bar(x, performance)
	ind = np.arange(len(block_size))
	ax.set_xticks(ind)
	block_size_small = [ "{:03.1f}".format(x/1000) for x in block_size]
	ax.set_xticklabels(block_size_small, {'fontsize': 7})
	ax.set_ylabel('Flops/cycle')
	ax.set_xlabel('Input Block Size(4KB)')
	ax.set_title('Performance per Block Size')
	ax.plot(x, performance)
	plt.savefig(sys.argv[2])
	plt.show(block=True)
	# ax2=ax.twinx()
	# ax2.plot(ax.get_xticks(),
 #         linestyle='-',
 #         marker='o', linewidth=2.0)
	# plt.show(block=True)






