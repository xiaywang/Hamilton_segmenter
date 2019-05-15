import sys
import colors
result={}
with open(sys.argv[1], "r") as file:
	for line in file:
		if line.lstrip() and line[0] != "S" and line[0] != "D":
			name, count=line.split(":")
			count=count.lstrip()
			if name in result.keys():
				result[name].append(float(count))
				baseline=result[name][1]
				test=result[name][0]
				if test != 0:
					delta = "{0:.0%}".format((test-baseline)/test)
				else:
					delta = "infinity"
				result[name].append(delta)
			else:
				result[name]=[float(count)]
		# else:
		# 	print("exception: %s" %line)

print(colors.title("Comparing to floating point baseline:"))
print(colors.title("For detailed report, please refer to %s"%(sys.argv[1])))
for key, value in result.items():
	if len(value) == 3:
		if value[2][0] == "-":
			print("%s: %s" %(colors.yellow(key), colors.green(value[2])))
		else:
			print("%s: %s " %(colors.yellow(key), colors.red(value[2])))
	else:
		print(colors.red("%s misses data, please refer to report for detail"%(key)))


