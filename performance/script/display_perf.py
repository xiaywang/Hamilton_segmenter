import sys
print("file name is %s"%sys.argv[1])
result={}
with open(sys.argv[1], "r") as file:
	for line in file:
		if line.lstrip() and line[0] != "S":
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
			print(" ".join([name, count]))


		else:
			print("exception: %s" %line)

print(result)

for key, value in result.items():
	if value[2][0] == "-":
		print("%s has decreased %s comparing to float baseline\n" %(key, value[2]))

	else:
		print("%s has increased %s comparing to float baseline\n" %(key, value[2]))
