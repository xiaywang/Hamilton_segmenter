import sys
import colors
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
   # do your stuff

path = '../output/'

dir_results= {}
for filename in glob.glob(os.path.join(path, 'ALL*')):
	with open(filename, "r") as file:
		opt_code=filename.split("/")[-1][3:10]
		print("opt_code:%s \n" %(opt_code))
		result = {}
		for line in file:
			if line[:5] == "Compa":
				break 
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
		dir_results[opt_code] = result

print(dir_results)

metric_name = "performance"
#metric_name = "performance (w/ comp)"
N = len(dir_results)
opt_val = []
base_val = []
delta = []
compute_delta= []
for key, value in dir_results.items():
	opt_val.append(value[metric_name][0])

	base_val.append(value[metric_name][1])
	delta.append(value[metric_name][2])
	# dt="{0:.0%}".format((opt_val[-1]-base_val[-1])/base_val[-1])
	dt=(opt_val[-1]-base_val[-1])/base_val[-1]
	dt="{0:.1f}".format(dt)
	dt=str(dt)+"x"
	compute_delta.append(dt)

print("opt_val:%s\n" %(opt_val))
print("base_val:%s\n" %(base_val))
print("compute_delta: %s\n" %(compute_delta))

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, base_val, width)

p2 = ax.bar(ind + width, opt_val, width)

ax.set_title('Performance (Flops/Cycle) Comparison')
#ax.set_title('Performance with Compare Operations(Flops/Cycle) Comparison')
ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ticklabels = set(dir_results.keys())

label_flags=["QRSFILT_OPT", "BLOCKING_SIZE_QRSFILT", "FASTNOTCH", "INIT_INLINE", "BDAC_OPT", "AVX_OPT", "NOISECHECK_OPT"]
label_flags_short=["Qrsfilt", "Blocking", "Fastnotch", "Inline", "Bdac", "AVX", "Noisecheck"]
decode_label =[]
for code in ticklabels:
	label=""
	for index, pos in enumerate(code):
		if pos == "1":
			label+=label_flags_short[index][0]+"+"
	label=label[:-1]
	print("label:%s\n" %(label))
	decode_label.append(label)

# ax.set_xticklabels(ticklabels)
ax.set_xticklabels(decode_label,{'fontsize': 7})

ax.legend((p1[0], p2[0]), ('Baseline', 'Optimization'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for counter, rect in enumerate(rects):
        text = compute_delta[counter]
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % (text),color='green',
                ha='center', va='bottom')

autolabel(p2)

ax.autoscale_view()

# ax.set_ylabel('Flops/cycle')
ax.set_xlabel('Optimization')

# save_file="ALL"+metric_name

# plt.savefig("ALL_performance_with_compare")


plt.show()

