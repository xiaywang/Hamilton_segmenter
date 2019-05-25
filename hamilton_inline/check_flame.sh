#check if there is ourtest is running 
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`
PROCESS_NUM=$(ps -ef | grep "ourtest" | grep -v "grep" | wc -l)
if [ $PROCESS_NUM -ne 0 ]
	then echo "${red}Ourtest is already running${reset}"
		exit
fi
if [ $# -lt 1 ]; then
    echo "${red}Please specify an output name${reset}"
    exit 1
fi
echo "#define MAIN_BLOCK_SIZE 1000 \n#define FLAME\n" > performance.h
make clean all
time ./ourtest &
OUTPUT="$(pidof ourtest)"
echo "${OUTPUT}"
cd ~/Flame/FlameGraph
perf record -F 99 -p $OUTPUT -g -- sleep 40
current_path=$PWD
current_dir=${current_path##*/}
foldfile="_out"
flamefile="_perf-kernel.svg"
now=$(date +"%T")
name=$1
dash="_"
perf script | ./stackcollapse-perf.pl > $name$dash$now$foldfile
./flamegraph.pl $name$dash$now$foldfile > $name$dash$now$flamefile
wait
echo "ourtest is done"