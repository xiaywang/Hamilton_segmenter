time ./ourtest &
OUTPUT="$(pidof ourtest)"
echo "${OUTPUT}"
cd ~/Flame/FlameGraph
perf record -F 99 -p $OUTPUT -g -- sleep 20
wait
echo "ourtest is done"
current_path=$PWD
current_dir=${current_path##*/}
foldfile="_out"
flamefile="_perf-kernel.svg"
now=$(date +"%T")
perf script | ./stackcollapse-perf.pl > $now$foldfile
./flamegraph.pl out.perf-folded > $now$flamefile