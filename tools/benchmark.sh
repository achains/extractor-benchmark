#!/bin/bash

cmake -B build -S .. -DIMAGE_DIR="$1" -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_VERBOSE=ON
cmake --build build -j 6

# Disable turboboost
# Intel
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
# Disable hyper threading
echo 0 > /sys/devices/system/cpu/cpuX/online
# Set scaling_governor to ‘performance’
for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
  echo performance > "$i"
done

max_frames="$2"
# Run CAPE benchmark
./build/benchmark/cape_benchmark "$max_frames"
# Run deplex benchmark
./build/benchmark/deplex_benchmark "$max_frames"
# Run PEAC benchmark
./build/benchmark/peac_benchmark "$max_frames"