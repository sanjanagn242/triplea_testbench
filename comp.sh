lscpu                      # model, sockets, cores, threads, base/max MHz, cache sizes
cat /proc/cpuinfo | grep MHz | head -n 8   # actual current clocks
free -h
numactl --hardware         # NUMA layout (if available)