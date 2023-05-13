#!/bin/sh
LC_ALL=C.UTF-8
objdump -T /usr/local/cuda/lib64/libcudart.so | grep libcudart | awk '{print $7}' | sort | sed '1d' | sed '$d' > cudart-exports.txt