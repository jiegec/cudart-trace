# cudart-trace

Trace calls to cudart.

## Building

Use CMake:

```shell
mkdir build
cd build
cmake ..
make
```

## Usage

Run CUDA applications with LD_PRELOAD:

```shell
LD_PRELOAD=./build/libcudart-trace.so python3 -c "import torch"
```