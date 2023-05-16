# cudart-trace

Trace calls to cudart. You can use this tool to locate kernels from dynamic libraries (e.g. from torch).

Example output:

```
> __cudaRegisterFunction(fatCubinHandle=0x5654afbbc940, hostFun=0x7fa048ea6c90, deviceFun=0x7fa04ab4ef38, deviceName=void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 3>), thread_limit=-1, tid=(nil), bid=(nil), bDim=(nil), gDim=(nil), wSize=(nil))
> cudaMalloc(devPtr=0x7ffc3f8ad6e8(0x770000007c -> 0x7f9f1b600000), size=2097152) = 0
> __cudaPushCallConfiguration(gridDim={1, 1, 1}, blockDim={128, 1, 1}, sharedMem=0, stream=(nil)) = 0
> __cudaPopCallConfiguration(gridDim=0x7ffc3f8ad294(->{1, 1, 1}), blockDim=0x7ffc3f8ad2a0(->{128, 1, 1}), sharedMem=0x7ffc3f8ad260(->0), stream=0x7ffc3f8ad270) = 0
> cudaLaunchKernel(hostFun=0x7fa048ea6c90(void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 3>)), gridDim={1, 1, 1}, blockDim={128, 1, 1}, args=0x7ffc3f8adca0, sharedMem=0, stream=(nil)) = 0
```

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

nvcc should be called with `--cudart shared` for LD_PRELOAD to work.

Customization via environment variables:

- `CUDART_TRACE_OUTPUT_FILE=log`: Print trace to file instead of stderr
