#include "driver_types.h"
#include "vector_types.h"

#include <assert.h>
#include <dlfcn.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifdef HAVE_BOOST
#include <boost/core/demangle.hpp>
#endif

extern "C" {

FILE *__cudart_trace_output_stream = nullptr;
std::map<const void *, std::string> __cudart_trace_map_host_fun_to_name;

void __attribute__((constructor)) init_cudart_trace() {
  char *output_file = getenv("CUDART_TRACE_OUTPUT_FILE");
  if (output_file) {
    __cudart_trace_output_stream = fopen(output_file, "w");
    assert(__cudart_trace_output_stream);
  } else {
    __cudart_trace_output_stream = stderr;
  }
}

// memory

typedef cudaError_t (*cudaMalloc_type)(void **devPtr, size_t size);

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  cudaMalloc_type orig;
  orig = (cudaMalloc_type)dlsym(RTLD_NEXT, "cudaMalloc");
  auto before = *devPtr;
  auto ret = orig(devPtr, size);
  auto after = *devPtr;
  fprintf(__cudart_trace_output_stream,
          "> cudaMalloc(devPtr=%p(%p -> %p), size=%zu) = %d\n", devPtr, before,
          after, size, ret);
  return ret;
}

typedef cudaError_t (*cudaMemcpy_type)(void *dst, const void *src, size_t count,
                                       enum cudaMemcpyKind kind);

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
  cudaMemcpy_type orig;
  orig = (cudaMemcpy_type)dlsym(RTLD_NEXT, "cudaMemcpy");
  auto ret = orig(dst, src, count, kind);
  fprintf(__cudart_trace_output_stream,
          "> cudaMemcpy(dst=%p, src=%p, count=%zu, kind=%d) = %d\n", dst, src,
          count, kind, ret);
  return ret;
}

// launch kernel

typedef cudaError_t (*cudaLaunchKernel_type)(const char *hostFun, dim3 gridDim,
                                             dim3 blockDim, const void **args,
                                             size_t sharedMem,
                                             cudaStream_t stream);

cudaError_t cudaLaunchKernel(const char *hostFun, dim3 gridDim, dim3 blockDim,
                             const void **args, size_t sharedMem,
                             cudaStream_t stream) {
  cudaLaunchKernel_type orig;
  orig = (cudaLaunchKernel_type)dlsym(RTLD_NEXT, "cudaLaunchKernel");
  auto ret = orig(hostFun, gridDim, blockDim, args, sharedMem, stream);
  fprintf(
      __cudart_trace_output_stream,
      "> cudaLaunchKernel(hostFun=%p(%s), gridDim={%d, %d, %d}, blockDim={%d, "
      "%d, %d}, args=%p, sharedMem=%zu, stream=%p) = %d\n",
      hostFun, __cudart_trace_map_host_fun_to_name[hostFun].c_str(), gridDim.x,
      gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, args, sharedMem,
      stream, ret);
  return ret;
}

typedef unsigned (*__cudaPushCallConfiguration_type)(
    dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream);

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream) {
  __cudaPushCallConfiguration_type orig;
  orig = (__cudaPushCallConfiguration_type)dlsym(RTLD_NEXT,
                                                 "__cudaPushCallConfiguration");
  auto ret = orig(gridDim, blockDim, sharedMem, stream);
  fprintf(__cudart_trace_output_stream,
          "> __cudaPushCallConfiguration(gridDim={%d, %d, %d}, blockDim={%d, "
          "%d, %d}, sharedMem=%zu, stream=%p) = %d\n",
          gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
          sharedMem, stream, ret);
  return ret;
}

typedef unsigned (*__cudaPopCallConfiguration_type)(dim3 *gridDim,
                                                    dim3 *blockDim,
                                                    size_t *sharedMem,
                                                    void *stream);

unsigned __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                    size_t *sharedMem, void *stream) {
  __cudaPopCallConfiguration_type orig;
  orig = (__cudaPopCallConfiguration_type)dlsym(RTLD_NEXT,
                                                "__cudaPopCallConfiguration");
  auto ret = orig(gridDim, blockDim, sharedMem, stream);
  auto gridDimAfter = *gridDim;
  auto blockDimAfter = *blockDim;
  size_t sharedMemAfter = *sharedMem;
  fprintf(__cudart_trace_output_stream,
          "> __cudaPopCallConfiguration(gridDim=%p(->{%d, %d, %d}), "
          "blockDim=%p(->{%d, %d, %d}), "
          "sharedMem=%p(->%zu), stream=%p) = %d\n",
          gridDim, gridDimAfter.x, gridDimAfter.y, gridDimAfter.z, blockDim,
          blockDimAfter.x, blockDimAfter.y, blockDimAfter.z, sharedMem,
          sharedMemAfter, stream, ret);
  return ret;
}

// registration

typedef void **(*__cudaRegisterFatBinary_type)(void **fatCubin);

void **__cudaRegisterFatBinary(void **fatCubin) {
  __cudaRegisterFatBinary_type orig;
  orig =
      (__cudaRegisterFatBinary_type)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
  auto ret = orig(fatCubin);
  fprintf(__cudart_trace_output_stream,
          "> __cudaRegisterFatBinary(fatCubin=%p) = %p\n", fatCubin, ret);
  return ret;
}

typedef void (*__cudaRegisterFatBinaryEnd_type)(void **fatCubinHandle);

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  __cudaRegisterFatBinaryEnd_type orig;
  orig = (__cudaRegisterFatBinaryEnd_type)dlsym(RTLD_NEXT,
                                                "__cudaRegisterFatBinaryEnd");
  fprintf(__cudart_trace_output_stream,
          "> __cudaRegisterFatBinaryEnd(fatCubinHandle=%p)\n", fatCubinHandle);
  return orig(fatCubinHandle);
}

typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

void __cudaInitModule(void **fatCubinHandle) {
  __cudaInitModule_type orig;
  orig = (__cudaInitModule_type)dlsym(RTLD_NEXT, "__cudaInitModule");
  fprintf(__cudart_trace_output_stream,
          "> __cudaInitModule(fatCubinHandle=%p)\n", fatCubinHandle);
  return orig(fatCubinHandle);
}

typedef void (*__cudaUnregisterFatBinary_type)(void **fatCubinHandle);

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  __cudaUnregisterFatBinary_type orig;
  orig = (__cudaUnregisterFatBinary_type)dlsym(RTLD_NEXT,
                                               "__cudaUnregisterFatBinary");
  fprintf(__cudart_trace_output_stream,
          "> __cudaUnregisterFatBinary(fatCubinHandle=%p)\n", fatCubinHandle);
  return orig(fatCubinHandle);
}

typedef void (*__cudaRegisterFunction_type)(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  __cudaRegisterFunction_type orig;
  orig =
      (__cudaRegisterFunction_type)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
#ifdef HAVE_BOOST
  std::string demangled = boost::core::demangle(deviceName);
  const char *printDeviceName = demangled.c_str();
#else
  const char *printDeviceName = deviceName;
#endif
  __cudart_trace_map_host_fun_to_name[hostFun] = printDeviceName;
  fprintf(
      __cudart_trace_output_stream,
      "> __cudaRegisterFunction(fatCubinHandle=%p, hostFun=%p, deviceFun=%p, "
      "deviceName=%s, thread_limit=%d, tid=%p, bid=%p, bDim=%p, gDim=%p, "
      "wSize=%p)\n",
      fatCubinHandle, hostFun, deviceFun, printDeviceName, thread_limit, tid,
      bid, bDim, gDim, wSize);
  return orig(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);
}

// library calls

typedef cudaError_t (*cudaDeviceSynchronize_type)();

cudaError_t cudaDeviceSynchronize() {
  cudaDeviceSynchronize_type orig;
  orig = (cudaDeviceSynchronize_type)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
  auto ret = orig();
  fprintf(__cudart_trace_output_stream, "> cudaDeviceSynchronize() = %d\n",
          ret);
  return ret;
}

typedef cudaError_t (*cudaDeviceReset_type)();

cudaError_t cudaDeviceReset() {
  cudaDeviceReset_type orig;
  orig = (cudaDeviceReset_type)dlsym(RTLD_NEXT, "cudaDeviceReset");
  auto ret = orig();
  fprintf(__cudart_trace_output_stream, "> cudaDeviceReset() = %d\n", ret);
  return ret;
}
}