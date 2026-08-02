#include "cuda_smoke.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

__global__ void vectorAddKernel(
    const float* input_a,
    const float* input_b,
    float* output,
    int count)
{
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (index < count)
    {
        output[index] = input_a[index] + input_b[index];
    }
}

bool checkCuda(
    cudaError_t status,
    const char* operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }

    std::fprintf(
        stderr,
        "CUDA_ERROR operation=%s message=%s\n",
        operation,
        cudaGetErrorString(status));

    return false;
}

}  // namespace

bool runCudaSmokeTest(CudaSmokeResult& result)
{
    result = CudaSmokeResult{};

    if (!checkCuda(
            cudaGetDeviceCount(&result.device_count),
            "cudaGetDeviceCount"))
    {
        return false;
    }

    if (result.device_count <= 0)
    {
        std::fprintf(stderr, "CUDA_ERROR no CUDA device found\n");
        return false;
    }

    cudaDeviceProp device_property{};

    if (!checkCuda(
            cudaGetDeviceProperties(&device_property, 0),
            "cudaGetDeviceProperties"))
    {
        return false;
    }

    std::strncpy(
        result.device_name,
        device_property.name,
        sizeof(result.device_name) - 1);

    result.device_name[sizeof(result.device_name) - 1] = '\0';
    result.compute_major = device_property.major;
    result.compute_minor = device_property.minor;
    result.global_memory_bytes = device_property.totalGlobalMem;

    constexpr int element_count = 256;
    constexpr std::size_t byte_count =
        element_count * sizeof(float);

    std::vector<float> host_a(element_count);
    std::vector<float> host_b(element_count);
    std::vector<float> host_output(element_count, 0.0F);

    for (int index = 0; index < element_count; ++index)
    {
        host_a[index] = static_cast<float>(index);
        host_b[index] = static_cast<float>(index * 2);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;

    bool success = true;

    success &=
        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_a),
                byte_count),
            "cudaMalloc(device_a)");

    success &=
        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_b),
                byte_count),
            "cudaMalloc(device_b)");

    success &=
        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_output),
                byte_count),
            "cudaMalloc(device_output)");

    if (success)
    {
        success &=
            checkCuda(
                cudaMemcpy(
                    device_a,
                    host_a.data(),
                    byte_count,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(host_a)");

        success &=
            checkCuda(
                cudaMemcpy(
                    device_b,
                    host_b.data(),
                    byte_count,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(host_b)");
    }

    if (success)
    {
        constexpr int threads_per_block = 128;

        const int block_count =
            (element_count + threads_per_block - 1) /
            threads_per_block;

        vectorAddKernel<<<block_count, threads_per_block>>>(
            device_a,
            device_b,
            device_output,
            element_count);

        success &=
            checkCuda(
                cudaGetLastError(),
                "vectorAddKernel launch");

        success &=
            checkCuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize");

        success &=
            checkCuda(
                cudaMemcpy(
                    host_output.data(),
                    device_output,
                    byte_count,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(host_output)");
    }

    if (success)
    {
        for (int index = 0; index < element_count; ++index)
        {
            const float expected =
                host_a[index] + host_b[index];

            result.max_abs_error = std::max(
                result.max_abs_error,
                std::abs(host_output[index] - expected));
        }

        success = result.max_abs_error <= 1.0e-6F;
    }

    if (device_output != nullptr)
    {
        cudaFree(device_output);
    }

    if (device_b != nullptr)
    {
        cudaFree(device_b);
    }

    if (device_a != nullptr)
    {
        cudaFree(device_a);
    }

    result.success = success;
    return success;
}
