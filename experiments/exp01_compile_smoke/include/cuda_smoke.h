#pragma once

#include <cstddef>

struct CudaSmokeResult
{
    bool success = false;
    int device_count = 0;
    char device_name[256] = {};
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t global_memory_bytes = 0;
    float max_abs_error = 0.0F;
};

bool runCudaSmokeTest(CudaSmokeResult& result);
