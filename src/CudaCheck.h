#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include <cstdlib>

// Helper for error handling
inline void cudaCheckReportError(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) 
                  << "(" << cudaGetErrorName(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CU_CHECK(val) cudaCheckReportError((val), #val, __FILE__, __LINE__)

inline void optixCheckReportError(OptixResult result, char const *const func, const char *const file, int const line)
{
    if (result != OPTIX_SUCCESS)
    {
        std::cerr << "OptiX error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) 
                  << "(" << optixGetErrorName(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define OPTIX_CHECK(val) optixCheckReportError((val), #val, __FILE__, __LINE__)

