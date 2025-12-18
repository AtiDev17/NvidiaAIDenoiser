#pragma once

#include "CudaCheck.h"
#include <vector>

class CudaBuffer
{
public:
    CudaBuffer() : m_ptr(nullptr), m_size(0) {}

    ~CudaBuffer()
    {
        free();
    }

    // Disable copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Enable move
    CudaBuffer(CudaBuffer&& other) noexcept : m_ptr(other.m_ptr), m_size(other.m_size)
    {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            free();
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    void alloc(size_t size)
    {
        free();
        m_size = size;
        if (m_size > 0)
        {
            CU_CHECK(cudaMalloc(&m_ptr, m_size));
        }
    }

    void free()
    {
        if (m_ptr)
        {
            CU_CHECK(cudaFree(m_ptr));
            m_ptr = nullptr;
        }
        m_size = 0;
    }

    void copyToDevice(const void* host_data, size_t size)
    {
        if (size > m_size)
        {
            alloc(size);
        }
        CU_CHECK(cudaMemcpy(m_ptr, host_data, size, cudaMemcpyHostToDevice));
    }

    void copyFromDevice(void* host_data, size_t size) const
    {
        CU_CHECK(cudaMemcpy(host_data, m_ptr, size, cudaMemcpyDeviceToHost));
    }

    CUdeviceptr d_ptr() const { return (CUdeviceptr)m_ptr; }
    void* ptr() const { return m_ptr; }
    size_t size() const { return m_size; }

private:
    void* m_ptr;
    size_t m_size;
};
