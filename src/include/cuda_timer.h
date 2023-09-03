// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: cuda timer

#ifndef __CUDA_HGEMM_CUDA_TIMER_H__
#define __CUDA_HGEMM_CUDA_TIMER_H__

#include "common.h"

class CudaTimer {
public:
    CudaTimer() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        HGEMM_CHECK(m_start);
        HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
        HGEMM_CHECK(m_end);
    }

    ~CudaTimer() {
        if (m_start) {
            HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
            m_start = nullptr;
        }

        if (m_end) {
            HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
            m_end = nullptr;
        }
    }

    void start() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_start));
    }

    float end() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_end));
        HGEMM_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
        HGEMM_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));

        return m_elapsed_time;
    }

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};

#endif  // __CUDA_HGEMM_CUDA_TIMER_H__