// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: logging

#ifndef __CUDA_HGEMM_LOGGING_H__
#define __CUDA_HGEMM_LOGGING_H__

#include <cstdio>
#include <cstring>
#include <ctime>

inline char *curr_time() {
    time_t raw_time = time(nullptr);
    struct tm *time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S", time_info)] = '\0';

    return now_time;
}

#define HGEMM_LOG_TAG "HGEMM"
#define HGEMM_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                                                         \
    do {                                                                                                          \
        fprintf(stderr, "[%s %s %s:%d %s] " format "\n", HGEMM_LOG_TAG, curr_time(),                              \
                HGEMM_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__);                                 \
    } while (0)

#endif  // __CUDA_HGEMM_LOGGING_H__
