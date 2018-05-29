#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include "stdlib.h"
#include "cublas_v2.h"

class MemoryPool
{
    size_t next;
    size_t size;
    void* pool;
    cublasHandle_t handle;
public:
    MemoryPool(cublasHandle_t handle = NULL);
    bool createPool(size_t sz);
    size_t availablePool();
    void* allocateMemory(size_t sz);
    void freeMemory(void *mem, size_t sz);
    void destroyPool();
};

#endif // MEMORYPOOL_H
