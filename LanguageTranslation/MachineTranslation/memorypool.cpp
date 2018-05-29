#include "memorypool.h"

MemoryPool::MemoryPool(cublasHandle_t handle):next(0),size(0),pool(NULL),handle(handle)
{

}

bool MemoryPool::createPool(size_t sz)
{
    if(!handle) {
        //pool = malloc(sz);
        if(1) {
            size = sz;
            return true;
        }
        return false;
    } else {
        return false;
    }

}

size_t MemoryPool::availablePool()
{
    if(!handle) {
        return size-next;
    } else {
        return 0;
    }
}

void *MemoryPool::allocateMemory(size_t sz)
{
    if(!handle) {
        if(availablePool() >= sz) {
            next += sz;
            //return pool+next;
            return malloc(sz);
        } else {
            return NULL;
        }
    } else {
        return NULL;
    }

}

void MemoryPool::freeMemory(void *mem, size_t sz)
{
    if(!handle) {
        free(mem);
        next -= sz;
    } else {

    }
}

void MemoryPool::destroyPool()
{
    if(!handle) {
        //free(pool);
    } else {

    }
}

