#ifndef ENUMS_H
#define ENUMS_H

#include "stdlib.h"

#define PRINT 0
#define PRINT_LIMITED 1
#define EPOCH 1
#define GPU 0

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define CUBLAS_SAFE_CALL(call)                                        \
do {                                                                  \
    cublasStatus_t  stat = call;                                      \
    if (CUBLAS_STATUS_SUCCESS != stat) {                              \
        fprintf (stderr, "Cublas error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cuBLASStatusString(stat) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define rand01 ((double) rand() / (RAND_MAX))


enum MATRIX_TYPE
{
    ROW_MAJOR,
    COLUMN_MAJOR
};

struct bucket
{
    size_t data_points;
    size_t first_sen_len;
    size_t second_sen_len;
};


#endif // ENUMS_H

