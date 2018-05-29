#ifndef UTILITY_H
#define UTILITY_H

#include "enums.h"

#include "stdio.h"
#include "cublas_v2.h"
#include "stdlib.h"
#include <iostream>

float randomInRange(float min, float max);
double randomInRange(double min, double max);

int min(int a, int b);
size_t min(size_t a, size_t b);
size_t max(size_t a, size_t b);
size_t ceil(size_t n, size_t k);

const char * cuBLASStatusString(cublasStatus_t stat);

bool gemv(cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float                 *y, int incy,
                           cublasHandle_t handle = NULL);

bool gemv(cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double                *y, int incy,
                           cublasHandle_t handle = NULL);

bool gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float                 *C, int ldc,
                           cublasHandle_t handle = NULL);

bool gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double                *C, int ldc,
                           cublasHandle_t handle = NULL);

bool geam(
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const float *alpha,
        const float *A, int lda,
        const float *beta,
        const float *B, int ldb,
        float *C, int ldc,
        cublasHandle_t handle = NULL);

bool geam(
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const double *alpha,
        const double *A, int lda,
        const double *beta,
        const double *B, int ldb,
        double *C, int ldc,
        cublasHandle_t handle = NULL);


void PRINTMATRIX(const char* s, int* mat, size_t row, size_t col, cublasHandle_t handle,std::ostream &file = std::cout, MATRIX_TYPE type = COLUMN_MAJOR);
void PRINTMATRIX(const char* s, float* mat, size_t row, size_t col, cublasHandle_t handle,std::ostream &file = std::cout, MATRIX_TYPE type = COLUMN_MAJOR);
void PRINTMATRIX(const char* s, double* mat, size_t row, size_t col, cublasHandle_t handle,std::ostream &file = std::cout, MATRIX_TYPE type = COLUMN_MAJOR);

#endif // UTILITY_H
