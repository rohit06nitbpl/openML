#include "assert.h"
#include "utility.h"
#include "templatedutility.h"



float randomInRange(float min, float max)
{
    // this  function assumes max > min, you may want
    // more robust error checking for a non-debug build
    assert(max > min);
    float random = ((float) rand()) / (float) RAND_MAX;

    // generate (in your case) a float between 0 and (4.5-.78)
    // then add .78, giving you a float between .78 and 4.5
    float range = max - min;
    return (random*range) + min;
}


double randomInRange(double min, double max)
{
    // this  function assumes max > min, you may want
    // more robust error checking for a non-debug build
    assert(max > min);
    double random = ((double) rand()) / (double) RAND_MAX;

    // generate (in your case) a float between 0 and (4.5-.78)
    // then add .78, giving you a float between .78 and 4.5
    double range = max - min;
    return (random*range) + min;
}


const char * cuBLASStatusString(cublasStatus_t stat)
{
    switch (stat)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";

    }

    return "UNKNOWN";
}


bool gemv(cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float                 *y, int incy,
                           cublasHandle_t handle) {

    if(handle == NULL) {
        if(trans == CUBLAS_OP_N) {
            return multMatrixVector(false,m,n,alpha,A,x,beta,y,COLUMN_MAJOR);
        } else if (trans == CUBLAS_OP_T ) {
            return multMatrixVector(true,m,n,alpha,A,x,beta,y,COLUMN_MAJOR);
        } else {
            return false;
        }

    } else {
        CUBLAS_SAFE_CALL(cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
        return true;
    }
    return false;
}

bool gemv(cublasOperation_t trans,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *x, int incx,
                           const double          *beta,
                           double                *y, int incy,
                           cublasHandle_t handle) {

    if(handle == NULL) {
        if(trans == CUBLAS_OP_N) {
            return multMatrixVector(false,m,n,alpha,A,x,beta,y,COLUMN_MAJOR);
        } else if (trans == CUBLAS_OP_T ) {
            return multMatrixVector(true,m,n,alpha,A,x,beta,y,COLUMN_MAJOR);
        } else {
            return false;
        }
    } else {
        CUBLAS_SAFE_CALL(cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
        return true;
    }
    return false;
}

bool gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float                 *C, int ldc,
                           cublasHandle_t handle) {
    if(handle == NULL) {
        if(transa == CUBLAS_OP_N && transb == CUBLAS_OP_T) {
            return multMatrixMatrix(false,true,m,n,k,alpha,A,B,beta,C,COLUMN_MAJOR);
        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        CUBLAS_SAFE_CALL(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
        return true;
    }
    return false;
}

bool gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double                *C, int ldc,
                           cublasHandle_t handle) {
    if(handle == NULL) {
        if(transa == CUBLAS_OP_N && transb == CUBLAS_OP_T) {
            return multMatrixMatrix(false,true,m,n,k,alpha,A,B,beta,C,COLUMN_MAJOR);
        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        CUBLAS_SAFE_CALL(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
        return true;
    }
    return false;
}

bool geam(
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const float *alpha,
        const float *A, int lda,
        const float *beta,
        const float *B, int ldb,
        float *C, int ldc,
        cublasHandle_t handle)
{
    if(handle == NULL) {
        if(transa == CUBLAS_OP_N && transb == CUBLAS_OP_N) {
            return addMatrixMatrix(false,false,m,n,alpha,A,beta,B,C,COLUMN_MAJOR);
        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        CUBLAS_SAFE_CALL(cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
        return true;
    }
    return false;

}

bool geam(
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const double *alpha,
        const double *A, int lda,
        const double *beta,
        const double *B, int ldb,
        double *C, int ldc,
        cublasHandle_t handle)
{
    if(handle == NULL) {
        if(transa == CUBLAS_OP_N && transb == CUBLAS_OP_N) {
            return addMatrixMatrix(false,false,m,n,alpha,A,beta,B,C,COLUMN_MAJOR);
        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        CUBLAS_SAFE_CALL(cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
        return true;
    }
    return false;

}


int min(int a, int b)
{
    return (a<b)?a:b;
}

size_t min(size_t a, size_t b)
{
    return (a<b)?a:b;
}

size_t max(size_t a, size_t b)
{
    return (a>b)?a:b;
}

size_t ceil(size_t n, size_t k)
{
    return (n % k) ? n/k+1 : n/k;
}


void PRINTMATRIX(const char* s, int* mat, size_t row, size_t col, cublasHandle_t handle, std::ostream &file, MATRIX_TYPE type) {
    if(!handle) {
        if(file.good()) {
            file<<s;
            printContiguousMatrix(mat,row,col,type,file);
        } else {
            printf("could not write to stream in PRINTMATRIX methos\n");
        }
    } else if (handle) {

    }

}


void PRINTMATRIX(const char* s, float* mat, size_t row, size_t col, cublasHandle_t handle, std::ostream &file, MATRIX_TYPE type) {
    if(!handle) {
        if(file.good()) {
            file<<s;
            printContiguousMatrix(mat,row,col,type,file);
        } else {
            printf("could not write to stream in PRINTMATRIX methos\n");
        }
    } else if (handle) {

    }

}


void PRINTMATRIX(const char* s, double* mat, size_t row, size_t col, cublasHandle_t handle, std::ostream &file, MATRIX_TYPE type) {
    if(!handle) {
        if(file.good()) {
            file<<s;
            printContiguousMatrix(mat,row,col,type,file);
        } else {
            printf("could not write to stream in PRINTMATRIX methos\n");
        }
    } else if (handle) {

    }

}
