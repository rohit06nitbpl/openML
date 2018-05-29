#ifndef SOFTMAXCELL_H
#define SOFTMAXCELL_H

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "stdlib.h"
#include "utility.h"
#include "activationfunctions.h"
#include "templatedutility.h"
#include "kernels.h"
#include "memorypool.h"

template <class T>
class SoftmaxCell
{
    T *W_VH;
    T *V_BIAS;

    T *GRAD_W_VH;
    T *GRAD_V_BIAS;

    T *D_W_VH;
    T *D_V_BIAS;

    T *D_GRAD_W_VH;
    T *D_GRAD_V_BIAS;

    T *V_SCORE;
    T *D_V_SCORE;

    T *WINNER_INDEX;
    T *D_WINNER_INDEX;

    size_t N_V;
    size_t N_H;

    MATRIX_TYPE type;
    cublasHandle_t handle;

    MemoryPool *pool;
    MemoryPool *gpupool;

    ActivationFunctions<T>* a_func;
    std::fstream &dumpfile;

public:
    SoftmaxCell(std::fstream &dumpfile, MATRIX_TYPE type = COLUMN_MAJOR);
    ~SoftmaxCell();

    void setActivationFunction(ActivationFunctions<T> *a_f);
    void setCublasHandle(cublasHandle_t handle);
    cublasHandle_t getCublasHandle();
    void setCPUPOOL(MemoryPool *pool);
    void setGPUPOOL(MemoryPool *pool);
    void setDumpFileHandle(std::fstream &dumpfile);

    void setOutputSize(size_t N_V);
    void setHiddenInputSize(size_t N_H);

    bool initialize(size_t N_V, size_t N_H, MATRIX_TYPE type, T* W_VH = NULL, T* V_BIAS = NULL);

    bool setWVH(T* W_VH, size_t sz, MATRIX_TYPE type);
    bool setVBIAS(T* V_BIAS, size_t sz, MATRIX_TYPE type);
    bool createVSCORE();
    bool createWinnerIndex();

    void updateCell();
    void propagateDeltaBack(T* delta);
    void activate(T* input);
    void classify(T* wIndex);
    void setError(size_t target_index);
    void accumulateGradient(T* input);
    void backwardPass(T target_index, T* input, T* delta_back);
    void print(std::ostream &out);
    void printwvh(std::ostream &out);
    void printvbias(std::ostream &out);
};

#include "softmaxcell.cpp"

#endif // SOFTMAXCELL_H
