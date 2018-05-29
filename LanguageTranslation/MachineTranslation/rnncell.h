#ifndef RNNCELL_H
#define RNNCELL_H

#include "stdlib.h"
#include "cublas_v2.h"
#include <fstream>

#include "enums.h"
#include "activationfunctions.h"
#include "utility.h"
#include "memorypool.h"


template <class T>
class RNNCell
{
protected:
    MATRIX_TYPE type;
    cublasHandle_t  handle;
    MemoryPool *pool;
    MemoryPool *gpupool;

public:
    RNNCell(MATRIX_TYPE type = COLUMN_MAJOR);
    ~RNNCell();
    void setCublasHandle(cublasHandle_t  handle);
    cublasHandle_t getCublasHandle();
    void setCPUPOOL(MemoryPool *pool);
    void setGPUPOOL(MemoryPool *pool);
    void setDumpFileHandle(std::fstream &dumpfile);
    virtual void updateCell();
    virtual void accumulateGradient(T* hidden_in, T *data_in,T* derivative_out, T* delta_in);
    virtual void propagateDeltaBack(T* delta);
    virtual void stepForward(T* hidden_in, T* data_in, T* hidden_out, T* derivative_out);
    virtual void print(std::ostream & out);
    virtual void printwhh(std::ostream & out);
    virtual void printwhx(std::ostream & out);
    virtual void printvbias(std::ostream & out);

};

template <class T>
class SimpleRNNCell : public RNNCell<T>
{
    T *W_HH;
    T *W_HX;
    T *V_BIAS;

    T *GRAD_W_HH;
    T *GRAD_W_HX;
    T *GRAD_V_BIAS;

    T *D_W_HH;
    T *D_W_HX;
    T *D_V_BIAS;

    T *D_GRAD_W_HH;
    T *D_GRAD_W_HX;
    T *D_GRAD_V_BIAS;


    size_t N_H;
    size_t N_X;

    ActivationFunctions<T>* a_func;
    std::fstream &dumpfile;

public:
    SimpleRNNCell(std::fstream &dumpfile, MATRIX_TYPE type = COLUMN_MAJOR);
    ~SimpleRNNCell();

    void setActivationFunction(ActivationFunctions<T> *a_f);
    void setHiddenInputSize(size_t N_H);
    void setDataInputSize(size_t N_X);

    bool initialize(size_t N_H, size_t N_X, MATRIX_TYPE type, T* W_HH = NULL, T* W_HX = NULL, T* V_BIAS = NULL);

    bool setWHH(T* W_HH, size_t sz, MATRIX_TYPE type);
    bool setWHX(T* W_HX, size_t sz, MATRIX_TYPE type);
    bool setVBIAS(T* V_BIAS, size_t sz, MATRIX_TYPE type);

    virtual void updateCell();
    virtual void accumulateGradient(T* hidden_in, T *data_in,T* derivative_out,T* delta_in);
    virtual void propagateDeltaBack(T* delta);
    virtual void stepForward(T* hidden_in, T* data_in, T* hidden_out, T* derivative_out);
    virtual void print(std::ostream & out);
    virtual void printwhh(std::ostream & out);
    virtual void printwhx(std::ostream & out);
    virtual void printvbias(std::ostream & out);


};

#include "rnncell.cpp"

#endif // RNNCELL_H

