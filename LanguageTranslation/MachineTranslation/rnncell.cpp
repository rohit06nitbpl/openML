#ifndef RNNCELL_CPP
#define RNNCELL_CPP

#include "string.h"
#include <cuda_runtime.h>
#include "rnncell.h"
#include "utility.h"
#include "enums.h"
#include "templatedutility.h"
#include "kernels.h"

template<class T>
RNNCell<T>::RNNCell(MATRIX_TYPE type):type(type),handle(NULL),pool(NULL),gpupool(NULL)
{


}

template<class T>
RNNCell<T>::~RNNCell()
{

}

template<class T>
void RNNCell<T>::setCublasHandle(cublasHandle_t handle)
{
    this->handle = handle;
}

template<class T>
cublasHandle_t RNNCell<T>::getCublasHandle()
{
    return handle;
}

template<class T>
void RNNCell<T>::setCPUPOOL(MemoryPool *pool)
{
    this->pool = pool;
}

template<class T>
void RNNCell<T>::setGPUPOOL(MemoryPool *pool)
{
    this->gpupool = pool;
}

/*template<class T>
void RNNCell<T>::setDumpFileHandle(std::fstream &dumpfile)
{
    this->dumpfile = dumpfile;
}*/

template<class T>
void RNNCell<T>::updateCell()
{

}
template<class T>
void RNNCell<T>::accumulateGradient(T* hidden_in, T *data_in, T* derivative_out, T* delta_in)
{

}
template<class T>
void RNNCell<T>::propagateDeltaBack(T* delta)
{

}

template<class T>
void RNNCell<T>::stepForward(T* hidden_in, T* data_in, T* hidden_out, T *derivative_out)
{

}

template<class T>
void RNNCell<T>::print(std::ostream &out)
{

}

template<class T>
void RNNCell<T>::printwhh(std::ostream &out)
{

}

template<class T>
void RNNCell<T>::printwhx(std::ostream &out)
{

}
template<class T>
void RNNCell<T>::printvbias(std::ostream &out)
{

}

template<class T>
SimpleRNNCell<T>::SimpleRNNCell(std::fstream &dumpfile, MATRIX_TYPE type):RNNCell<T>(type),W_HH(NULL),W_HX(NULL),V_BIAS(NULL),N_H(0),N_X(0),\
                                                  GRAD_W_HH(NULL),GRAD_W_HX(NULL),GRAD_V_BIAS(NULL),\
                                                  D_W_HH(NULL),D_W_HX(NULL),D_V_BIAS(NULL),\
                                                  D_GRAD_W_HH(NULL),D_GRAD_W_HX(NULL),D_GRAD_V_BIAS(NULL),a_func(NULL),\
                                                  dumpfile(dumpfile)
{

}

template<class T>
SimpleRNNCell<T>::~SimpleRNNCell() {

}

template<class T>
void SimpleRNNCell<T>::setActivationFunction(ActivationFunctions<T> *a_f)
{
    this->a_func = a_f;
}

template<class T>
void SimpleRNNCell<T>::setHiddenInputSize(size_t N_H)
{
    this->N_H = N_H;
}

template<class T>
void SimpleRNNCell<T>::setDataInputSize(size_t N_X)
{
    this->N_X = N_X;
}

template<class T>
bool SimpleRNNCell<T>::initialize(size_t N_H, size_t N_X, MATRIX_TYPE type, T *W_HH, T *W_HX, T *V_BIAS)
{
    if(type == COLUMN_MAJOR) {
        this->N_H = N_H;
        this->N_X = N_X;
        if(W_HX == NULL && W_HH == NULL && V_BIAS == NULL) {

            T* W1 = (T*)this->pool->allocateMemory(sizeof(T)*N_H*N_X);
            T* W2 = (T*)this->pool->allocateMemory(sizeof(T)*N_H*N_H);
            T* V = (T*)this->pool->allocateMemory(sizeof(T)*N_H);
            if(W1 && W2 && V) {
                T r1 = sqrt(6.0/double((N_H+N_X)));
                T r2 = sqrt(6.0/double((N_H+N_H)));
                for(size_t i = 0; i<(N_H*N_X); i++) {
                    W1[i] = randomInRange(-r1,r1);
                }
                for(size_t i = 0; i<(N_H*N_H); i++) {
                    W2[i] = randomInRange(-r2,r2);
                }
                for(size_t i = 0; i<N_H; i++) {
                    V[i] = rand01;
                }
            }

            if (this->setWHX(W1,N_H*N_X,type) && \
                this->setWHH(W2,N_H*N_H,type) && \
                this->setVBIAS(V,N_H,type)) {

                if(PRINT) PRINTMATRIX("INITIAL W_HX\n",W1,N_H,N_X,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL W_HH\n",W2,N_H,N_H,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL V_BIAS\n",V,N_H,1,this->handle,this->dumpfile);
            } else {
                printf("Failed in Initialization of RNN Cell\n");
            }

            this->pool->freeMemory(W1,sizeof(T)*N_H*N_X);
            this->pool->freeMemory(W2,sizeof(T)*N_H*N_H);
            this->pool->freeMemory(V,sizeof(T)*N_H);

        } else if (W_HX != NULL && W_HH != NULL && V_BIAS != NULL) {

            if(this->setWHX(W_HX,N_H*N_X,type) && \
               this->setWHH(W_HH,N_H*N_H,type) && \
               this->setVBIAS(V_BIAS,N_H,type)) {
                if(PRINT) PRINTMATRIX("INITIAL W_HX\n",W_HX,N_H,N_X,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL W_HH\n",W_HH,N_H,N_H,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL V_BIAS\n",V_BIAS,N_H,1,this->handle,this->dumpfile);
            } else {
                printf("Failed in Initialization of RNN Cell\n");
            }

        } else {
            //ALL OTHER CASE NOT SUPPORTED
            return false;
        }

    } else {
        // THIS CASE CAN BE ADDED LATER
        return false;
    }
    return true;
}

template<class T>
bool SimpleRNNCell<T>::setWHH(T *W_HH, size_t sz, MATRIX_TYPE type)
{
    if(this->handle == NULL) {
        if(W_HH != NULL && N_H != 0 && sz == N_H*N_H) {
            this->W_HH = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            this->GRAD_W_HH = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->W_HH != NULL && this->GRAD_W_HH != NULL) {
                setContiguousMatrix(W_HH,this->W_HH,N_H,N_H,type,this->type);
                memset(this->GRAD_W_HH,0,sizeof(T)*sz);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(W_HH != NULL && N_H != 0 && sz == N_H*N_H && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_W_HH,sizeof(T)*sz));
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_GRAD_W_HH,sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),W_HH,1,D_W_HH,1));
            T* V_ZERO = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if(V_ZERO) memset(V_ZERO,0,sizeof(T)*sz);
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),V_ZERO,1,D_GRAD_W_HH,1));
            this->pool->freeMemory(V_ZERO,sizeof(T)*sz);
            return true;
        } else {
            return false;
        }
    }
}

template<class T>
bool SimpleRNNCell<T>::setWHX(T *W_HX, size_t sz, MATRIX_TYPE type)
{
    if(this->handle == NULL) {
        if(W_HX != NULL && N_H != 0 && N_X != 0 && sz == N_H*N_X) {
            this->W_HX = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            this->GRAD_W_HX = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->W_HX != NULL && this->GRAD_W_HX != NULL) {
                setContiguousMatrix(W_HX,this->W_HX,N_H,N_X,type,this->type);
                memset(this->GRAD_W_HX,0,sizeof(T)*sz);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(W_HX != NULL && N_H != 0 && N_X != 0 && sz == N_H*N_X && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_W_HX,sizeof(T)*sz));
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_GRAD_W_HX,sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),W_HX,1,D_W_HX,1));
            T* V_ZERO = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if(V_ZERO) memset(V_ZERO,0,sizeof(T)*sz);
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),V_ZERO,1,D_GRAD_W_HX,1));
            this->pool->freeMemory(V_ZERO,sizeof(T)*sz);
            return true;
        } else {
            return false;
        }
    }
}

template<class T>
bool SimpleRNNCell<T>::setVBIAS(T *V_BIAS, size_t sz, MATRIX_TYPE type)
{
    if(this->handle == NULL) {
        if(V_BIAS != NULL && N_H != 0 && sz == N_H) {
            this->V_BIAS = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            this->GRAD_V_BIAS = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->V_BIAS != NULL && this->GRAD_V_BIAS != NULL) {
                setContiguousMatrix(V_BIAS,this->V_BIAS,N_H,1,type,this->type);
                memset(this->GRAD_V_BIAS,0,sizeof(T)*sz);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(V_BIAS != NULL && N_H != 0 && sz == N_H && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_V_BIAS,sizeof(T)*sz));
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_GRAD_V_BIAS,sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),V_BIAS,1,D_V_BIAS,1));
            T* V_ZERO = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if(V_ZERO) memset(V_ZERO,0,sizeof(T)*sz);
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),V_ZERO,1,D_GRAD_V_BIAS,1));
            this->pool->freeMemory(V_ZERO,sizeof(T)*sz);
            return true;
        } else {
            return false;
        }

    }

}

template<class T>
void SimpleRNNCell<T>::updateCell()
{
    T alpha = 1.0;
    T beta = 1.0;
    T alpha_reset = 0.0;
    T beta_reset = 0.0;
    if(this->handle == NULL) {
        if(geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_H,&alpha,W_HH,N_H,&beta,GRAD_W_HH,N_H,W_HH,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_X,&alpha,W_HX,N_H,&beta,GRAD_W_HX,N_H,W_HX,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,V_BIAS,N_H,&beta,GRAD_V_BIAS,N_H,V_BIAS,N_H) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_H,&alpha_reset,GRAD_W_HH,N_H,&beta_reset,GRAD_W_HH,N_H,GRAD_W_HH,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_X,&alpha_reset,GRAD_W_HX,N_H,&beta_reset,GRAD_W_HX,N_H,GRAD_W_HX,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha_reset,GRAD_V_BIAS,N_H,&beta_reset,GRAD_V_BIAS,N_H,GRAD_V_BIAS,N_H)){

            int* _null = NULL;
            if(PRINT) PRINTMATRIX("RNN CELL AFTER UPDATE\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("W_HH\n",W_HH,N_H,N_H,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("W_HX\n",W_HX,N_H,N_X,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("V_BIAS\n",V_BIAS,N_H,1,this->handle,this->dumpfile);

        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if(geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_H,&alpha,D_W_HH,N_H,&beta,D_GRAD_W_HH,N_H,D_W_HH,N_H,this->handle) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_X,&alpha,D_W_HX,N_H,&beta,D_GRAD_W_HX,N_H,D_W_HX,N_H,this->handle) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,D_V_BIAS,N_H,&beta,D_GRAD_V_BIAS,N_H,D_V_BIAS,N_H,this->handle) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_H,&alpha_reset,D_GRAD_W_HH,N_H,&beta_reset,D_GRAD_W_HH,N_H,D_GRAD_W_HH,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,N_X,&alpha_reset,D_GRAD_W_HX,N_H,&beta_reset,D_GRAD_W_HX,N_H,D_GRAD_W_HX,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha_reset,D_GRAD_V_BIAS,N_H,&beta_reset,D_GRAD_V_BIAS,N_H,D_GRAD_V_BIAS,N_H)){

            int* _null = NULL;
            if(PRINT) PRINTMATRIX("RNN CELL AFTER UPDATE\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("W_HH\n",D_W_HH,N_H,N_H,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("W_HX\n",D_W_HX,N_H,N_X,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("V_BIAS\n",D_V_BIAS,N_H,1,this->handle,this->dumpfile);

        } else {
            exit(EXIT_FAILURE);
        }
    }

}

template<class T>
void SimpleRNNCell<T>::accumulateGradient(T* hidden_in, T *data_in, T* derivative_out, T* delta_in)
{
    int* _null = NULL;
    if(PRINT) PRINTMATRIX("RNN CELL ACCUMULATE GRADIENT\n",_null,0,0,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("HIDDEN IN\n",hidden_in,N_H,1,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DATA IN\n",data_in,N_X,1,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DERIVATIVE OUT\n",derivative_out,N_H,1,this->handle,this->dumpfile);
    // DELTA_IN GETS CHANGED, ALWAYS FIRST RUN THIS METHOD THEN RUN "propagateDeltaBack" USING DELTA_IN MODIFIED BY THIS METHOD
    T alpha = 1.0;
    T beta = 1.0;
    if(this->handle == NULL) {
        if(multVectorElementByElement(derivative_out,delta_in,delta_in,N_H) && \
           gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_H,N_H,1,&alpha,delta_in,N_H,hidden_in,N_H,&beta,GRAD_W_HH,N_H) && \
           gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_H,N_X,1,&alpha,delta_in,N_H,data_in,N_X,&beta,GRAD_W_HX,N_H) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,delta_in,N_H,&beta,GRAD_V_BIAS,N_H,GRAD_V_BIAS,N_H)) {
            if(PRINT) PRINTMATRIX("GRAD_WHH\n",GRAD_W_HH,N_H,N_H,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_WHX\n",GRAD_W_HX,N_H,N_X,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_V_BAIS\n",GRAD_V_BIAS,N_H,1,this->handle,this->dumpfile);

        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if(mult_vector_element_by_element(derivative_out,delta_in,delta_in,N_H) && \
           gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_H,N_H,1,&alpha,delta_in,N_H,hidden_in,N_H,&beta,D_GRAD_W_HH,N_H,this->handle) && \
           gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_H,N_X,1,&alpha,delta_in,N_H,data_in,N_X,&beta,D_GRAD_W_HX,N_H,this->handle) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,delta_in,N_H,&beta,D_GRAD_V_BIAS,N_H,D_GRAD_V_BIAS,N_H,this->handle)) {
            if(PRINT) PRINTMATRIX("GRAD_WHH\n",D_GRAD_W_HH,N_H,N_H,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_WHX\n",D_GRAD_W_HX,N_H,N_X,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_V_BAIS\n",D_GRAD_V_BIAS,N_H,1,this->handle,this->dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }
    }

    if(PRINT) PRINTMATRIX("DELTA MULTIPLIED WITH DERIVATIVE OUT(ELEMENT BY ELEMENT)\n",delta_in,N_H,1,this->handle,this->dumpfile);


}

template<class T>
void SimpleRNNCell<T>::propagateDeltaBack(T* delta)
{
    T alpha = 1.0;
    T beta = 0.0; // DELTA GETS OVERWRITTEN AND FINALLY HOLDS DELTA FOR BACK OF CELL (OR DELTA OUT)
    if(this->handle == NULL) {
        if(gemv(CUBLAS_OP_T,N_H,N_H,&alpha,W_HH,N_H,delta,1,&beta,delta,1)){

        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if(gemv(CUBLAS_OP_T,N_H,N_H,&alpha,D_W_HH,N_H,delta,1,&beta,delta,1,this->handle)){

        } else {
            exit(EXIT_FAILURE);
        }

    }
    int* _null = NULL;
    if(PRINT) PRINTMATRIX("RNN CELL PROPAGATE DELTA BACK\n",_null,0,0,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DELTA BACK\n",delta,N_H,1,this->handle,this->dumpfile);
}

template<class T>
void SimpleRNNCell<T>::stepForward(T *hidden_in, T *data_in, T *hidden_out, T* derivative_out)
{
    // RE-WRITES DERIVATIVE OUT
    T alpha = 1.0;
    T beta_first_time = 0.0; //REWRITING HIDDEN_OUT IN FORWARD PASS
    T beta = 1.0;
    if(this->handle == NULL) {
        if(gemv(CUBLAS_OP_N,N_H,N_H,&alpha,W_HH,N_H,hidden_in,1,&beta_first_time,hidden_out,1) &&\
           gemv(CUBLAS_OP_N,N_H,N_X,&alpha,W_HX,N_H,data_in,1,&beta,hidden_out,1) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,V_BIAS,N_H,&beta,hidden_out,N_H,hidden_out,N_H) &&\
           this->a_func->getActivation(hidden_out,hidden_out,N_H) &&\
           this->a_func->getDerivative(hidden_out,derivative_out,N_H)){

        } else {
            exit(EXIT_FAILURE);
        }
    } else {
        if(gemv(CUBLAS_OP_N,N_H,N_H,&alpha,D_W_HH,N_H,hidden_in,1,&beta_first_time,hidden_out,1,this->handle) &&\
           gemv(CUBLAS_OP_N,N_H,N_X,&alpha,D_W_HX,N_H,data_in,1,&beta,hidden_out,1,this->handle) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_H,1,&alpha,D_V_BIAS,N_H,&beta,hidden_out,N_H,hidden_out,N_H,this->handle) &&\
           this->a_func->getActivation(hidden_out,hidden_out,N_H,this->handle) &&\
           this->a_func->getDerivative(hidden_out,derivative_out,N_H)){

        } else {
            exit(EXIT_FAILURE);
        }
    }
    int* _null = NULL;
    if(PRINT) PRINTMATRIX("RNN CELL STEP FORWARD\n",_null,0,0,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("HIDDEN IN\n",hidden_in,N_H,1,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DATA IN\n",data_in,N_X,1,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("HIDDEN OUT\n",hidden_out,N_H,1,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DERIVATIVE OUT\n",derivative_out,N_H,1,this->handle,this->dumpfile);
}

template<class T>
void SimpleRNNCell<T>::print(std::ostream &out)
{
    PRINTMATRIX("W_HH\n",W_HH,N_H,N_H,this->handle,out);
    PRINTMATRIX("W_HX\n",W_HX,N_H,N_X,this->handle,out);
    PRINTMATRIX("V_BIAS\n",V_BIAS,N_H,1,this->handle,out);
}
template<class T>
void SimpleRNNCell<T>::printwhh(std::ostream &out)
{
    PRINTMATRIX("",W_HH,N_H,N_H,this->handle,out);
}

template<class T>
void SimpleRNNCell<T>::printwhx(std::ostream &out)
{
    PRINTMATRIX("",W_HX,N_H,N_X,this->handle,out);
}
template<class T>
void SimpleRNNCell<T>::printvbias(std::ostream &out)
{
    PRINTMATRIX("",V_BIAS,N_H,1,this->handle,out);
}

#endif // RNNCELL_CPP
