#ifndef SOFTMAXCELL_CPP
#define SOFTMAXCELL_CPP

#include "softmaxcell.h"


template<class T>
SoftmaxCell<T>::SoftmaxCell(std::fstream &dumpfile, MATRIX_TYPE type):type(type),handle(NULL),W_VH(NULL),V_BIAS(NULL),GRAD_W_VH(NULL),GRAD_V_BIAS(NULL),\
                                              D_W_VH(NULL),D_V_BIAS(NULL),D_GRAD_W_VH(NULL),D_GRAD_V_BIAS(NULL),\
                                              WINNER_INDEX(NULL),D_WINNER_INDEX(NULL),pool(NULL),gpupool(NULL),a_func(NULL),\
                                              dumpfile(dumpfile)
{

}

template<class T>
SoftmaxCell<T>::~SoftmaxCell()
{

}

template<class T>
void SoftmaxCell<T>::setActivationFunction(ActivationFunctions<T> *a_f)
{
    this->a_func = a_f;
}

template<class T>
void SoftmaxCell<T>::setCublasHandle(cublasHandle_t handle)
{
    this->handle = handle;
}

template<class T>
cublasHandle_t SoftmaxCell<T>::getCublasHandle()
{
    return handle;
}

template<class T>
void SoftmaxCell<T>::setCPUPOOL(MemoryPool *pool)
{
    this->pool = pool;
}

template<class T>
void SoftmaxCell<T>::setGPUPOOL(MemoryPool *pool)
{
    this->gpupool = pool;
}

/*template<class T>
void SoftmaxCell<T>::setDumpFileHandle(std::fstream &dumpfile)
{
    this->dumpfile = dumpfile;
}*/

template<class T>
void SoftmaxCell<T>::setOutputSize(size_t N_V)
{
    this->N_V = N_V;
}

template<class T>
void SoftmaxCell<T>::setHiddenInputSize(size_t N_H)
{
    this->N_H = N_H;
}

template<class T>
bool SoftmaxCell<T>::initialize(size_t N_V, size_t N_H, MATRIX_TYPE type, T *W_VH, T *V_BIAS)
{
    if(type == COLUMN_MAJOR) {
        this->N_H = N_H;
        this->N_V = N_V;
        if(W_VH == NULL && V_BIAS == NULL) {

            T* W = (T*)this->pool->allocateMemory(sizeof(T)*N_V*N_H);
            T* V = (T*)this->pool->allocateMemory(sizeof(T)*N_V);
            if(W && V) {
                T r = sqrt(6.0/double(N_V+N_H));
                for(size_t i = 0; i<(N_V*N_H); i++) {
                    W[i] = randomInRange(-r,r);
                }
                for(size_t i = 0; i<N_V; i++) {
                    V[i] = rand01;
                }
            }

            if(this->setWVH(W,N_V*N_H,type) && \
               this->setVBIAS(V,N_V,type) && \
               this->createVSCORE() && \
               this->createWinnerIndex()) {
                if(PRINT) PRINTMATRIX("INITIAL W_VH\n",W,N_V,N_H,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL V_BIAS\n",V,N_V,1,this->handle,this->dumpfile);
            } else {
                printf("Failed in Initialization of SOFTMAX Cell\n");
            }

            this->pool->freeMemory(W,sizeof(T)*N_V*N_H);
            this->pool->freeMemory(V,sizeof(T)*N_V);

        } else if (W_VH != NULL && V_BIAS != NULL) {

            if(this->setWVH(W_VH,N_V*N_H,type) &&\
               this->setVBIAS(V_BIAS,N_V,type) &&\
               this->createVSCORE() && \
               this->createWinnerIndex()) {
                if(PRINT) PRINTMATRIX("INITIAL W_VH\n",W_VH,N_V,N_H,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("INITIAL V_BIAS\n",V_BIAS,N_V,1,this->handle,this->dumpfile);
            } else {
                printf("Failed in Initialization of SOFTMAX Cell\n");
            }

        } else {
            //ALL OTHER CASE NOT SUPPORTED
            return false;
        }

    } else {
        // THIS CASE CAN BE ADDED LATER
        return false;
    }
}

template<class T>
bool SoftmaxCell<T>::setWVH(T *W_VH, size_t sz, MATRIX_TYPE type)
{
    if(handle == NULL) {
        if(W_VH != NULL && N_H != 0 && N_V != 0 && sz == N_H*N_V) {
            this->W_VH = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            this->GRAD_W_VH = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->W_VH != NULL && this->GRAD_W_VH != NULL) {
                setContiguousMatrix(W_VH,this->W_VH,N_V,N_H,type,this->type);
                memset(this->GRAD_W_VH,0,sizeof(T)*sz);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(W_VH != NULL && N_H != 0 && N_V != 0 && sz == N_H*N_V && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_W_VH,sizeof(T)*sz));
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_GRAD_W_VH,sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),W_VH,1,D_W_VH,1));
            T* V_ZERO = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if(V_ZERO) memset(V_ZERO,0,sizeof(T)*sz);
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),V_ZERO,1,D_GRAD_W_VH,1));
            this->pool->freeMemory(V_ZERO,sizeof(T)*sz);
            return true;
        } else {
            return false;
        }

    }
}

template<class T>
bool SoftmaxCell<T>::setVBIAS(T *V_BIAS, size_t sz, MATRIX_TYPE type)
{
    if(handle == NULL) {
        if(V_BIAS != NULL && N_V != 0 && sz == N_V) {
            this->V_BIAS = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            this->GRAD_V_BIAS = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->V_BIAS != NULL && this->GRAD_V_BIAS != NULL) {
                setContiguousMatrix(V_BIAS,this->V_BIAS,N_V,1,type,this->type);
                memset(this->GRAD_V_BIAS,0,sizeof(T)*sz);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(V_BIAS != NULL && N_V != 0 && sz == N_V && type == COLUMN_MAJOR) {
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
bool SoftmaxCell<T>::createVSCORE()
{
    if(handle == NULL) {
        if(N_V != 0 ) {
            this->V_SCORE = (T*)this->pool->allocateMemory(sizeof(T)*N_V);
            if(V_SCORE) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }

    } else {
        if(N_V != 0 ) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_V_SCORE,sizeof(T)*N_V));
        } else {
            return false;
        }
    }

}

template<class T>
bool SoftmaxCell<T>::createWinnerIndex()
{
    if(handle == NULL) {
        if(1) {
            this->WINNER_INDEX = (T*)this->pool->allocateMemory(sizeof(T)*1);
            if(WINNER_INDEX) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }

    } else {
        if(1) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&D_WINNER_INDEX,sizeof(T)*1));
        } else {
            return false;
        }
    }


}

template<class T>
void SoftmaxCell<T>::updateCell()
{
    T alpha = 1.0;
    T beta = 1.0;
    T alpha_reset = 0.0;
    T beta_reset = 0.0;
    int* _null = NULL;
    if(handle == NULL) {
        if(geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,N_H,&alpha,W_VH,N_V,&beta,GRAD_W_VH,N_V,W_VH,N_V) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,V_BIAS,N_V,&beta,GRAD_V_BIAS,N_V,V_BIAS,N_V) &&\
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,N_H,&alpha_reset,GRAD_W_VH,N_V,&beta_reset,GRAD_W_VH,N_V,GRAD_W_VH,N_V) && \
           geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha_reset,GRAD_V_BIAS,N_V,&beta_reset,GRAD_V_BIAS,N_V,GRAD_V_BIAS,N_V)) {
                if(PRINT) PRINTMATRIX("SOFTMAX CELL AFTER UPDATE\n",_null,0,0,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("W_VH\n",W_VH,N_V,N_H,handle,dumpfile);
                if(PRINT) PRINTMATRIX("V_BIAS\n",V_BIAS,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if (geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,N_H,&alpha,D_W_VH,N_V,&beta,D_GRAD_W_VH,N_V,D_W_VH,N_V,handle) &&\
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,D_V_BIAS,N_V,&beta,D_GRAD_V_BIAS,N_V,D_V_BIAS,N_V,handle) &&\
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,N_H,&alpha_reset,D_GRAD_W_VH,N_V,&beta_reset,D_GRAD_W_VH,N_V,D_GRAD_W_VH,N_V) && \
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha_reset,D_GRAD_V_BIAS,N_V,&beta_reset,D_GRAD_V_BIAS,N_V,D_GRAD_V_BIAS,N_V)) {
                if(PRINT) PRINTMATRIX("SOFTMAX CELL AFTER UPDATE\n",_null,0,0,this->handle,this->dumpfile);
                if(PRINT) PRINTMATRIX("W_VH\n",D_W_VH,N_V,N_H,handle,dumpfile);
                if(PRINT) PRINTMATRIX("V_BIAS\n",D_V_BIAS,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    }
}

template<class T>
void SoftmaxCell<T>::propagateDeltaBack(T *delta)
{
    T alpha = 1.0;
    T beta = 0.0; //DELTA BEING RE-WRITTEN
    int* _null = NULL;
    if (handle == NULL) {
        if(gemv(CUBLAS_OP_T,N_V,N_H,&alpha,W_VH,N_V,V_SCORE,1,&beta,delta,1)) {

        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if (gemv(CUBLAS_OP_T,N_V,N_H,&alpha,D_W_VH,N_V,D_V_SCORE,1,&beta,delta,1,handle)) {

        } else {
            exit(EXIT_FAILURE);
        }

    }
    if(PRINT) PRINTMATRIX("SOFTMAX CELL PROPAGATE DELTA BACK\n",_null,0,0,this->handle,this->dumpfile);
    if(PRINT) PRINTMATRIX("DELTA BACK\n",delta,N_H,1,handle,dumpfile);
}

template<class T>
void SoftmaxCell<T>::activate(T *input)
{
    T alpha = 1.0;
    T beta_gemv = 0.0; //should be zero, Since V_SCORE has to be rewritten.
    T beta_geam = 1.0;
    int* _null = NULL;
    if(handle == NULL) {
        if (gemv(CUBLAS_OP_N,N_V,N_H,&alpha,W_VH,N_V,input,1,&beta_gemv,V_SCORE,1) && \
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,V_SCORE,N_V,&beta_geam,V_BIAS,N_V,V_SCORE,N_V) && \
            this->a_func->getActivation(V_SCORE,V_SCORE,N_V)) {
            if(PRINT) PRINTMATRIX("SOFTMAX CELL ACTIVATE\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("ACTIVATION\n",V_SCORE,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }
    } else {
        if (gemv(CUBLAS_OP_N,N_V,N_H,&alpha,D_W_VH,N_V,input,1,&beta_gemv,D_V_SCORE,1,handle) && \
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,D_V_SCORE,N_V,&beta_geam,D_V_BIAS,N_V,D_V_SCORE,N_V,handle) && \
            this->a_func->getActivation(D_V_SCORE,D_V_SCORE,N_V,handle)) {
            if(PRINT) PRINTMATRIX("SOFTMAX CELL ACTIVATE\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("ACTIVATION\n",D_V_SCORE,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }
    }
}

template<class T>
void SoftmaxCell<T>::classify(T *wIndex)
{
    int *_null = NULL;
    if (handle == NULL) {
        if(this->a_func->getClass(V_SCORE,wIndex,N_V)){
            if(PRINT) PRINTMATRIX("SOFTMAX CLASSIFY\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("CLASSIFY\n",wIndex,1,1,this->handle,this->dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if (this->a_func->getClass(D_V_SCORE,D_WINNER_INDEX,N_V,handle)) {
            if(PRINT) PRINTMATRIX("SOFTMAX CLASSIFY\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("CLASSIFY\n",D_WINNER_INDEX,1,1,this->handle,this->dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    }

}

template<class T>
void SoftmaxCell<T>::setError(size_t target_index)
{
    T alpha = 0.0;
    T beta = -1.0;
    int* _null = NULL;
    if(handle == NULL) {
        /**/
        if (geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,V_SCORE,N_V,&beta,V_SCORE,N_V,V_SCORE,N_V) &&\
            addToElementInVector(V_SCORE,target_index,1.0)) {
            if(PRINT) PRINTMATRIX("SOFTMAX SET ERROR\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("ERROR\n",V_SCORE,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if (geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,D_V_SCORE,N_V,&beta,D_V_SCORE,N_V,D_V_SCORE,N_V,handle) &&\
            add_to_element_in_vector(D_V_SCORE,target_index,1.0)) {
            if(PRINT) PRINTMATRIX("SOFTMAX SET ERROR\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("ERROR\n",D_V_SCORE,N_V,1,handle,dumpfile);

        } else {
            exit(EXIT_FAILURE);
        }
    }
}

template<class T>
void SoftmaxCell<T>::accumulateGradient(T *input)
{
    T alpha = 1.0;
    T beta = 1.0;
    int* _null = NULL;
    if(handle == NULL) {
        if (gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_V,N_H,1,&alpha,V_SCORE,N_V,input,N_H,&beta,GRAD_W_VH,N_V) && \
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,V_SCORE,N_V,&beta,GRAD_V_BIAS,N_V,GRAD_V_BIAS,N_V)) {
            if(PRINT) PRINTMATRIX("SOFTMAX CELL ACCUMULATE GRADIENT\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_W_VH\n",GRAD_W_VH,N_V,N_H,handle,dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_V_BAIS\n",GRAD_V_BIAS,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }

    } else {
        if (gemm(CUBLAS_OP_N,CUBLAS_OP_T,N_V,N_H,1,&alpha,D_V_SCORE,N_V,input,N_H,&beta,D_GRAD_W_VH,N_V,handle) && \
            geam(CUBLAS_OP_N,CUBLAS_OP_N,N_V,1,&alpha,D_V_SCORE,N_V,&beta,D_GRAD_V_BIAS,N_V,D_GRAD_V_BIAS,N_V)) {
            if(PRINT) PRINTMATRIX("SOFTMAX CELL ACCUMULATE GRADIENT\n",_null,0,0,this->handle,this->dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_W_VH\n",D_GRAD_W_VH,N_V,N_H,handle,dumpfile);
            if(PRINT) PRINTMATRIX("GRAD_V_BAIS\n",D_GRAD_V_BIAS,N_V,1,handle,dumpfile);
        } else {
            exit(EXIT_FAILURE);
        }
    }
}

template<class T>
void SoftmaxCell<T>::backwardPass(T target_label, T* input, T* delta_back)
{
    setError(target_label);
    accumulateGradient(input);
    propagateDeltaBack(delta_back);

}

template<class T>
void SoftmaxCell<T>::print(std::ostream &out)
{
    PRINTMATRIX("W_VH\n",W_VH,N_V,N_H,handle,out);
    PRINTMATRIX("V_BIAS\n",V_BIAS,N_V,1,handle,out);
}

template<class T>
void SoftmaxCell<T>::printwvh(std::ostream &out)
{
    PRINTMATRIX("",W_VH,N_V,N_H,handle,out);
}

template<class T>
void SoftmaxCell<T>::printvbias(std::ostream &out)
{
    PRINTMATRIX("",V_BIAS,N_V,1,handle,out);
}

#endif // SOFTMAXCELL_CPP
