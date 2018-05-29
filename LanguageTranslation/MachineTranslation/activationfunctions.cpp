#ifndef ACTIVATIONFUNCTIONS_CPP
#define ACTIVATIONFUNCTIONS_CPP

#include "activationfunctions.h"


template <class T>
bool ActivationFunctions<T>::getActivation(T *sum, T *result, size_t sz, cublasHandle_t handle)
{
    return false;
}

template <class T>
bool ActivationFunctions<T>::getClass(T* activation, T* result, size_t sz, cublasHandle_t handle)
{
    return false;
}

template <class T>
bool ActivationFunctions<T>::getDerivative(T *activation, T *result, size_t sz, cublasHandle_t handle)
{
    return false;
}

template <class T>
bool SigmoidFunction<T>::getActivation(T *sum, T* result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            T pw = pow(exp(1.0),-sum[i]);
            if(isnan(pw)) {
                T s = -sum[i];
                printf("pow returns nan for %f\n",s);

            }
            T tmp = 1/(1+pw);
            if(isnan(tmp)) {
                printf("");
            }
            result[i] = tmp;
        }
        return true;
    } else {

    }
    return false;
}

template <class T>
bool SigmoidFunction<T>::getClass(T* activation, T *result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            result[i] = activation[i]>0.5?1:0;
        }
        return true;
    } else {

    }
    return false;
}

template <class T>
bool SigmoidFunction<T>::getDerivative(T *activation, T* result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            result[i] = activation[i]*(1.0-activation[i]);
        }
        return true;
    } else {

    }
    return false;

}

template <class T>
bool HyperbolicFunction<T>::getActivation(T* sum , T* result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            T p = pow(exp(1.0),sum[i]);
            T n = pow(exp(1.0),-sum[i]);
            result[i] = (p-n)/(p+n);
        }
        return true;
    } else {

    }
    return false;

}

template <class T>
bool HyperbolicFunction<T>::getClass(T *activation, T* result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            result[i] = activation[i]>0.0?1:-1;
        }
        return true;
    } else {

    }
    return false;

}

template <class T>
bool HyperbolicFunction<T>::getDerivative(T* activation, T *result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        for(size_t i = 0; i < sz; i++) {
            result[i] = 1.0 - (activation[i]*activation[i]);
        }
        return true;
    } else {

    }
    return false;

}

template <class T>
bool SoftmaxFunction<T>::getActivation(T *sum, T* result, size_t sz, cublasHandle_t handle)
{
    if(handle == NULL) {
        T normalization_factor = 0.0;
        for(size_t i = 0; i < sz; i++) {
            result[i] = exp(sum[i]);
            normalization_factor += result[i];
        }
        for(size_t i = 0; i < sz; i++) {
            result[i] = result[i]/normalization_factor;
        }
        return true;
    } else {

    }
    return false;
}

template <class T>
bool SoftmaxFunction<T>::getClass(T *activation, T* result, size_t sz, cublasHandle_t handle)
{
    // SUPPORTING SINGLE BEST ANSWER AT THIS TIME
    T max = 0.0;
    int r = 0;
    if(handle == NULL) {
        size_t i = 0;
        for(; i < sz; i++) {
            if (activation[i]>max) {
                max = activation[i];
                r = i;
            }

        }
        result[0] = r;
        return true;
    } else {

    }
    return false;
}

template <class T>
bool SoftmaxFunction<T>::getDerivative(T *activation, T* result, size_t sz, cublasHandle_t handle)
{
    //DO NOT USE THIS FUNCTION FOR SOFTMAX
    return false;
}

#endif // ACTIVATIONFUNCTIONS_CPP


