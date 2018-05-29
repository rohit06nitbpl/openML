#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include "utility.h"
#include "cublas_v2.h"


template <class T>
class ActivationFunctions
{
public:
public:
    virtual bool getActivation(T* sum, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getClass(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getDerivative(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
};

template <class T>
class SigmoidFunction : public ActivationFunctions<T>
{
    virtual bool getActivation(T *sum, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getClass(T *activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getDerivative(T *activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
};

template <class T>
class HyperbolicFunction : public ActivationFunctions<T>
{
    virtual bool getActivation(T* sum, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getClass(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getDerivative(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
};

template <class T>
class SoftmaxFunction : public ActivationFunctions<T>
{
    virtual bool getActivation(T* sum, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getClass(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
    virtual bool getDerivative(T* activation, T* result, size_t sz, cublasHandle_t  handle = NULL);
};

#include "activationfunctions.cpp"

#endif // ACTIVATIONFUNCTIONS_H
