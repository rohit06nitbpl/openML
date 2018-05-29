#ifndef TEMPLATEDUTILITY
#define TEMPLATEDUTILITY

#include "enums.h"
#include "cublas_v2.h"
#include <fstream>
#include <sstream>
#include <time.h>
#include <iostream>

template <typename T>
void setContiguousMatrix(T* source, T* dest, size_t row, size_t col, MATRIX_TYPE s_t, MATRIX_TYPE d_t);

template <typename T>
void printContiguousMatrix(T* mat, size_t row, size_t col, MATRIX_TYPE type, std::ostream &out = std::cout);

template <typename T>
bool multMatrixVector(bool trans,size_t m,size_t n, const T* alpha, const T* A, const T* x, const T* beta, T* y, MATRIX_TYPE type);

template <typename T>
bool multMatrixMatrix(bool transa, bool transb, size_t m, size_t n, size_t k, const T* alpha, const T* A, const T* B,const T* beta, T* C, MATRIX_TYPE type);

template <typename T>
bool addMatrixMatrix(bool transa, bool transb, size_t m, size_t n, const T* alpha, const T* A, const T* beta, const T* B, T* C, MATRIX_TYPE type);

template <typename T>
bool setElementInVector(T* x, size_t index, T value);

template <typename T>
bool addToElementInVector(T* x, size_t index, T value);

template <typename T>
bool multVectorElementByElement(T* x, T* y, T* result, size_t sz);

//template <typename T>
//void PRINTMATRIX(const char* s, T* mat, size_t row, size_t col, cublasHandle_t handle,std::ostream &file = std::cout, MATRIX_TYPE type = COLUMN_MAJOR);

template <typename T>
bool READMATRIX(T* mat, size_t row, size_t col,std::istream &file = std::cin,MATRIX_TYPE dest_type = COLUMN_MAJOR);

#include "templatedutility.cpp"

#endif // TEMPLATEDUTILITY

