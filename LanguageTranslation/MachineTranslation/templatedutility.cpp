

template <typename T>
void setContiguousMatrix(T* source, T* dest, size_t row, size_t col, MATRIX_TYPE s_t, MATRIX_TYPE d_t) {
    if(source && dest && row>0 && col>0) {
        if (s_t == d_t) {
            for (size_t s = 0 ; s < row*col ; s++) {
                dest[s] = source[s];
            }
        } else if (s_t == ROW_MAJOR && d_t == COLUMN_MAJOR) {
            for( size_t i = 0; i < row; i++) {
                for(size_t j = 0; j < col; j++) {
                    dest[IDX2C(i,j,row)] = source[j+i*col];
                }
            }
        } else if (s_t == COLUMN_MAJOR && d_t == ROW_MAJOR) {
            size_t index = 0;
            for( size_t i = 0; i < row; i++) {
                for(size_t j = 0; j < col; j++) {
                    dest[index++] = source[IDX2C(i,j,row)];
                }
            }
        }
    } else{
        printf("Either Matrix is NULL or ROW,COL value is zero in setContiguousMatrix method\n");
    }
}

template <typename T>
void printContiguousMatrix(T* mat, size_t row, size_t col, MATRIX_TYPE type, std::ostream &out) {
    if(mat && row>0 && col>0) {
        if (type == ROW_MAJOR) {
            for( size_t i = 0; i < row; i++) {
                for(size_t j = 0; j < col; j++) {
                    out<<mat[j+i*col]<<" ";
                }
                out<<std::endl;
            }
        } else if(type == COLUMN_MAJOR) {
            for( size_t i = 0; i < row; i++) {
                for(size_t j = 0; j < col; j++) {
                    out << mat[IDX2C(i,j,row)] << " ";
                }
                out<<std::endl;
            }
        }
    }
}

template <typename T>
bool multMatrixVector(bool trans,size_t m,size_t n, const T* alpha, const T* A, const T* x, const T* beta, T* y, MATRIX_TYPE type) {
    if(alpha && A && x && beta && y) {
        if(type == COLUMN_MAJOR) {
            if(trans) {
                size_t idx = 0;
                T accum[n] = {0.0};
                for(size_t i = 0 ;i < n; i++) {

                    for( size_t j = 0; j<m; j++) {
                        accum[i] += (*alpha)*A[idx++]*x[j];
                    }
                }
                for (size_t i = 0; i < n; i++) {
                    y[i] = (*beta)*y[i] + accum[i];
                }
            } else {
                size_t idx = 0;
                T accum[m] = {0.0};
                for(size_t i = 0 ;i < n; i++) {
                    for( size_t j = 0; j<m; j++) {
                        T a = A[idx++];
                        T b = x[i];
                        accum[j] += (*alpha)*a*b;
                    }
                }
                for(size_t j = 0; j<m; j++) {

                    T c = (*beta)*y[j];
                    T d = accum[j];
                    y[j] = c+d;
                }
            }
            return true;
        } else {
            // TO Implement
            return false;
        }
    } else {
        printf("Some pointers are NULL in multMatrixVector method\n");
        return false;
    }

}

template <typename T>
bool multMatrixMatrix(bool transa, bool transb, size_t m, size_t n, size_t k, const T* alpha, const T* A, const T* B,const T* beta, T* C, MATRIX_TYPE type){
    if(alpha && A && B && beta && C) {
        if(type == COLUMN_MAJOR) {
            if(transa == false && transb == true && k == 1) {
                size_t idx = 0;
                for(size_t i = 0 ; i < n; i++) {
                    for(size_t j= 0; j<m; j++) {
                        C[idx] = (*alpha)*A[j]*B[i] + (*beta)*C[idx];
                        idx++;
                    }
                }
                return true;
            } else {
                //TO IMPLEMENT
                return false;
            }

        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        printf("Some pointers are NULL in multMatrixMatrix method\n");
        return false;
    }

}

template <typename T>
bool addMatrixMatrix(bool transa, bool transb, size_t m, size_t n, const T* alpha, const T* A, const T* beta, const T* B, T* C, MATRIX_TYPE type){
    if(alpha && A && B && beta && C) {
        if(type == COLUMN_MAJOR) {
            if(transa == false && transb == false) {
                size_t idx = 0;
                for( ; idx < m*n; idx++) {
                        C[idx] = (*alpha)*A[idx] + (*beta)*B[idx];
                }
                return true;
            } else {
                //TO IMPLEMENT
                return false;
            }

        } else {
            // TO IMPLEMENT
            return false;
        }
    } else {
        printf("Some pointers are NULL in addMatrixMatrix method\n");
        return false;
    }
}


template <typename T>
bool setElementInVector(T *x, size_t index, T value)
{
    if(x) {
        x[index] = value;
        return true;
    } else {
        printf("x is NULL in setElementInVector\n");
        return false;
    }


}

template <typename T>
bool multVectorElementByElement(T *x, T *y, T* result, size_t sz)
{
    if(x && y && result) {
        for(size_t i = 0 ; i<sz ; i++) {
            result[i] = x[i]*y[i];
        }
        return true;
    } else {
        printf("x or y or result is NULL in multVectorElementByElement\n");
        return false;
    }


}

template <typename T>
bool addToElementInVector(T *x, size_t index, T value)
{
    if(x) {
        x[index] = x[index]+value;
        return true;
    } else {
        printf("x is NULL in addToElementInVector\n");
        return false;
    }
}


template <typename T>
bool READMATRIX(T *mat, size_t row, size_t col, std::istream &file, MATRIX_TYPE dest_type)
{
    if(file.good()) {
        std::string line;
        std::string word;
        std::stringstream ss;
        size_t i = 0;
        while(std::getline(file,line) && i < row) {
            ss.clear();
            ss.str(line);
            for(size_t j = 0; j<col; j++) {
                ss >> word;
                if(dest_type == COLUMN_MAJOR) {
                    mat[IDX2C(i,j,row)] = atof(word.c_str());
                } else {
                    mat[i*col+j] = atof(word.c_str());
                }
            }
            i++;
        }
    } else {
        printf("Could not read from stream in READMATRIX method\n");
    }
}
