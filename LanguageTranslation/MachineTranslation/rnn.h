#ifndef RNN_H
#define RNN_H

#include "cublas_v2.h"
#include <vector>
#include <unordered_map>
#include "rnncell.h"
#include "softmaxcell.h"
#include "memorypool.h"



template<class T>
class RNN
{
    RNNCell<T> *cell;
    SoftmaxCell<T> *soft_max;

    T* W_XS;
    T* W_XD;

    size_t N_X;//WORD-VEC SIZE
    size_t N_S;//SOURCE-VOCAB-SIZE
    size_t N_D;//DEST-VOCAB-SIZE
    size_t N_H;//HIDDEN-LAYER-SIZE

    MATRIX_TYPE type;
    cublasHandle_t handle;
    std::vector<bucket> bucket_list;
    //std::vector<std::string> source_vocab;
    std::vector<std::string> dest_vocab;
    std::unordered_map<std::string,int> source_hash;

    MemoryPool *pool;
    MemoryPool *gpupool;
    std::fstream &dumpfile;



public:

    RNN(std::fstream &dumpfile, MATRIX_TYPE type = COLUMN_MAJOR);

    void setSourceVocabSize(size_t sz);
    void setDestVocabSize(size_t sz);
    void setWordVecSize(size_t sz);
    void setHiddenLayerSize(size_t sz);

    bool setWXS(T* W_XS, size_t sz, MATRIX_TYPE type);
    bool setWXD(T* W_XD, size_t sz, MATRIX_TYPE type);

    void setRNNCell(RNNCell<T> *cell);
    void setSoftMaxCell(SoftmaxCell<T> *soft_max);

    bool createCublasHandle();
    cublasHandle_t getCublasHandle();

    void setCPUPOOL(MemoryPool *pool);
    void setGPUPOOL(MemoryPool *pool);

    size_t trainSpaceRequiredForOneDataPoint(size_t l1, size_t l2);

    //bool createBucket(const char* s_buckets, const char* d_buckets);
    bool createBucket(const char* bucket_info);
    bool setWordVecMatrices(const char *s_word_vec, const char *d_word_vec);
    bool setDataMatrices(const char *s_sen, const char *d_sen, std::streampos& pos1, std::streampos& pos2,size_t index_in_bucket_list, T* W1, T* W2);

    bool _train(T *W_1, size_t b_1, T *W_2, size_t b_2, size_t n_data_points);
    bool train(const char* s_sen, const char* s_word_vec, size_t s_vocab_size, \
               const char* d_sen, const char* d_word_vec, size_t d_vocab_size, const char* bucket_info, \
               size_t word_vec_size, size_t hidden_layer_size, size_t epoch);
    bool _predict(std::fstream &sfile,std::fstream &dfile);
    bool predict(const char *source,const char* s_word_vec, size_t s_vocab_size, \
                 const char* d_word_vec, size_t d_vocab_size,size_t word_vec_size, size_t hidden_layer_size,\
                 const char* output);

    bool stepCell(T* data_in, T* hidden_state_buffer, T* derivative_buffer, T* delta_buffer, int current_index, int target_label);
    bool stepBack(int index_data_point, T* data, int bucket_size, T* hidden_state_buffer, T* derivative_buffer, T* delta_buffer, int current_index, int n_back_steps);
    bool updateCell();
};

#include "rnn.cpp"

#endif // RNN_H
