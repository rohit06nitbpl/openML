#ifndef RNN_CPP
#define RNN_CPP

#include "rnn.h"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#define START_INDEX_IN_FILE 0
#define DEST_END 28

void write_test(RNNCell<double> *cell, SoftmaxCell<double> *soft_max)
{
    std::fstream file;
    file.open("./test/prev/w_vh.txt",std::ios::out | std::ios::app);
    if(file.is_open()) {
       soft_max->printwvh(file);
       file.close();
    } else{
        printf("can not write ./test/prev/w_vh.txt\n");
    }

    file.open("./test/prev/v_bias_v.txt",std::ios::out | std::ios::app);
    if(file.is_open()) {
        soft_max->printvbias(file);
       file.close();
    } else{
        printf("can not write ./test/prev/v_bias_v.txt\n");
    }

    file.open("./test/prev/w_hh.txt",std::ios::out | std::ios::app);
    if(file.is_open()) {
       cell->printwhh(file);
       file.close();
    } else{
        printf("can not write ./test/prev/w_hh.txt\n");
    }

    file.open("./test/prev/w_hx.txt",std::ios::out | std::ios::app);
    if(file.is_open()) {
       cell->printwhx(file);
       file.close();
    } else{
        printf("can not write ./test/prev/w_hx.txt\n");
    }

    file.open("./test/prev/v_bias_h.txt",std::ios::out | std::ios::app);
    if(file.is_open()) {
        cell->printvbias(file);
       file.close();
    } else{
        printf("can not write ./test/prev/v_bias_h.txt\n");
    }
}


template<class T>
size_t RNN<T>::trainSpaceRequiredForOneDataPoint(size_t l1, size_t l2)
{
    size_t S_HIN = sizeof(T)*(l1+l2+1)*N_H;
    size_t S_DOUT = sizeof(T)*(l1+l2)*N_H;
    size_t S_DELTA = sizeof(T)*N_H;
    size_t S_DATA_FIRST_LANG = sizeof(T)*N_X*l1;
    size_t S_DATA_SECOND_LANG = sizeof(T)*N_X*l2;
    return S_HIN+S_DOUT+S_DELTA+S_DATA_FIRST_LANG+S_DATA_SECOND_LANG;
}

template<class T>
bool RNN<T>::createBucket(const char *bucket_info)
{
    size_t max_mem_aval_for_data_proccessing = 65536000; //TODO //64MB

    std::fstream f_bucket_info;

    std::string s_line;
    std::stringstream ss;

    f_bucket_info.open(bucket_info,std::ios::in);

    if(f_bucket_info.is_open()) {
        std::string n_line,b_1,b_2;
        size_t nline = 0, b1 = 0, b2 = 0, mem_est = 0, one_dp_est = 0;
        while(std::getline(f_bucket_info,s_line)) {
            ss.clear();
            ss.str(s_line);
            ss >> n_line; ss >> b_1; ss >> b_2;
            nline = atol(n_line.c_str());
            b1 = atol(b_1.c_str());
            b2 = atol(b_2.c_str());

            one_dp_est = trainSpaceRequiredForOneDataPoint(b1,b2);
            mem_est = nline*one_dp_est;
            do {

                if(mem_est > max_mem_aval_for_data_proccessing) {
                    size_t diff = mem_est - max_mem_aval_for_data_proccessing;
                    size_t num_line_to_exclude = ceil(diff,one_dp_est);
                    size_t new_line = nline - num_line_to_exclude;
                    if(new_line > 0) {
                        bucket b;
                        b.data_points = new_line;
                        b.first_sen_len = b1;
                        b.second_sen_len = b2;
                        bucket_list.push_back(b);
                        mem_est -= new_line*one_dp_est;
                        nline -= new_line;
                    } else {
                        printf("Do not have enough memmory for even one sentence training.\n");
                        return false;
                    }
                } else {
                    if(nline>0) {
                        bucket b;
                        b.data_points = nline;
                        b.first_sen_len = b1;
                        b.second_sen_len = b2;
                        bucket_list.push_back(b);
                    }
                    break;

                }
            } while(1);
        }



    } else {
        printf("Cannot open bucket info files.\n");
        return false;
    }

    f_bucket_info.close();
    return true;

}

template<class T>
RNN<T>::RNN(std::fstream &dumpfile, MATRIX_TYPE type):type(type),W_XD(NULL),W_XS(NULL),N_S(0),N_D(0),N_H(0),N_X(0),\
                              cell(NULL),soft_max(NULL),handle(NULL),pool(NULL),gpupool(NULL),\
                              dumpfile(dumpfile)
{

}

template<class T>
void RNN<T>::setSourceVocabSize(size_t sz)
{
    this->N_S = sz;
}

template<class T>
void RNN<T>::setDestVocabSize(size_t sz)
{
    this->N_D = sz;
}

template<class T>
void RNN<T>::setWordVecSize(size_t sz)
{
    this->N_X = sz;
}

template<class T>
void RNN<T>::setHiddenLayerSize(size_t sz)
{
    this->N_H = sz;
}

template<class T>
bool RNN<T>::setWXS(T *W_XS, size_t sz, MATRIX_TYPE type)
{
    if(handle == NULL) {
        if(W_XS != NULL && N_X != 0 && N_S != 0 && sz == N_X*N_S) {
            this->W_XS = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->W_XS != NULL) {
                setContiguousMatrix(W_XS,this->W_XS,N_X,N_S,type,this->type);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(W_XS != NULL && N_X != 0 && N_S != 0 && sz == N_X*N_S && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&(this->W_XS),sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),W_XS,1,this->W_XS,1));
            return true;
        } else {
            return false;
        }
    }
}

template<class T>
bool RNN<T>::setWXD(T *W_XD, size_t sz, MATRIX_TYPE type)
{
    if(handle == NULL) {
        if(W_XD != NULL && N_X != 0 && N_D != 0 && sz == N_X*N_D) {
            this->W_XD = (T*)this->pool->allocateMemory(sizeof(T)*sz);
            if (this->W_XD != NULL) {
                setContiguousMatrix(W_XD,this->W_XD,N_X,N_D,type,this->type);
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        if(W_XD != NULL && N_X != 0 && N_D != 0 && sz == N_X*N_D && type == COLUMN_MAJOR) {
            CUDA_SAFE_CALL(cudaMalloc((void **)&(this->W_XD),sizeof(T)*sz));
            CUBLAS_SAFE_CALL(cublasSetVector(sz,sizeof(T),W_XD,1,this->W_XD,1));
            return true;
        } else {
            return false;
        }
    }
}

template<class T>
void RNN<T>::setRNNCell(RNNCell<T> *cell)
{
    this->cell = cell;
}

template<class T>
void RNN<T>::setSoftMaxCell(SoftmaxCell<T> *soft_max)
{
    this->soft_max = soft_max;
}

template<class T>
bool RNN<T>::createCublasHandle()
{
    CUBLAS_SAFE_CALL(cublasCreate(&handle));
    return true;
}

template<class T>
cublasHandle_t RNN<T>::getCublasHandle()
{
    return this->handle;
}

template<class T>
void RNN<T>::setCPUPOOL(MemoryPool *pool)
{
    this->pool = pool;
}

template <class T>
void RNN<T>::setGPUPOOL(MemoryPool *pool)
{
    this->gpupool = pool;
}



template<class T>
bool RNN<T>::setWordVecMatrices(const char *s_word_vec, const char *d_word_vec)
{
    std::fstream f_s_word_vec;
    std::fstream f_d_word_vec;

    std::string s_line;
    std::string d_line;
    std::string word;
    size_t counter = 0;

    T* W1 = NULL;
    T* W2 = NULL;

    f_s_word_vec.open(s_word_vec,std::ios::in);
    f_d_word_vec.open(d_word_vec,std::ios::in);

    if(f_s_word_vec.is_open() && f_d_word_vec.is_open()) {
        std::stringstream ss;
        W1 = (T*)this->pool->allocateMemory(sizeof(T)*N_X*N_S);
        W2 = (T*)this->pool->allocateMemory(sizeof(T)*N_X*N_D);
        if(W1 && W2) {
            while(std::getline(f_s_word_vec,s_line)) {

                if(s_line.size()>0) {
                    if(counter <= N_S) {
                        ss.clear();
                        ss.str(s_line);
                        std::getline(ss,word,' ');
                        source_hash[word] = counter;
                        for(size_t i = 0; i < N_X; i++) {
                            std::getline(ss,word,' ');
                            W1[counter*N_X+i] = atof(word.c_str());
                        }
                    } else {
                        printf("source word vector file is larger than expected.\n");
                        return false;
                    }
                }
                counter++;
            }
            counter = 0;
            while(std::getline(f_d_word_vec,d_line)) {
                if(d_line.size()>0) {
                    if(counter <= N_D) {
                        ss.clear();
                        ss.str(d_line);
                        std::getline(ss,word,' ');
                        dest_vocab.push_back(word);
                        for(size_t i = 0; i < N_X; i++) {
                            std::getline(ss,word,' ');
                            W2[counter*N_X+i] = atof(word.c_str());
                        }
                    } else {
                        printf("destination word vector file is larger than expected.\n");
                        return false;
                    }
                }
                counter++;
            }
        } else {
            printf("Unable to allocate memory for W1 and W2 in setWordVecMatrices method in RNN\n");

        }
    } else {
        printf("Unable to open word vector files.\n");
        return false;
    }

    f_s_word_vec.close();
    f_d_word_vec.close();

    //assert((int)N_S == source_vocab.size());
    //assert((int)N_D == dest_vocab.size());

    if(handle == NULL) {
        W_XS = W1;
        W_XD = W2;
    } else {
        CUDA_SAFE_CALL(cudaMalloc((void **)&W_XS,sizeof(T)*N_X*N_S));
        CUDA_SAFE_CALL(cudaMalloc((void **)&W_XD,sizeof(T)*N_X*N_D));
        CUBLAS_SAFE_CALL(cublasSetVector(N_X*N_S,sizeof(T),W1,1,W_XS,1));
        CUBLAS_SAFE_CALL(cublasSetVector(N_X*N_D,sizeof(T),W2,1,W_XD,1));
        this->pool->freeMemory(W1,sizeof(T)*N_X*N_S);
        this->pool->freeMemory(W2,sizeof(T)*N_X*N_D);
    }

    return true;
}

template<class T>
bool RNN<T>::setDataMatrices(const char *s_sen, const char *d_sen, std::streampos& pos1, std::streampos& pos2, size_t index_in_bucket_list, T* W1, T* W2)
{
    std::fstream f_s_sen;
    std::fstream f_d_sen;

    std::string s_line;
    std::string d_line;
    std::string word;

    f_s_sen.open(s_sen,std::ios::in);
    f_d_sen.open(d_sen,std::ios::in);


    if(f_s_sen.is_open() && f_d_sen.is_open()) {
        std::stringstream ss;
        f_s_sen.seekg(pos1);
        f_d_sen.seekg(pos2);
        size_t sz = bucket_list.at(index_in_bucket_list).data_points;
        for(size_t n = 0; n<sz; n++) {
            std::getline(f_s_sen,s_line);
            ss.clear();
            ss.str(s_line);
            for(size_t i = 0; i<bucket_list.at(index_in_bucket_list).first_sen_len; i++) {
                if(std::getline(ss,word,' ')) {
                    W1[n*bucket_list.at(index_in_bucket_list).first_sen_len + i] = atol(word.c_str());
                } else {
                    W1[n*bucket_list.at(index_in_bucket_list).first_sen_len + i] = -1;
                }

            }
            std::getline(f_d_sen,d_line);
            ss.clear();
            ss.str(d_line);
            for(size_t i = 0; i<bucket_list.at(index_in_bucket_list).second_sen_len; i++) {
                if(std::getline(ss,word,' ')) {
                    W2[n*bucket_list.at(index_in_bucket_list).second_sen_len + i] = atol(word.c_str());
                } else {
                    W2[n*bucket_list.at(index_in_bucket_list).second_sen_len + i] = -1;
                }
            }

        }
        pos1 = f_s_sen.tellg();
        pos2 = f_d_sen.tellg();
    } else {
        printf("Unable to open data files.\n");
        return false;
    }
    f_s_sen.close();
    f_d_sen.close();
    return true;

}

template<class T>
bool RNN<T>::stepCell(T* data_in, T* hidden_state_buffer, T* derivative_buffer, T* delta_buffer, int current_index, int target_label){
    int next_index = current_index+1;
    cell->stepForward(&hidden_state_buffer[current_index*N_H],data_in,&hidden_state_buffer[next_index*N_H],&derivative_buffer[current_index*N_H]);
    soft_max->activate(&hidden_state_buffer[(next_index)*N_H]);
    soft_max->backwardPass(target_label,&hidden_state_buffer[(next_index)*N_H],delta_buffer);
    return true;
}

template<class T>
bool RNN<T>::stepBack(int index_data_point, T* data, int bucket_size, T* hidden_state_buffer, T* derivative_buffer, T* delta_buffer, int current_index, int n_back_steps){
    if(PRINT_LIMITED) PRINTMATRIX("\nBack Prop ",&index_data_point,1,1,NULL,this->dumpfile);
    for(int back_index = current_index; (back_index >= 0 && back_index >(current_index-n_back_steps)) ; back_index--) {
        int label = data[index_data_point*bucket_size+back_index]-START_INDEX_IN_FILE;//TODO
        if(PRINT_LIMITED) PRINTMATRIX("\nWord ",&label,1,1,NULL,this->dumpfile);
        cell->accumulateGradient(&hidden_state_buffer[back_index*N_H],&W_XS[N_X*label],&derivative_buffer[back_index*N_H],delta_buffer);
        cell->propagateDeltaBack(delta_buffer);
    }
    return true;
}

template<class T>
bool RNN<T>::updateCell(){
    cell->updateCell();
    soft_max->updateCell();
    return true;
}

template<class T>
bool RNN<T>::_train(T *encoder_data, size_t encoder_bucket_size, T *decoder_data, size_t decoder_bucket_size, size_t n_data_points)
{
    T* hidden_state_buffer;
    T* derivative_buffer;
    T* delta_buffer;

    size_t n_back_steps = 4;
    size_t index_encoder_sen_end;
    size_t encoder_sen_end_label = source_hash["SEN-END"]; //TODO //11
    //size_t decoder_sen_end_label = DEST_END; //TODO //9

    if(handle == NULL) {
        hidden_state_buffer  = (T*)this->pool->allocateMemory(sizeof(T)*(encoder_bucket_size+decoder_bucket_size+1)*N_H);
        derivative_buffer = (T*)this->pool->allocateMemory(sizeof(T)*(encoder_bucket_size+decoder_bucket_size)*N_H);
        delta_buffer = (T*)this->pool->allocateMemory(sizeof(T)*N_H);
        if(hidden_state_buffer && derivative_buffer && delta_buffer) {
            memset(hidden_state_buffer,0,sizeof(T)*(encoder_bucket_size+decoder_bucket_size)*N_H);
            memset(derivative_buffer,0,sizeof(T)*(encoder_bucket_size+decoder_bucket_size)*N_H);
            memset(delta_buffer,0,sizeof(T)*N_H);
            // NEED TO INITIALIZE TO ZERO EVEN IF THEY ARE RE-WRITTEN IN FUNCTION CALL

        } else {
            return false;
        }
    } else {
        CUDA_SAFE_CALL(cudaMalloc((void **)&hidden_state_buffer,sizeof(T)*(encoder_bucket_size+decoder_bucket_size+1)*N_H));
        CUDA_SAFE_CALL(cudaMalloc((void **)&derivative_buffer,sizeof(T)*(encoder_bucket_size+decoder_bucket_size)*N_H));
        CUDA_SAFE_CALL(cudaMalloc((void **)&delta_buffer,sizeof(T)*N_H));
        // NO NEED TO INITIALIZE TO ZERO SINCE THEY ARE RE-WRITTEN IN FUNCTION CALL
    }

    for(int index_data_point = 0; index_data_point<n_data_points; index_data_point++) {

        int current_index = 0;
        int decoder_sen_start_label = -1;

        //ENCODER
        for(;current_index<encoder_bucket_size; current_index++) {
            int next_index = current_index+1;
            int current_label = encoder_data[index_data_point*encoder_bucket_size+current_index]-START_INDEX_IN_FILE;

            if(PRINT_LIMITED) PRINTMATRIX("\nSentence Number ",&index_data_point,1,1,NULL,this->dumpfile);
            if(PRINT_LIMITED) PRINTMATRIX("\nWord ",&current_label,1,1,NULL,this->dumpfile);

            if (current_label >= 0 && current_label != encoder_sen_end_label) {
                int target_label = encoder_data[index_data_point*encoder_bucket_size+(next_index)]-START_INDEX_IN_FILE;
                if (target_label != -1) {
                    stepCell(&W_XS[N_X*current_label],hidden_state_buffer,derivative_buffer,delta_buffer,current_index,target_label);
                    stepBack(index_data_point,encoder_data,encoder_bucket_size,hidden_state_buffer,derivative_buffer,delta_buffer,current_index,n_back_steps);
                    updateCell();
                }

            } else if (current_label >= 0 && current_label == encoder_sen_end_label) {
                decoder_sen_start_label = decoder_data[index_data_point*decoder_bucket_size]-START_INDEX_IN_FILE;
                if (decoder_sen_start_label >= 0) {
                    stepCell(&W_XS[N_X*current_label],hidden_state_buffer,derivative_buffer,delta_buffer,current_index,decoder_sen_start_label);
                    stepBack(index_data_point,encoder_data,encoder_bucket_size,hidden_state_buffer,derivative_buffer,delta_buffer,current_index,n_back_steps);
                    updateCell();
                }
                break;
            } else {
                break; //return false;
            }
        }
        index_encoder_sen_end = current_index;
        current_index++;

        //DECODER
        /*
        for(int decoder_index = 0; decoder_index<decoder_bucket_size; decoder_index++) {
            int decoder_next_index = decoder_index + 1;
            int current_label = decoder_data[index_data_point*decoder_bucket_size+decoder_index]-START_INDEX_IN_FILE;

            if(PRINT_LIMITED) PRINTMATRIX("\nDecode: Sentence Number ",&index_data_point,1,1,NULL,this->dumpfile);
            if(PRINT_LIMITED) PRINTMATRIX("\nWord ",&current_label,1,1,NULL,this->dumpfile);

            if (current_label >= 0) {
                int target_label = decoder_data[index_data_point*decoder_bucket_size+(decoder_next_index)]-START_INDEX_IN_FILE;
                if (target_label) {
                    stepCell(&W_XD[N_X*current_label],hidden_state_buffer,derivative_buffer,delta_buffer,current_index,target_label);
                    int n_back_step_decoder = decoder_index+1;
                    if (n_back_step_decoder > n_back_steps) {
                        n_back_step_decoder = n_back_steps;
                    }
                    stepBack(index_data_point,decoder_data,decoder_bucket_size,hidden_state_buffer,derivative_buffer,delta_buffer,decoder_index,n_back_step_decoder);
                    int n_back_step_encoder = n_back_steps - n_back_step_decoder;
                    if (n_back_step_encoder > 0) {
                        stepBack(index_data_point,encoder_data,encoder_bucket_size,hidden_state_buffer,derivative_buffer,delta_buffer,current_index-n_back_step_decoder,n_back_step_encoder);
                    }
                    updateCell();
                }
            } else {
                break; //return false;
            }
            current_index++;
        }*/
        if(PRINT_LIMITED) PRINTMATRIX("\nDone: Sentence Number ",&index_data_point,1,1,NULL,this->dumpfile);
    }

    if(handle == NULL) {
        this->pool->freeMemory(hidden_state_buffer,sizeof(T)*(encoder_bucket_size+decoder_bucket_size+1)*N_H);
        this->pool->freeMemory(derivative_buffer,sizeof(T)*(encoder_bucket_size+decoder_bucket_size)*N_H);
        this->pool->freeMemory(delta_buffer,sizeof(T)*N_H);
    } else {
        CUDA_SAFE_CALL(cudaFree(hidden_state_buffer));
        CUDA_SAFE_CALL(cudaFree(derivative_buffer));
        CUDA_SAFE_CALL(cudaFree(delta_buffer));
    }
    return true;
}

template<class T>
bool RNN<T>::train(const char *s_sen, const char *s_word_vec, size_t s_vocab_size, \
                   const char *d_sen, const char *d_word_vec, size_t d_vocab_size, const char *bucket_info, \
                   size_t word_vec_size, size_t hidden_layer_size, size_t epoch)
{

    T* W_1 = NULL;
    T* W_2 = NULL;

    T* W1 = NULL;
    T* W2 = NULL;

    if(s_sen && s_word_vec && d_sen && d_word_vec && bucket_info) {
        this->setSourceVocabSize(s_vocab_size);
        this->setDestVocabSize(d_vocab_size);
        this->setWordVecSize(word_vec_size);
        this->setHiddenLayerSize(hidden_layer_size);
        if(createBucket(bucket_info) && \
           setWordVecMatrices(s_word_vec,d_word_vec)) {
           if(PRINT) PRINTMATRIX("Source Word Vector\n",W_XS,N_X,N_S,this->handle,this->dumpfile);
           if(PRINT) PRINTMATRIX("\nDest Word Vector\n",W_XD,N_X,N_D,this->handle,this->dumpfile);
           if(PRINT) {
               size_t sz = bucket_list.size();
               int temp[3*sz];
               for(size_t i = 0;i<sz;i++) {
                   temp[i*3+0] = bucket_list.at(i).data_points;
                   temp[i*3+1] = bucket_list.at(i).first_sen_len;
                   temp[i*3+2] = bucket_list.at(i).second_sen_len;
               }
               PRINTMATRIX("Buckets\n",temp,sz,3,this->handle,this->dumpfile,ROW_MAJOR);
           }
        } else {
            return false;
        }
    } else {
        return false;
    }
    if (cell && soft_max) {
        for(int e = 0; e < epoch; e++) {
            clock_t start_time;
            start_time = clock();

            std::streampos pos1;
            std::streampos pos2;

            for(int i = 0; i<bucket_list.size(); i++) {
                size_t dp = bucket_list.at(i).data_points;
                size_t l1 = bucket_list.at(i).first_sen_len;
                size_t l2 = bucket_list.at(i).second_sen_len;
                W1 = (T*)this->pool->allocateMemory(sizeof(T)*dp*l1);
                W2 = (T*)this->pool->allocateMemory(sizeof(T)*dp*l2);
                if(W1 && W2) {
                    if(setDataMatrices(s_sen,d_sen,pos1,pos2,i,W1,W2)) {
                        if(handle == NULL) {
                            W_1 = W1;
                            W_2 = W2;
                        } else {
                            CUDA_SAFE_CALL(cudaMalloc((void **)&W_1,sizeof(T)*dp*l1));
                            CUDA_SAFE_CALL(cudaMalloc((void **)&W_2,sizeof(T)*dp*l2));
                            CUBLAS_SAFE_CALL(cublasSetVector(dp*l1,sizeof(T),W1,1,W_1,1));
                            CUBLAS_SAFE_CALL(cublasSetVector(dp*l2,sizeof(T),W2,1,W_2,1));
                        }

                        _train(W_1,l1,W_2,l2,dp);

                    } else {
                        printf("setDataMatrices returned false.\n");
                        return false;
                    }

                } else {
                    printf("Could not allocate memory for computing data matrices in train method.\n");
                    return false;
                }
                this->pool->freeMemory(W1,sizeof(T)*dp*l1); W1 = NULL;
                this->pool->freeMemory(W2,sizeof(T)*dp*l2); W2 = NULL;
                if(handle != NULL) {
                    CUDA_SAFE_CALL(cudaFree(W_1));
                    CUDA_SAFE_CALL(cudaFree(W_2));
                }

                std::cout<< "Sentences Processed "<<dp<<" In Bucket index "<<i<<std::endl;
            }
            std::cout << "Epoch: " << e+1 <<" Time Taken(Microsecond) " <<(clock() - start_time) << std::endl;
            if((e+1)%5 == 0) {
                if(EPOCH) this->cell->print(this->dumpfile);
                if(EPOCH) this->soft_max->print(this->dumpfile);
            }
        }


    } else {
        printf("RNN CELL or SOFTMAX CELL not set.\n");
        return false;
    }

    write_test(cell,soft_max);
    return true;
}

template<class T>
bool RNN<T>::_predict(std::fstream &sfile, std::fstream &dfile)
{
    int *_null = NULL;
    if (sfile.good() && dfile.good()) {
        std::string line;
        std::stringstream ss;
        std::string word;

        T* hin = (T*)this->pool->allocateMemory(N_H*sizeof(T));
        T* hout = (T*)this->pool->allocateMemory(N_H*sizeof(T));
        T* dout = (T*)this->pool->allocateMemory(N_H*sizeof(T));
        T* wIndex = (T*)this->pool->allocateMemory(1*sizeof(T));
        memset(hin,0,N_H*sizeof(T));
        memset(hout,0,N_H*sizeof(T));
        memset(dout,0,N_H*sizeof(T));
        memset(wIndex,0,1*sizeof(T));

        while(std::getline(sfile,line)) {
            ss.clear();
            ss.str(line);
            ss >> word;
            /*T* hin = (T*)this->pool->allocateMemory(N_H*sizeof(T));
            T* hout = (T*)this->pool->allocateMemory(N_H*sizeof(T));
            T* dout = (T*)this->pool->allocateMemory(N_H*sizeof(T));
            T* wIndex = (T*)this->pool->allocateMemory(1*sizeof(T));
            memset(hin,0,N_H*sizeof(T));
            memset(hout,0,N_H*sizeof(T));
            memset(dout,0,N_H*sizeof(T));
            memset(wIndex,0,1*sizeof(T));*/
            

            //SEN-START
            int sindex = source_hash["SEN-STRT"];
            if(PRINT_LIMITED) PRINTMATRIX("\nword ",&sindex,1,1,NULL,this->dumpfile);
            T* din = &W_XS[N_X*sindex];
            this->cell->stepForward(hin,din,hout,dout);

            this->soft_max->activate(hout);
            this->soft_max->classify(wIndex);

            T* tmp = hin;
            hin = hout;
            hout = tmp;
            int counter = 0;
            while(word.size()>0) {
                counter++;
                int sindex = source_hash[word];
                if(PRINT_LIMITED) PRINTMATRIX("\nword ",&sindex,1,1,NULL,this->dumpfile);
                T* din = &W_XS[N_X*sindex];
                this->cell->stepForward(hin,din,hout,dout);

                this->soft_max->activate(hout);
                this->soft_max->classify(wIndex);

                if(PRINT_LIMITED) PRINTMATRIX("\nPredict: word ",wIndex,1,1,NULL,this->dumpfile);

                word.erase();
                T* tmp = hin;
                hin = hout;
                hout = tmp;
                ss >> word;
            }

            //SEN-END
            sindex = source_hash["SEN-END"];
            if(PRINT_LIMITED) PRINTMATRIX("\nword ",&sindex,1,1,NULL,this->dumpfile);
            din = &W_XS[N_X*sindex];
            this->cell->stepForward(hin,din,hout,dout);

            tmp = hin;
            hin = hout;
            hout = tmp;


            this->soft_max->activate(hin);
            this->soft_max->classify(wIndex);

            if(PRINT_LIMITED) PRINTMATRIX("\nPredict: word ",wIndex,1,1,NULL,this->dumpfile);

            int prev_index = (int)*wIndex;

            dfile << dest_vocab.at(prev_index) << " ";


            int itr = 0;
            do {
                itr++;
                if(itr > (counter+5)) {
                    break;
                }
                if(PRINT_LIMITED) PRINTMATRIX("\nword ",_null,0,0,NULL,this->dumpfile);
                if(PRINT) PRINTMATRIX("",&prev_index,1,1,NULL,this->dumpfile);
                if(PRINT) PRINTMATRIX(dest_vocab.at(prev_index).c_str(),_null,0,0,NULL,this->dumpfile);
                if(PRINT) PRINTMATRIX("\n",_null,0,0,NULL,this->dumpfile);
                T *din = &W_XD[prev_index];
                this->cell->stepForward(hin,din,hout,dout);

                T* tmp = hin;
                hin = hout;
                hout = tmp;


                this->soft_max->activate(hin);
                this->soft_max->classify(wIndex);

                prev_index = (int)*wIndex;

                dfile << dest_vocab.at(prev_index) << " ";


            } while(dest_vocab.at(prev_index).compare("SEN-END")!=0);

            dfile<<std::endl;



        }
        this->pool->freeMemory(hin,N_H*sizeof(T));
        this->pool->freeMemory(hout,N_H*sizeof(T));
        this->pool->freeMemory(dout,N_H*sizeof(T));
        this->pool->freeMemory(wIndex,1*sizeof(T));
        //this->pool->freeMemory(hin,N_H*sizeof(T));



    } else {
        printf("files can not be read or written in _predict method\n");
    }
}

template<class T>
bool RNN<T>::predict(const char *source, const char *s_word_vec, size_t s_vocab_size, const char *d_word_vec, size_t d_vocab_size, size_t word_vec_size, size_t hidden_layer_size, const char *output)
{
    if(source && s_word_vec && d_word_vec && output) {
        this->setSourceVocabSize(s_vocab_size);
        this->setDestVocabSize(d_vocab_size);
        this->setWordVecSize(word_vec_size);
        this->setHiddenLayerSize(hidden_layer_size);
        if(setWordVecMatrices(s_word_vec,d_word_vec)) {
           if(PRINT) PRINTMATRIX("Source Word Vector\n",W_XS,N_X,N_S,this->handle,this->dumpfile);
           if(PRINT) PRINTMATRIX("Dest Word Vector\n",W_XD,N_X,N_D,this->handle,this->dumpfile);

        } else {
            return false;
        }
    } else{

    }
    if (cell && soft_max) {
        std::fstream sfile;
        sfile.open(source,std::ios::in);
        std::fstream dfile;
        dfile.open(output,std::ios::out|std::ios::app);
        if(sfile.is_open()) {
            if(dfile.is_open()) {
                _predict(sfile,dfile);
            }
        }

        sfile.close();
        dfile.close();
    }



}

/*
template<class T>
bool RNN<T>::createBucket(const char *s_buckets, const char *d_buckets)
{
    size_t max_mem_aval_for_data_proccessing = 2048; //TODO

    std::fstream f_s_buckets;
    std::fstream f_d_buckets;

    std::string s_line;
    std::string d_line;

    f_s_buckets.open(s_buckets,std::ios::in);
    f_d_buckets.open(d_buckets,std::ios::in);


    if(f_s_buckets.is_open() && f_d_buckets.is_open()) {
        std::stringstream ss;
        size_t mem_est = 0;
        bool done = false;
        bool s_first_time = true;
        bool d_first_time = true;
        std::string word;
        size_t sdp = 0, slen = 0, ddp = 0, dlen = 0, curr_dp = 0, new_dp = 0, new_slen = 0, new_dlen = 0;
        size_t l1_max;
        size_t l2_max;
        while(1) {
            if(sdp == 0) {
                if(s_first_time && std::getline(f_s_buckets,s_line)) {
                    if(s_line.size() > 0) {
                        ss.clear();ss.str(s_line);
                        ss >> word; if(word.size()>0) l1_max = atol(word.c_str());

                    } else {
                        return false;
                    }
                }
                if(std::getline(f_s_buckets,s_line)) {
                    if(s_line.size() > 1) {
                        ss.clear();ss.str(s_line);
                        ss >> word; if(word.size()>0) sdp = atol(word.c_str());
                        ss >> word; if(word.size()>0) slen = atol(word.c_str());

                    } else {
                        return false;
                    }

                } else {
                    done = true;
                }
            }

            if(ddp == 0 ) {
                if(d_first_time && std::getline(f_d_buckets,d_line)) {
                    if(d_line.size() > 0) {
                        ss.clear();ss.str(d_line);
                        ss >> word; if(word.size()>0) l2_max = atol(word.c_str());

                    } else {
                        return false;
                    }
                }
                if(std::getline(f_d_buckets,d_line)) {
                    if (d_line.size()>1) {
                        ss.clear();ss.str(d_line);
                        ss >> word; if(word.size()>0) ddp = atol(word.c_str());
                        ss >> word; if(word.size()>0) dlen = atol(word.c_str());

                    } else {
                        return false;
                    }
                } else {
                    done = true;
                }
            }

            if(s_first_time && d_first_time) {
                size_t max_size_one_dp = trainSpaceRequiredForOneDataPoint(l1_max,l2_max);
                if(max_size_one_dp > max_mem_aval_for_data_proccessing) {
                    printf("can not create bucket even by one sentence pair a bucket(considering max size sentences).\n");
                    return false;
                }
                s_first_time = false;
                d_first_time = false;
            }

            if((sdp!= 0 && ddp != 0) || done) {

                curr_dp = min(sdp,ddp);
                sdp -= curr_dp;
                ddp -= curr_dp;
                new_dp += curr_dp;
                new_slen = slen;
                new_dlen = dlen;
                mem_est = new_dp*trainSpaceRequiredForOneDataPoint(new_slen,new_dlen);
                if(mem_est >= max_mem_aval_for_data_proccessing || done) {
                    size_t diff = 0;
                    if( mem_est <= max_mem_aval_for_data_proccessing) {

                    } else {
                        diff = mem_est - max_mem_aval_for_data_proccessing;
                    }

                    size_t num_dp_to_exclude = ceil(diff,trainSpaceRequiredForOneDataPoint(new_slen,new_dlen));
                    new_dp -= num_dp_to_exclude;
                    sdp += num_dp_to_exclude;
                    ddp += num_dp_to_exclude;
                    if(new_dp > 0) {
                        bucket b;
                        b.data_points = new_dp;
                        b.first_sen_len = new_slen;
                        b.second_sen_len = new_dlen;
                        bucket_list.push_back(b);
                    }
                    new_dp = 0;
                    new_slen = 0;
                    new_dlen = 0;
                    mem_est = 0;
                    if(done && sdp == 0 && ddp == 0) {
                        break;
                    }
                } else {

                }

            } else if((sdp == 0 && ddp != 0) || (sdp!= 0 && ddp == 0)){
                printf("Reached problem in createBucket\n");
                return false;

            } else {
                break;
            }

        }

    } else {
        printf("Cannot open bucket files.\n");
        return false;
    }

    f_s_buckets.close();
    f_d_buckets.close();

    return true;
}*/

#endif // RNN_CPP
