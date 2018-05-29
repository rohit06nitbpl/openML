#include <fstream>
#include <stdlib.h>

#include "rnn.h"
#include "memorypool.h"

#define TEST 1

void trn(double *w_vh, double *v_bias_v, double *w_hh, double *w_hx, double *v_bias_h, size_t epoch)
{
    size_t N_H = 0;
    size_t N_X = 0;
    size_t N_S = 0;
    size_t N_D = 0;

    if(TEST) {
        N_H = 10;
        N_X = 30;
        N_S = 30;
        N_D = 30;
    } else {
        N_H = 100;
        N_X = 100;
        N_S = 5890;
        N_D = 7999;
    }

    cublasHandle_t handle = NULL;
    /* initialize random seed: */
    srand (time(NULL));

    // Create Dump File
    std::fstream file;
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "./output/%Y-%m-%d.%X.txt", &tstruct);
    file.open(buf,std::ios::out | std::ios::app);

    //create RNN
    RNN<double> rnn(file);

    //create Memory Pool
    MemoryPool cpupool;
    MemoryPool gpupool(handle);

    //Create GPU HANDLE
    if(GPU) {
        if(rnn.createCublasHandle()) {
            handle = rnn.getCublasHandle();
            gpupool.createPool(1024000); //1 MB
        }
    } else {
        cpupool.createPool(1073741824); //1 GB
    }

    rnn.setCPUPOOL(&cpupool);
    rnn.setGPUPOOL(&gpupool);

    ActivationFunctions<double>* a_f = new SigmoidFunction<double>;
    ActivationFunctions<double>* a_f_softmax = new SoftmaxFunction<double>;

    //create RNN Cell
    SimpleRNNCell<double> *rnncell = new SimpleRNNCell<double>(file);
    rnncell->setCPUPOOL(&cpupool);
    rnncell->setGPUPOOL(&gpupool);
    rnncell->setActivationFunction(a_f);
    rnncell->setCublasHandle(handle);
    rnncell->initialize(N_H,N_X,COLUMN_MAJOR,w_hh,w_hx,v_bias_h);

    //create SOFTMAX Cell
    SoftmaxCell<double> *softmax_cell = new SoftmaxCell<double>(file);
    softmax_cell->setCPUPOOL(&cpupool);
    softmax_cell->setGPUPOOL(&gpupool);
    softmax_cell->setActivationFunction(a_f_softmax);
    softmax_cell->setCublasHandle(handle);
    softmax_cell->initialize(N_D,N_H,COLUMN_MAJOR,w_vh,v_bias_v);

    //Set RNN
    //rnn.setCPUPOOL(&cpupool);
    //rnn.setGPUPOOL(&gpupool);
    rnn.setRNNCell(rnncell);
    rnn.setSoftMaxCell(softmax_cell);

    if(TEST) {
         rnn.train("./test/en_label.txt","./test/en_hot_vec.txt",N_S,"./test/de_label.txt","./test/de_hot_vec.txt",N_D,"./test/bucket_info.txt",N_X,N_H,epoch);
    } else {
        rnn.train("./EN-DE/en_label.txt","./EN-DE/en_word_vec.txt",N_S,"./EN-DE/de_label.txt","./EN-DE/de_word_vec.txt",N_D,"./EN-DE/bucket_info.txt",N_X,N_H,epoch);
    }



}

void first_run() {

   trn(NULL,NULL,NULL,NULL,NULL,100);

}

void second_run() {

    size_t N_H = 0;
    size_t N_X = 0;
    size_t N_S = 0;
    size_t N_D = 0;
    size_t EPC = 10;

    if(TEST) {
        N_H = 10;
        N_X = 30;
        N_S = 30;
        N_D = 30;

        //READ PREVIOUS WEIGHT
        double *W_VH = (double*)malloc(N_D*N_H*sizeof(double));
        double *V_BIAS_V = (double*)malloc(N_D*sizeof(double));
        double *W_HH = (double*)malloc(N_H*N_H*sizeof(double));
        double *W_HX = (double*)malloc(N_H*N_X*sizeof(double));
        double *V_BIAS_H = (double*)malloc(N_H*sizeof(double));

        std::fstream file;
        file.open("./test/prev/w_vh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_VH,N_D,N_H,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_vh.txt\n");
        }

        file.open("./test/prev/v_bias_v.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_V,N_D,1,file);
           file.close();
        } else{
            printf("can not read ./test/prev/v_bias_v.txt\n");
        }

        file.open("./test/prev/w_hh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HH,N_H,N_H,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_hh.txt\n");
        }

        file.open("./test/prev/w_hx.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HX,N_H,N_X,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_hx.txt\n");
        }

        file.open("./test/prev/v_bias_h.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_H,N_H,1,file);
           file.close();
        } else{
            printf("can not read ./test/prev/v_bias_h.txt\n");
        }

        trn(W_VH,V_BIAS_V,W_HH,W_HX,V_BIAS_H,EPC);

        //FREE MEM
        free(W_VH);
        free(V_BIAS_V);
        free(W_HH);
        free(W_HX);
        free(V_BIAS_H);


    } else {
        N_H = 100;
        N_X = 100;
        N_S = 5890;
        N_D = 7999;

        //READ PREVIOUS WEIGHT
        double *W_VH = (double*)malloc(N_D*N_H*sizeof(double));
        double *V_BIAS_V = (double*)malloc(N_D*sizeof(double));
        double *W_HH = (double*)malloc(N_H*N_H*sizeof(double));
        double *W_HX = (double*)malloc(N_H*N_X*sizeof(double));
        double *V_BIAS_H = (double*)malloc(N_H*sizeof(double));

        std::fstream file;
        file.open("./EN-DE/prev/w_vh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_VH,N_D,N_H,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_vh.txt\n");
        }

        file.open("./EN-DE/prev/v_bias_v.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_V,N_D,1,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/v_bias_v.txt\n");
        }

        file.open("./EN-DE/prev/w_hh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HH,N_H,N_H,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_hh.txt\n");
        }

        file.open("./EN-DE/prev/w_hx.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HX,N_H,N_X,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_hx.txt\n");
        }

        file.open("./EN-DE/prev/v_bias_h.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_H,N_H,1,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/v_bias_h.txt\n");
        }


        trn(W_VH,V_BIAS_V,W_HH,W_HX,V_BIAS_H,EPC);

        //FREE MEM
        free(W_VH);
        free(V_BIAS_V);
        free(W_HH);
        free(W_HX);
        free(V_BIAS_H);

    }

}

void pred(double *w_vh, double *v_bias_v, double *w_hh, double *w_hx, double *v_bias_h)
{
    size_t N_H = 0;
    size_t N_X = 0;
    size_t N_S = 0;
    size_t N_D = 0;

    if(TEST) {
        N_H = 10;
        N_X = 30;
        N_S = 30;
        N_D = 30;
    } else {
        N_H = 100;
        N_X = 100;
        N_S = 5890;
        N_D = 7999;
    }

    cublasHandle_t handle = NULL;
    /* initialize random seed: */
    srand (time(NULL));

    // Create Dump File
    std::fstream file;
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "./output/pred-%Y-%m-%d.%X.txt", &tstruct);
    file.open(buf,std::ios::out | std::ios::app);

    //create RNN
    RNN<double> rnn(file);

    //create Memory Pool
    MemoryPool cpupool;
    MemoryPool gpupool(handle);

    //Create GPU HANDLE
    if(GPU) {
        if(rnn.createCublasHandle()) {
            handle = rnn.getCublasHandle();
            gpupool.createPool(1024000); //1 MB
        }
    } else {
        cpupool.createPool(1073741824); //1 GB
    }

    rnn.setCPUPOOL(&cpupool);
    rnn.setGPUPOOL(&gpupool);

    ActivationFunctions<double>* a_f = new SigmoidFunction<double>;
    ActivationFunctions<double>* a_f_softmax = new SoftmaxFunction<double>;

    //create RNN Cell
    SimpleRNNCell<double> *rnncell = new SimpleRNNCell<double>(file);
    rnncell->setCPUPOOL(&cpupool);
    rnncell->setGPUPOOL(&gpupool);
    rnncell->setActivationFunction(a_f);
    rnncell->setCublasHandle(handle);
    rnncell->initialize(N_H,N_X,COLUMN_MAJOR,w_hh,w_hx,v_bias_h);

    //create SOFTMAX Cell
    SoftmaxCell<double> *softmax_cell = new SoftmaxCell<double>(file);
    softmax_cell->setCPUPOOL(&cpupool);
    softmax_cell->setGPUPOOL(&gpupool);
    softmax_cell->setActivationFunction(a_f_softmax);
    softmax_cell->setCublasHandle(handle);
    softmax_cell->initialize(N_D,N_H,COLUMN_MAJOR,w_vh,v_bias_v);

    //Set RNN
    //rnn.setCPUPOOL(&cpupool);
    //rnn.setGPUPOOL(&gpupool);
    rnn.setRNNCell(rnncell);
    rnn.setSoftMaxCell(softmax_cell);

    if(TEST) {
         rnn.predict("./test/test.txt","./test/en_hot_vec.txt",N_S,"./test/de_hot_vec.txt",N_D,N_X,N_H,"./test/output/pred.txt");
    } else {
        rnn.predict("./EN-DE/test.txt","./EN-DE/en_word_vec.txt",N_S,"./EN-DE/de_word_vec.txt",N_D,N_X,N_H,"./EN-DE/output/pred.txt");
    }

}


void pred_run() {

    size_t N_H = 0;
    size_t N_X = 0;
    size_t N_S = 0;
    size_t N_D = 0;
    //size_t EPC = 10;

    if(TEST) {
        N_H = 10;
        N_X = 30;
        N_S = 30;
        N_D = 30;

        //READ PREVIOUS WEIGHT
        double *W_VH = (double*)malloc(N_D*N_H*sizeof(double));
        double *V_BIAS_V = (double*)malloc(N_D*sizeof(double));
        double *W_HH = (double*)malloc(N_H*N_H*sizeof(double));
        double *W_HX = (double*)malloc(N_H*N_X*sizeof(double));
        double *V_BIAS_H = (double*)malloc(N_H*sizeof(double));

        std::fstream file;
        file.open("./test/prev/w_vh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_VH,N_D,N_H,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_vh.txt\n");
        }

        file.open("./test/prev/v_bias_v.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_V,N_D,1,file);
           file.close();
        } else{
            printf("can not read ./test/prev/v_bias_v.txt\n");
        }

        file.open("./test/prev/w_hh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HH,N_H,N_H,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_hh.txt\n");
        }

        file.open("./test/prev/w_hx.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HX,N_H,N_X,file);
           file.close();
        } else{
            printf("can not read ./test/prev/w_hx.txt\n");
        }

        file.open("./test/prev/v_bias_h.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_H,N_H,1,file);
           file.close();
        } else{
            printf("can not read ./test/prev/v_bias_h.txt\n");
        }

        pred(W_VH,V_BIAS_V,W_HH,W_HX,V_BIAS_H);

        //FREE MEM
        free(W_VH);
        free(V_BIAS_V);
        free(W_HH);
        free(W_HX);
        free(V_BIAS_H);


    } else {
        N_H = 100;
        N_X = 100;
        N_S = 5890;
        N_D = 7999;

        //READ PREVIOUS WEIGHT
        double *W_VH = (double*)malloc(N_D*N_H*sizeof(double));
        double *V_BIAS_V = (double*)malloc(N_D*sizeof(double));
        double *W_HH = (double*)malloc(N_H*N_H*sizeof(double));
        double *W_HX = (double*)malloc(N_H*N_X*sizeof(double));
        double *V_BIAS_H = (double*)malloc(N_H*sizeof(double));

        std::fstream file;
        file.open("./EN-DE/prev/w_vh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_VH,N_D,N_H,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_vh.txt\n");
        }

        file.open("./EN-DE/prev/v_bias_v.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_V,N_D,1,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/v_bias_v.txt\n");
        }

        file.open("./EN-DE/prev/w_hh.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HH,N_H,N_H,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_hh.txt\n");
        }

        file.open("./EN-DE/prev/w_hx.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(W_HX,N_H,N_X,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/w_hx.txt\n");
        }

        file.open("./EN-DE/prev/v_bias_h.txt",std::ios::in);
        if(file.is_open()) {
           READMATRIX(V_BIAS_H,N_H,1,file);
           file.close();
        } else{
            printf("can not read ./EN-DE/prev/v_bias_h.txt\n");
        }


        pred(W_VH,V_BIAS_V,W_HH,W_HX,V_BIAS_H);

        //FREE MEM
        free(W_VH);
        free(V_BIAS_V);
        free(W_HH);
        free(W_HX);
        free(V_BIAS_H);

    }


}


int main(int argc, char *argv[])
{
    first_run();
    //second_run();
    pred_run();
}
