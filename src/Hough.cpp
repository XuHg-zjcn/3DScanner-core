#include "../include/Hough.h"
#include <cassert>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <iostream>

array2d<uint8_t> Hough::img_in({0,0}, false);
array2d<uint8_t> Hough::img_out({0,0}, false);
Hough::Hough(MT_para *paras)
{
    //ctor
    assert(paras->N_thread>0);                //0<线程数<=50
    assert(paras->N_thread<=50);
    n_thread = paras->N_thread;

    threads_running = true; //开启后立即运行
    sem_init(&sem0, 0, 0);  //初始化信号量
    sem_init(&sem1, 0, 0);
    img_in.copy_from_old(paras->in_size);
    img_out.copy_from_old(paras->out_size);
    threads = new pthread_t[n_thread];
    init(paras);
}
Hough::~Hough()
{
    //dtor
    sem_destroy(&sem0);
    sem_destroy(&sem1);
}
//初始化
void Hough::init(MT_para *paras) {
    uint32_t rows_per_thread;
    pthread_attr_t attr;
    size_t stacksize;                  //修改栈大小=4kB

    range_float &rads = paras->rads;
    xy<uint32_t> &in_size = paras->in_size;
    xy<uint32_t> &out_size = paras->out_size;
    uint32_t &N_lines = out_size.x;
    uint32_t &N_rads = out_size.y;

    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stacksize);
    stacksize = 4000; //4KB
    pthread_attr_setstacksize(&attr, stacksize);

    float delta_rads = rads.step * out_size.x;
    assert(out_size.x % n_thread == 0);
    rows_per_thread = out_size.y / n_thread;

    thread_para T_para;
    T_para.in_X0.copy_from_old(paras->start);
    T_para.out_area.x.a = 0;
    T_para.out_area.x.b = out_size.x;
    T_para.out_area.y.a = 0;
    T_para.out_area.y.b = rows_per_thread;
    T_para.rads.start = rads.start;
    T_para.rads.step = rads.step;
    T_para.N_length = paras->N_length;
    for(int i=0;i<n_thread;i++) {
        thread_obj = new Hough_core(&T_para, &img_in, &img_out);
        pthread_create(threads, &attr, thread_func, thread_obj);
        T_para.out_area.y.a += rows_per_thread; //a=range_per_thread*(N_thread)
        T_para.out_area.y.b += rows_per_thread; //b=range_per_thread*(N_thread+1)
        threads++;
        thread_obj++;
    }
}
//单线程任务
void* Hough::thread_func(void* args) {
    Hough_core *obj = (Hough_core*)args;
    while(threads_running) {
        sem_wait(&sem0);
        obj->lines_search();
        sem_post(&sem1);
    }
    pthread_exit(nullptr);
}

void Hough::update(uint8_t *pIn, uint8_t *pOut) {
    img_in.set_ptr(pIn);
    img_out.set_ptr(pOut);
    for(int i=0;i<n_thread;i++){
        sem_post(&sem0);
    }
}

void Hough::waitOK() {
    for(int i=0;i<n_thread;i++)
        sem_wait(&sem1);
}

void MT_para::size_from_OpenCV_Mat(Mat &mat_in, Mat &mat_out)
{
    in_size.x = mat_in.cols;
    in_size.y = mat_in.rows;
}

