#ifndef HOUGH_H
#define HOUGH_H

#include <cstdint>
#include <cstdlib>
#include <semaphore.h>
#include "Hough_core.h"
#include <opencv2/core/mat.hpp>
using namespace std;
using namespace cv;

class MT_para{
    public:
        xy<uint32_t> in_size;    //in image size
        xy<uint32_t> out_size;   //out_rows=N_line, out_cols=N_rad
        point_u32 start;      //search start xy
        range_float rads;        //rads range
        uint32_t N_length;       //points per line
        uint32_t N_thread;       //number of process threads
        void size_from_OpenCV_Mat(Mat &mat_in, Mat &mat_out);
};

class Hough
{
    public:
        Hough(MT_para *paras);
        virtual ~Hough();
        void     init(MT_para *paras);
        void     update(uint8_t *pIn, uint8_t *pOut);
        void     waitOK();

        inline static Hough_core *thread_obj;    //每个线程的对象

    protected:
        static void* thread_func(void* args);

    private:
        inline static uint32_t n_thread;
        inline static bool threads_running;      //set false to stop threads
        inline static pthread_t *threads;        //threads
        static array2d<uint8_t> img_in;   //camera image input
        static array2d<uint8_t> img_out;  //Hough result out
        inline static sem_t sem0;                //wait data
        inline static sem_t sem1;                //data process OK, post in end of process
};

#endif // HOUGH_H
