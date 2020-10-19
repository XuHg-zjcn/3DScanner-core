#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include "Hough.h"
#include "Hough_core.h"
#define pi 3.1415926535
#include "optflow_FFT.h"
using namespace std;
using namespace cv;

void runHough(Mat &gray, Mat &out);
void runFFT(Mat &gray, Mat &show_wave, Mat &show_ifft);
void runSNRTest(const Mat &gray, Mat &color, int n);
void draw_mask(Mat &color, AreaDesc *areas, int NAreas, int n);

int main(int argc, char *argv[])
{
    Mat img_color = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    Mat gray;
    Mat out;
    Mat show_wave, show_ifft;
    Mat scaleUp;
    if(img_color.empty()){
        cout<<"Test image 'lena.jpg'(256x256) not found!\n";
        return -1;
    }
    gray.create(img_color.size(), CV_8UC1);
    cvtColor(img_color, gray, CV_RGB2GRAY);

    runHough(gray, out);

    show_wave.create(64, 33, CV_8UC1);
    show_ifft.create(64, 64, CV_8UC1);

    runFFT(gray, show_wave, show_ifft);
    runSNRTest(gray, img_color, 32);

    resize(show_ifft, scaleUp, Size(), 4.0, 4.0, INTER_NEAREST);
    imshow("ifft_4x", scaleUp);
    imshow("wave", show_wave);
    imshow("img_color", img_color);

    waitKey(0);
    return 0;
}

void runHough(Mat &gray, Mat &out)
{
    MT_para m_para;
    timespec ts0{}, ts1{};
    Hough *mHough;
    out.create(50, 50, CV_8UC1);
    m_para.size_from_OpenCV_Mat(gray, out);
    m_para.start.set_xy16(128, 128);
    m_para.rads={start:0, step:pi/50};
    m_para.N_length=50;
    m_para.N_thread=1;
    mHough = new Hough(&m_para);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts0);
    mHough->update(gray.ptr(), out.ptr());
    mHough->waitOK();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts1);
    //cout<< (ts1.tv_nsec - ts0.tv_nsec)/1000 << "us" <<endl;
}

void runFFT(Mat &gray, Mat &show_wave, Mat &show_ifft)
{
    optflow_FFT *offt = new optflow_FFT(64);
    ifft_quality info;
    offt->fill_data(gray, 40, 130);
    offt->run(0);
    offt->fill_data(gray, 45, 133);
    offt->run(1);
    offt->calc_delta(true);
    offt->copy_mul(&show_wave);
    offt->run(2);
    offt->out_ifft(&show_ifft);
    offt->get_ifft_info(8, 0.9, 5, &info);
    cout<<"SNR  :"<<info.SNR<<endl;
    cout<<"TopP :"<<info.TopP<<endl;
    cout<<"Nmost:"<<info.Nmost<<endl;
    delete offt;
}

void runSNRTest(const Mat &gray, Mat &color, int n)
{
    optflow_FFT *offt = new optflow_FFT(n);
    Rect rect1 = Rect(0, 0, 250, 250);
    Rect rect2 = Rect(2, 3, 250, 250);
    Mat crop1 = gray(rect1);
    Mat crop2 = gray(rect2);
    offt->getGoodArea(crop1, crop2, 20, 0.0);
    offt->draw_mask(color);
    delete offt;
}
