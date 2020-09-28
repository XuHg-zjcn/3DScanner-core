#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <ctime>
#include <iostream>
#include "Hough.h"
#include "Hough_core.h"
#define pi 3.1415926535
using namespace std;
using namespace cv;
Hough *mHough;
MT_para m_para;
timespec ts0{}, ts1{};
int main(int argc, char *argv[])
{
    vector<Mat>Images(3);
    Mat img_color = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    Mat gray;
    Mat out;
    if(img_color.empty())
       return -1;
    gray.create(img_color.size(), CV_8UC1);
    cvtColor(img_color, gray, CV_RGB2GRAY);

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
    cout<< (ts1.tv_nsec - ts0.tv_nsec)/1000 << "us" <<endl;

    Images[0] = img_color;
    Images[1] = gray;
    Images[2] = out;
    imshow("out", out);
    imshow("in", gray);
    waitKey(0);
    return 0;
}
