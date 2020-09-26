#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <ctime>
#include "Hough.h"
#define pi 3.1415926535
using namespace std;
using namespace cv;
Hough *mHough;
MT_para m_para;
int main(int argc, char *argv[])
{
    vector<Mat>Images(3);
    Mat img_color = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    Mat gray;
    Mat out;
    if(img_color.empty())
       return -1;
    cvtColor(img_color, gray, CV_RGB2GRAY);

    out.create(100, 100, CV_8UC1);
    m_para.size_from_OpenCV_Mat(gray, out);
    m_para.start={100,100};
    m_para.rads={0,pi/100};
    m_para.N_length=50;
    m_para.N_thread=2;
    mHough = new Hough(&m_para);
    mHough->update(img_color.ptr(), out.ptr());
    mHough->waitOK();
    Images[0] = img_color;
    Images[1] = gray;
    Images[2] = out;
    imshow("out", out);
    waitKey(0);
    return 0;
}
