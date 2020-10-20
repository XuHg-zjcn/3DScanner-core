#include "optflow_FFT.h"
#include <opencv2/core/core.hpp>
#include <fftw3.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <cmath>
using namespace std;

#ifdef D3SCANNER_CORE_USE_SINGLE
#define double float
#endif

optflow_FFT::optflow_FFT(int n)
{
    //ctor
    this->n = n;
    crop_db =    (double*)fftw_malloc(sizeof(double)*n*n);
    out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n*(n/2+1));
    out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n*(n/2+1));
    mul  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n*(n/2+1));
    ifft =       (double*)fftw_malloc(sizeof(double)*n*n);
    if(fftw_import_wisdom_from_filename("wisdom.fftw")!=0){
        p1 = fftw_plan_dft_r2c_2d(n, n, crop_db, out1, FFTW_WISDOM_ONLY);
        p2 = fftw_plan_dft_r2c_2d(n, n, crop_db, out2, FFTW_WISDOM_ONLY);
        p_ifft = fftw_plan_dft_c2r_2d(n, n, mul, ifft, FFTW_WISDOM_ONLY);
    }else{
        p1 = fftw_plan_dft_r2c_2d(n, n, crop_db, out1, FFTW_ESTIMATE);
        p2 = fftw_plan_dft_r2c_2d(n, n, crop_db, out2, FFTW_ESTIMATE);
        p_ifft = fftw_plan_dft_c2r_2d(n, n, mul, ifft, FFTW_ESTIMATE);
    }
    NAreas = 0;
}

optflow_FFT::~optflow_FFT()
{
    //dtor
    fftw_free(crop_db);
    fftw_free(out1);
    fftw_free(out2);
    fftw_free(mul);
    fftw_free(ifft);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
}

void optflow_FFT::run(int i)
{
    switch(i) {
        case 0:
            fftw_execute(p1);
            break;
        case 1:
            fftw_execute(p2);
            break;
        case 2:
            fftw_execute(p_ifft);
            break;
        default:
            break;
    }
}

int optflow_FFT::save()
{
    FILE *fp;
    fp = fopen("wisdom.fftw", "w");
    if(fp == nullptr){
#ifndef D3SCANNER_CORE_NOTEST
        cout<<"file open faild"<<endl;
#endif
        return 1;
    }
    fftw_export_wisdom_to_file(fp);
    fclose(fp);
    return 0;
}

void optflow_FFT::fill_data(Mat &mat_in, int x0, int y0)
{
    uint8_t *p_u8;
    double *p_db = crop_db;
    for(int i=y0;i<y0+n;i++) {
        p_u8 = mat_in.ptr(i, x0);
        for(int j=0;j<n;j++) {
            *p_db++ = (*p_u8++)/255.0;
        }
    }
}

void optflow_FFT::calc_delta(bool sq2)
{
    double mul_real, mul_imag, sqrt2;
    for(int i=0;i<n*(n/2+1);i++) {
        mul_real = out1[i][0]*out2[i][0] + out1[i][1]*out2[i][1];
        mul_imag =-out1[i][0]*out2[i][1] + out1[i][1]*out2[i][0];
        sqrt2 = sqrt(mul_real*mul_real + mul_imag*mul_imag);
        if(sq2)
            sqrt2 = sqrt(sqrt2);
        mul[i][0] = mul_real/sqrt2;
        mul[i][1] = mul_imag/sqrt2;
    }
}

double line_sum(double *start, const double *last, double *results)
{
    double sum = 0;
    double x2;
    while(start<last) {
        x2 = (*start) * (*start);
        sum += x2;
        (*results++) = x2;
        start++;
    }
    return sum;
}

double optflow_FFT::corner4_sum(double *results, const int w)
{
    double sum = 0;
    for(int i=0;i<w;i++) {
        sum += line_sum(&ifft[i*n], &ifft[i*n+w], results);
        results += w;
        sum += line_sum(&ifft[i*n+n-w], &ifft[i*n+n], results);
        results += w;
    }
    for(int i=n-w;i<n;i++) {
        sum += line_sum(&ifft[i*n], &ifft[i*n+w], results);
        results += w;
        sum += line_sum(&ifft[i*n+n-w], &ifft[i*n+n], results);
        results += w;
    }
    return sum;
}

double optflow_FFT::ifft_sum()
{
    double sum = 0;
    double *p = ifft;
    for(int i=0;i<n*n;i++) {
        sum += (*p) * (*p);
        p++;
    }
    return sum;
}
//@para w:Signal window wide
//@para most: how most enegry in window, result to NtopMost, suggest 0.8~0.95
//@para SumNtop: number top to Sum, result to SumTop, suggest 4-6
//@para info: output info
void optflow_FFT::get_ifft_info(int w, double most, int SumNtop, ifft_quality *info)
{
    double mAll = 0;          //mean all
    double mWin = 0;          //mean in window
    double Signal, Noise;
    double *wsort = new double[w*w*4];
    double wsort_partsum = 0;
    double SumTop = 0;
    int Npart = 0;

    mWin = corner4_sum(wsort, w);
    mAll = ifft_sum();
    //mWin = Signal+w2*Noise;
    //mAll = Signal+n2*Noise;
    Signal = (n*n*mWin - w*w*4*mAll)/(n*n-w*w*4);
    Noise  = (mAll - mWin)/(n*n-w*w*4);

    sort(&wsort[0], &wsort[w*w], greater<double>());
    while(wsort_partsum < mWin*most+Noise*Npart && Npart<w*w) {
        wsort_partsum += wsort[Npart];
        Npart++;
    }

    for(int i=0;i < SumNtop;i++) {
        SumTop += wsort[i];
    }

    info->SNR = Signal/Noise/(n*n);
    info->TopP = SumTop/Signal;
    info->Nmost = Npart;
    delete[] wsort;
}

void optflow_FFT::xsum(double dx, double dy, fftw_complex &ret)
{
    int n21=n/2+1;
    double v;//, ret=0;
    ret[0]=0;
    ret[1]=0;
    for(int i=0;i<n21;i++) {
        for(int j=0;j<n21;j++) {
            v = (double)(i*dx+j*dy)/n*2*M_PI;
            ret[0] += mul[n21*i+j][0]*cos(v) - mul[n21*i+j][1]*sin(v);
            ret[1] += mul[n21*i+j][0]*sin(v) + mul[n21*i+j][1]*cos(v);
        }
    }
    ret[0]/=n21*n21;
    ret[1]/=n21*n21;
    ret[0]= sqrt(ret[0]*ret[0] + ret[1]*ret[1]);
}

void optflow_FFT::complex_to_u8(fftw_complex *pIn, uint8_t *pOut, int len)
{
    double v;
    double vmax=*pIn[0];
    double vmin=*pIn[0];
    fftw_complex *pIn1=pIn;
    for(int i=0;i<len;i++) {
        v = (*pIn1++)[0];
        if(v<vmin)
            vmin=v;
        else if(v>vmax)
            vmax=v;
    }
    pIn1=pIn;
    vmin=-10;
    vmax=10;
    for(int i=0;i<len;i++) {
        v = (*pIn1++)[0];
        v = (v-vmin)/(vmax-vmin)*255;
        v = v<0   ?   0 : v;
        v = v>255 ? 255 : v;
        *pOut++ = (uint8_t)v;
    }
}

void optflow_FFT::double_to_u8(double *pIn, uint8_t *pOut, int len)
{
    double v;
    double vmax=*pIn;
    double vmin=*pIn;
    double *pIn1=pIn;
    for(int i=0;i<len;i++) {
        v = *pIn1++;
        if(v<vmin)
            vmin=v;
        else if(v>vmax)
            vmax=v;
    }
    pIn1=pIn;
    for(int i=0;i<len;i++) {
        v = *pIn1++;
        v = (v-vmin)/(vmax-vmin)*255;
        v = v<0   ?   0 : v;
        v = v>255 ? 255 : v;
        *pOut++ = (uint8_t)v;
    }
}

void optflow_FFT::WT(Mat *out, double div)
{
    int w=out->cols;
    fftw_complex *vars=(fftw_complex*)malloc(sizeof(fftw_complex)*w*w);
    fftw_complex *vars1=vars;
    for(int i=-w/2;i<w/2;i++) {  //i,j = range(-width/2, width/2)
        for(int j=-w/2;j<w/2;j++) {
            xsum(i/div, j/div, *vars1++);
        }
    }
    complex_to_u8(vars, out->ptr(), w*w);
    free(vars);
}

//require mul
//call after calc_delta
void optflow_FFT::out_ifft(Mat *out)
{
    double_to_u8(ifft, out->ptr(), n*n);
}

void optflow_FFT::copy_mul(Mat *out)
{
    complex_to_u8(mul, out->ptr(), n*(n+1)/2);
}

bool operator> (const AreaDesc& x, const AreaDesc& y) {
    return x.scorce > y.scorce;
}

void optflow_FFT::getGoodArea(Mat &img1, Mat &img2, int max_NArea, double min_scorce)
{
    assert(img1.rows == img2.rows);
    assert(img1.cols == img2.cols);
    int NRow = img1.rows/n; //size of areas array
    int NCol = img1.cols/n; //per area is n*n pixel
    ifft_quality info;
    AreaDesc* pAreas;

    NAreas = NRow*NCol;
    assert(max_NArea <= NAreas);

    if(areas == nullptr) {
        areas = (AreaDesc*)malloc(sizeof(AreaDesc)*NAreas);
    }
    pAreas = areas;
    for(int i=0;i<NRow;i++) {
        for(int j=0;j<NCol;j++) {
            fill_data(img1, j*n, i*n);
            run(0);
            fill_data(img2, j*n, i*n);
            run(1);
            calc_delta(true);
            run(2);
            get_ifft_info(8, 0.9, 5, &info);
            pAreas->id = i*NCol+j;
            pAreas->x0 = j*n;
            pAreas->y0 = i*n;
            pAreas->is_Good = false;
            pAreas->qua = info;
            pAreas->scorce = info.SNR*info.TopP;
            pAreas++;
        }
    }
    AreaDesc *areas2 = new AreaDesc[NAreas];
    memcpy(areas2, areas, sizeof(AreaDesc)*NAreas);
    sort(areas2, areas2+NAreas, greater<AreaDesc>());
    for(int i=0;i<max_NArea;i++) {
        if(areas2[i].scorce >= min_scorce) {
            areas[areas2[i].id].is_Good = true;
        }else{
            break;
        }
    }
    delete[] areas2;
}

//@para color:color image to draw masks
void optflow_FFT::draw_mask(Mat &color)
{
    uint8_t *c = color.ptr();
    uint8_t *c1;
    for(int i=0;i<NAreas;i++) {                              //i:nth of Area
        if(areas->is_Good) {
#ifndef D3SCANNER_CORE_NOTEST
            cout<<right<<setw(5)<< areas->x0 << ',';
            cout<<right<<setw(5)<< areas->y0 << ',';
            cout<<' '<<left <<setw(4)<< areas->scorce <<endl;
#endif
            c1 = c + ((areas->y0)*color.cols + areas->x0)*3; //first pixel of Area
            for(int j=0;j<n;j++) {                           //j:row in Area
                for(int k=0;k<n;k++) {                       //K:row in Area
                    c1++;
                    *c1 = (*c1)/4*2 + 255/4*2;
                    c1++;
                    c1++;
                }
                c1 += (color.cols-n)*3;
            }
        }
        areas++;
    }
}
