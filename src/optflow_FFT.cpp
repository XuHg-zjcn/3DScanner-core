#include "optflow_FFT.h"
#include <opencv2/core/core.hpp>
#include <fftw3.h>
#include <unistd.h>
#include <time.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
using namespace std;

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
        p1 = fftw_plan_dft_r2c_2d(n, n, crop_db, out1, FFTW_PATIENT);
        p2 = fftw_plan_dft_r2c_2d(n, n, crop_db, out2, FFTW_PATIENT);
        p_ifft = fftw_plan_dft_c2r_2d(n, n, mul, ifft, FFTW_PATIENT);
    }
}

optflow_FFT::~optflow_FFT()
{
    //dtor
    fftw_free(crop_db);
    fftw_free(out1);
    fftw_free(out2);
    fftw_free(mul);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
}

void optflow_FFT::run(int n)
{
    switch(n) {
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
    if(fp==NULL){
        cout<<"file open faild"<<endl;
        return 1;
    }
    fftw_export_wisdom_to_file(fp);
    fclose(fp);
    return 0;
}

void optflow_FFT::fill_data(Mat &mat_in, int x0, int y0)
{
    uint8_t *ptr_row;
    double *ptr_db=crop_db;
    for(int i=y0;i<y0+n;i++) {
        ptr_row = mat_in.ptr(i, x0);
        for(int j=0;j<n;j++) {
            *ptr_db++=*ptr_row++;
        }
    }
}

void optflow_FFT::calc_delta()
{
    double mul_real, mul_imag, sqrt2, v;
    for(int i=0;i<n*(n/2+1);i++) {
        mul_real = out1[i][0]*out2[i][0] + out1[i][1]*out2[i][1];
        mul_imag =-out1[i][0]*out2[i][1] + out1[i][1]*out2[i][0];
        sqrt2 = sqrt(mul_real*mul_real + mul_imag*mul_imag);
        mul[i][0] = mul_real/sqrt2;
        mul[i][1] = mul_imag/sqrt2;
    }
}

//@para w:Signal window wide
//@para most: how most enegry in window, result to NtopMost, suggest 0.8~0.95
//@para SumNtop: number top to Sum, result to SumTop, suggest 4-6
//@para info: output info
void optflow_FFT::get_ifft_info(int w, double most, int SumNtop, ifft_quality *info)
{
    double mAll = n*n*n*n;//mean all
    double mWin = 0;          //mean in window
    double Signal, Noise;
    double *wsort = new double[w*w];
    double wsort_partsum = 0;
    double SumTop = 0;
    int Npart = 0;
    for(int i=0;i<w;i++) {
        for(int j=0;j<w;j++) {
            mWin += ifft[i*n+j]*ifft[i*n+j];
            wsort[i*w+j] = ifft[i*n+j]*ifft[i*n+j];
        }
    }


    //mWin = Signal+w*Noise;
    //mAll = Signal+n*Noise;
    Signal = (n*n*mWin - w*w*mAll)/(n*n-w*w);
    Noise  = (mAll - mWin)/(n*n-w*w);

    sort(&wsort[0], &wsort[w*w], greater<double>());
    while(wsort_partsum < mWin*most+Noise*Npart && Npart<w*w) {
        wsort_partsum += wsort[Npart];
        Npart++;
    }

    for(int i=0;i < SumNtop;i++) {
        SumTop += wsort[i];
    }

    info->SNR = Signal/Noise;
    info->TopP = SumTop/Signal;
    info->Nmost = Npart;
    delete wsort;
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
    double v,vmax,vmin;
    fftw_complex *pIn1=pIn;
    for(int i=0;i<len;i++) {
        v = (*pIn1++)[0];
        if(v<vmin)
            vmin=v;
        else if(v>vmax)
            vmax=v;
    }
    pIn1=pIn;
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
    double v,vmax,vmin;
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

    assert(max_NArea <= NRow*NCol);
    if(areas == nullptr) {
        areas = (AreaDesc*)malloc(sizeof(AreaDesc)*NRow*NCol);
    }
    pAreas = areas;
    for(int i=0;i<NRow;i++) {
        for(int j=0;j<NCol;j++) {
            fill_data(img1, j*n, i*n);
            run(0);
            fill_data(img2, j*n, i*n);
            run(1);
            calc_delta();
            run(2);
            get_ifft_info(8, 0.9, 5, &info);
            pAreas->id = i*NCol+j;
            pAreas->x0 = j*n;
            pAreas->y0 = i*n;
            pAreas->is_Good = false;
            //pAreas->qua = info;
            pAreas->scorce = info.SNR*info.TopP;
            pAreas++;
        }
    }
    AreaDesc *areas2 = new AreaDesc[NRow*NCol];
    memcpy(areas2, areas, sizeof(AreaDesc)*NRow*NCol);
    sort(areas2, areas2+NRow*NCol, greater<AreaDesc>());
    for(int i=0;i<max_NArea;i++) {
        if(areas2[i].scorce >= min_scorce) {
            areas[areas2[i].id].is_Good = true;
        }else{
            break;
        }
    }
    delete areas2;
}
