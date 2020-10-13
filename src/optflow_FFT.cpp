#include "optflow_FFT.h"
#include <fftw3.h>
#include <unistd.h>
#include <time.h>
#include <iostream>
#include <cmath>
using namespace std;
#define pi 3.1415926535
optflow_FFT::optflow_FFT(uint32_t n)
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
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
}

void optflow_FFT::run(uint32_t n)
{
    if(n==0)
        fftw_execute(p1);
    else
        fftw_execute(p2);
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

void optflow_FFT::fill_data(Mat &mat_in, uint32_t x0, uint32_t y0)
{
    uint8_t *ptr_row;
    double *ptr_db=crop_db;
    for(uint32_t i=y0;i<y0+n;i++) {
        ptr_row = mat_in.ptr(i, x0);
        for(uint32_t j=0;j<n;j++) {
            *ptr_db++=*ptr_row++;
        }
    }
}

void optflow_FFT::calc_delta()
{
    double mul_real, mul_imag, sqrt2, v;
    for(int i=0;i<64*33;i++) {
        mul_real = out1[i][0]*out2[i][0] + out1[i][1]*out2[i][1];
        mul_imag =-out1[i][0]*out2[i][1] + out1[i][1]*out2[i][0];
        sqrt2 = sqrt(mul_real*mul_real + mul_imag*mul_imag);
        mul[i][0] = mul_real/sqrt2;
        mul[i][1] = mul_imag/sqrt2;
    }
    for(int i=0;i<64;i++) {
        for(int j=0;j<33;j++) {
            v = (double)(i+2*j)/64*2*pi* 3.5;
            mul[33*i+j][0] = cos(v);
            mul[33*i+j][1] = sin(v);
        }
    }
}

void optflow_FFT::xsum(double dx, double dy, fftw_complex &ret)
{
    double v;//, ret=0;
    ret[0]=0;
    ret[1]=0;
    for(int i=0;i<64;i++) {
        for(int j=0;j<33;j++) {
            v = (double)(i*dx+j*dy)/64*2*pi;
            ret[0]+=mul[33*i+j][0]*cos(v) - mul[33*i+j][1]*sin(v);
            ret[1]+=mul[33*i+j][0]*sin(v) + mul[33*i+j][1]*cos(v);
        }
    }
    ret[0]/=33*64;
    ret[1]/=33*64;
    ret[0]= sqrt(ret[0]*ret[0] + ret[1]*ret[1]);
}

void optflow_FFT::copy_result(uint8_t* p1, uint8_t* p2)
{
    double v,vmax=0,vmin=0;
    fftw_complex *vars=(fftw_complex*)malloc(sizeof(fftw_complex)*64*64);
    fftw_complex *vars1=vars;
    for(int i=0;i<64*33;i++) {
        v = mul[i][0];
        v = v*128 + 128;
        v = v<0   ?   0 : v;
        v = v>255 ? 255 : v;
        *p1++ = (uint8_t)v;
    }
    /*fftw_execute(p_ifft);
    for(int i=0;i<64*64;i++) {
        abs_v = ifft[i]/20;
        abs_v += 32;
        abs_v = abs_v<0   ?   0 : abs_v;
        abs_v = abs_v>255 ? 255 : abs_v;
        *p2++ = (uint8_t)abs_v;
    }*/
    vars1=vars;
    for(int i=-32;i<32;i++) {
        for(int j=-32;j<32;j++) {
            xsum(i/5.0, j/5.0, *vars1++);
        }
    }
    vars1=vars;
    for(int i=0;i<64*64;i++) {
        v = (*vars1++)[0];
        if(v<vmin)
            vmin=v;
        else if(v>vmax)
            vmax=v;
    }
    vars1=vars;
    for(int i=0;i<64*64;i++) {
        v = (*vars1++)[0];
        v = (v-vmin)/(vmax-vmin)*255;
        v = v<0   ?   0 : v;
        v = v>255 ? 255 : v;
        *p2++ = (uint8_t)v;
    }
    free(vars);
}
