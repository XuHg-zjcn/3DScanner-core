#ifndef OPTFLOW_FFT_H
#define OPTFLOW_FFT_H
#include <stdint.h>
#include <fftw3.h>
#include <opencv2/core/core.hpp>
using namespace cv;
class optflow_FFT
{
    public:
        optflow_FFT(uint32_t n);
        virtual ~optflow_FFT();
        void run(uint32_t n);
        void fill_data(Mat &in, uint32_t x0, uint32_t y0);
        void calc_delta();
        double get_ifft_SNR(int window);
        void xsum(double dx, double dy, fftw_complex &ret);
        void complex_to_u8(fftw_complex *pIn, uint8_t *pOut, int len);
        void double_to_u8(double *pIn, uint8_t *pOut, int len);
        void WT(Mat *out, double div);
        void out_ifft(Mat *out);
        void copy_mul(Mat *out);

    protected:
        int save();
        int load();

    private:
        uint32_t n=0;
        fftw_plan p1;
        fftw_plan p2;
        fftw_plan p_ifft;
        double *crop_db;
        fftw_complex *out1;
        fftw_complex *out2;
        fftw_complex *mul;
        double *ifft;
};

#endif // OPTFLOW_FFT_H
