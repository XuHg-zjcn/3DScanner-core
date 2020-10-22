#ifndef OPTFLOW_FFT_H
#define OPTFLOW_FFT_H
#include <stdint.h>
#include <fftw3.h>
#include <opencv2/core/core.hpp>
using namespace cv;

#ifdef D3SCANNER_CORE_USE_SINGLE
#define fftw_malloc fftwf_malloc
#define fftw_free fftwf_free
#define fftw_destroy_plan fftwf_destroy_plan

#define fftw_plan fftwf_plan
#define fftw_plan_dft_r2c_2d fftwf_plan_dft_r2c_2d
#define fftw_plan_dft_c2r_2d fftwf_plan_dft_c2r_2d
#define fftw_execute fftwf_execute
#define fftw_import_wisdom_from_filename fftwf_import_wisdom_from_filename
#define fftw_export_wisdom_to_file fftwf_export_wisdom_to_file

#define fftw_complex fftwf_complex
#define double float
#endif

typedef struct {
    double SNR; //Signal(Enegry) / Noise(Power), Signal equal Noise Npixel
    double TopP;//Sum_Ntop(Enegry) / Signal(Enegry)
    int Nmost;  //Sum[0<=i<Nmost] Top(i) >= 0.9
}ifft_quality;

typedef struct AreaDesc{
    int id;
    int x0;
    int y0;
    bool is_Good;
    ifft_quality qua;
    double scorce;
}AreaDesc;

class optflow_FFT
{
    public:
        optflow_FFT(int n, int width, int height);
        virtual ~optflow_FFT();
        void run(int i);
        void fill_data(Mat &in, int x0, int y0);
        void calc_delta(bool sq2);
        void get_ifft_info(int w, double most, int SumNtop, ifft_quality *info);
        void WT(Mat *out, double div);
        void out_ifft(Mat *out);
        void copy_mul(Mat *out);
        void getGoodArea(Mat &img1, Mat &img2, int max_NArea, double min_scorce);
        void draw_mask(Mat &color);
        void draw_area_rgb(Mat &color, int x0, int y0) const;
        void draw_area_rgba(Mat &color, int x0, int y0) const;

    protected:
        static int save();
        //int load();

    private:
        int n;
        fftw_plan p1;
        fftw_plan p2;
        fftw_plan p_ifft;
        double *crop_db;
        fftw_complex *out1;
        fftw_complex *out2;
        fftw_complex *mul;
        double *ifft;
        int NRow;
        int NCol;
        AreaDesc *areas;

        static void complex_to_u8(fftw_complex *pIn, uint8_t *pOut, int len);
        static void double_to_u8(double *pIn, uint8_t *pOut, int len);
        void xsum(double dx, double dy, fftw_complex &ret);
        double corner4_sum(double *results, const int w);
        double ifft_sum();
};

#undef double
#endif // OPTFLOW_FFT_H
