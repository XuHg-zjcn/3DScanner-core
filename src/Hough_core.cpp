#include "Hough_core.h"

#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;
#define u8_max 255

Hough_core::Hough_core(const thread_para *init_paras, array2d<uint8_t> *img_in, array2d<uint8_t> *img_out)
{
    //ctor
    memcpy(&(this->paras), init_paras, sizeof(thread_para));
    this->img_in = img_in;
    this->img_out = img_out;
}

Hough_core::~Hough_core()
{
    //dtor
}

point_u32::point_u32():xy<uint32_t>{0, 0} {}
point_u32::point_u32(point_u32 &old):xy<uint32_t>{old.x, old.y} {}
point_u32::point_u32(uint32_t set_x, uint32_t set_y):xy<uint32_t>{set_x<<16U, set_y<<16U} {}

void point_u32::set_xy16(uint32_t x, uint32_t y) {
    this->x = x<<16U;
    this->y = y<<16U;
}

inline void point_u32::add_delta(xy<int> &delta) {
    x += delta.x;
    y += delta.y;
}
//更新分量
inline void point_u32::update_H16_BL8(H16_BL8 &upd) {
    upd.H16.x = x>>16U;
    upd.H16.y = y>>16U;
    upd.BL8.x = (x>>8U)&0xFFU;
    upd.BL8.y = (y>>8U)&0xFFU;
}
//双线性插值 output 24bit
inline uint32_t Hough_core::BiLinear(uint64_t *pNear4, uint32_t x, uint32_t y) {
    uint32_t sum=0;
    uint32_t ix = u8_max - x;
    uint32_t iy = u8_max - y;
    sum += *pNear4++ * ix * iy;
    sum += *pNear4++ * ix * y;
    sum += *pNear4++ * x * iy;
    sum += *pNear4   * x * y;
    return sum;
}
//获取用于双线性插值的四个点
inline void Hough_core::get_near4(xy<uint32_t> &pH16, uint64_t *pNear4) {
    *pNear4++ = *img_in->get(pH16.y+0,pH16.x+0);
    *pNear4++ = *img_in->get(pH16.y+0,pH16.x+1);
    *pNear4++ = *img_in->get(pH16.y+1,pH16.x+0);
    *pNear4   = *img_in->get(pH16.y+1,pH16.x+1);
    //*(img_in->get(pH16.y+0,pH16.x+0)) = 0; //show area
}
//N<256
inline uint32_t Hough_core::line_sum(point_u32 &start, xy<int> &delta, uint32_t N) {
    point_u32 point = point_u32(start);
    H16_BL8 xy_part;
    uint32_t sum_value=0;
    uint8_t *pPixel;
    uint8_t &x=xy_part.BL8.x;
    uint8_t &y=xy_part.BL8.y;
    uint8_t ix;
    uint8_t iy;
    for(uint32_t i=0;i<N;i++) {
        point.update_H16_BL8(xy_part);
        pPixel = img_in->get(xy_part.H16.x, xy_part.H16.y);
        ix = ~x;
        iy = ~y;

        sum_value += *pPixel * ix * iy;  //pixel(x,  y  )  img[y  ,x  ]
        pPixel++;
        sum_value += *pPixel *  x * iy;  //pixel(x+1,y  )  img[y  ,x+1]
        pPixel+=img_in->x;
        sum_value += *pPixel *  x *  y;  //pixel(x+1,y+1)  img[y+1,x+1]
        pPixel--;
        sum_value += *pPixel * ix *  y;  //pixel(x  ,y+1)  img[y+1,x  ]

        point.add_delta(delta);
    }
    sum_value = (sum_value/N)>>16U;
    //sum_value = 0x7f;
    //assert(sum_value<=0xff); //show area
    return sum_value;
}
//线程函数
void Hough_core::lines_search() {
    timespec ts0{}, ts1{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts0);
    float rad = paras.rads.start;           //i为0时的rad
    point_u32 point = point_u32(0U, 0U);      //无用
    xy<int> delta_point{0,0};               //线上点的间距
    xy<int> delta_line{0,0};                //平行线的间距
    for(uint32_t i=paras.out_area.y.a; i<paras.out_area.y.b; i++) {
        delta_point.x = (int)(cos(rad)*0x10000);
        delta_point.y = (int)(sin(rad)*0x10000);
        delta_line.x = -delta_point.y;
        delta_line.y = delta_point.x;

        point.copy_from_old(paras.in_X0);
        uint8_t *ptr = img_out->get(i, paras.out_area.x.a);
        for(uint32_t j=paras.out_area.x.a; j<paras.out_area.x.b; j++) {
            *ptr++ = line_sum(point, delta_point, paras.N_length);
            point.add_delta(delta_line);
        }

        rad += paras.rads.step;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts1);
    cout<< (ts1.tv_nsec - ts0.tv_nsec)/1000 << "us" <<endl;
}

