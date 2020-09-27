#ifndef HOUGH_CORE_H
#define HOUGH_CORE_H

#include <cstdint>
#include <cstdlib>
#include <cassert>

template<typename T>
class xy{
public:
    T x;
    T y;
    inline void copy_from_old(xy<T> &old) {
        x = old.x;
        y = old.y;
    }
    inline void set_xy(int x, int y) {
        this->x = x;
        this->y = y;
    }
};
typedef struct int_range{
    uint32_t a;
    uint32_t b;
}int_range;
typedef struct rads_range{
    float start;
    float step;
    int_range range;
}rads_range;
typedef struct H16_BL8{
    xy<uint16_t> H16;
    xy<uint8_t> BL8;
}H16_BL8;
class point_u32: public xy<uint32_t>{
public:
    point_u32();
    point_u32(point_u32 &old);
    point_u32(uint32_t set_x, uint32_t set_y);
    void set_xy16(uint32_t x, uint32_t y);
    inline void add_delta(xy<int> &delta);
    inline void update_H16_BL8(H16_BL8 &upd);
};
template<typename T>
class array2d: public xy<uint32_t>{
public:
    T *data;
    array2d(xy<uint32_t> xy_init, bool is_malloc) {
        copy_from_old(xy_init);
        ptr_init(is_malloc);
    }
    array2d(xy<uint32_t> xy_init, T *ptr) {
        copy_from_old(xy_init);
        this->data = ptr;
    }
    void ptr_init(bool is_malloc) {
        if(is_malloc)
            this->data = (T*)malloc(sizeof(T)*prod());
        else
            this->data = nullptr;
    }

    //auto prod = [x,y]{ return x*y; };
    inline uint64_t prod() {
        return x*y;
    }
    inline void set_ptr(T *p) {
        this->data = p;
    }
    inline T* get(uint32_t i, uint32_t j) {
        return data + (x*i) + j;
    }
    //T* get = [i,j]{ return data+(x*i)+j; };
};
typedef struct range_u32{
    uint32_t a;
    uint32_t b;
}range_u32;
typedef struct range_float{
    float start;
    float step;
}range_float;
typedef struct thread_para {
    xy<uint32_t> in_X0;
    xy<range_u32> out_area;
    range_float rads;
    uint32_t N_length;
}thread_para;

class Hough_core
{
    public:
        Hough_core(const thread_para *init_paras, array2d<uint8_t> *img_in, array2d<uint8_t> *img_out);
        virtual ~Hough_core();
        void     lines_search();

    protected:
        inline uint32_t BiLinear(uint64_t *pNear4, uint32_t x, uint32_t y);
        inline void     get_near4(xy<uint32_t> &pH16, uint64_t *pNear4);
        uint32_t line_sum(point_u32 &start, xy<int> &delta, uint32_t N);

    public:
        thread_para paras;
        array2d<uint8_t> *img_in;
        array2d<uint8_t> *img_out;
};

#endif // HOUGH_CORE_H
