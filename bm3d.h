#ifndef __BM3D_H__
#define __BM3D_H__


// #ifndef uint
// #define uint unsigned int
// #endif

#ifndef uchar
#define uchar unsigned char
#endif


#include "params.h"

class Bm3d
{
private:
    // image

    // model parameter
    Params h_fst_step_params;
    Params h_2nd_step_params;

    // device parameter

public:
    Bm3d();
    ~Bm3d();

    void set_fst_step_param();

    void set_2nd_step_param();

    void set_device_param();

    void copy_image_to_device(uchar *src_image,
                              int width,
                              int height,
                              int channels);

    void free_device_params();

    void denoise(uchar *src_image,
                 uchar *dst_image,
                 int width,
                 int height,
                 int channels,
                 int step,
                 int verbose);

    void denoise_fst_step();

    void denoise_2nd_step();

    /* data */
};

#endif
