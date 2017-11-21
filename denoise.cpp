#include <stdlib.h>
#include <stdio.h>
#include <string>

// #include "bm3d.hpp"
#define cimg_display 1
#define cimg_use_png
#include "CImg.h"
#include <getopt.h>

using namespace cimg_library;
using namespace std;


void usage(const char* progname) {
    printf("Usage: %s [options] InputFile OutputFile\n", progname);
    printf("Program Options:\n");
    printf("  -s  --sigma <INT>          Noisy level\n");
    printf("  -c  --color                Color Image\n");
    printf("  -t  --step  <INT>          Perform which step of denoise, 1: first step, 2: both step\n");
    printf("  -v  --verbose              Print addtional infomation\n");
    printf("  -?  --help                 This message\n");
}

int main(int argc, char** argv)
{
    int opt;
    int channels = 1;
    int step = 2;
    int verbose = 0;
    int sigma = 0;
    string input_file, output_file;

    while ((opt = getopt(argc, argv, "s:ct:v?")) != EOF) {
        switch (opt) {
        case 's':
            sigma = atoi(optarg);
            break;
        case 'c':
            // color image
            channels = 3;
            break;
        case 't':
            step = atoi(optarg);
            break;
        case 'v':
            verbose = 1;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (optind + 2 > argc) {
        fprintf(stderr, "Error: missing File name\n");
        usage(argv[0]);
        return 1;
    }

    input_file = argv[optind];
    output_file = argv[optind+1];
    if (verbose) {
        printf("Sigma: %d\n", sigma);
        if (channels == 1) {
            printf("Image: Grayscale\n");
        } else {
            printf("Image: Color\n");
        }
        printf("Steps: %d\n", step);
    }

    //Allocate images
    CImg<unsigned char> image(input_file.c_str());
    CImg<unsigned char> image2(image.width(), image.height(), 1, channels, 0);

    // //Convert color image to YCbCr color space
    // if (channels == 3)
    //     image = image.get_channels(0,2).RGBtoYCbCr();

    // Check for invalid input
    if(! image.data() )
    {
        fprintf(stderr, "Error: Could not open file\n");
        return 1;
    }

    printf("Width: %d, Height: %d\n", image.width(), image.height());
    image.display("Image");
    // //Launch BM3D
    // try {
    //     BM3D bm3d;
    //     //          (n, k,N, T,   p,sigma, L3D)
    //     bm3d.set_hard_params(19,8,16,3000,3,sigma, 2.7f);
    //     bm3d.set_wien_params(19,8,32,400, 3,sigma);
    //     bm3d.denoise_host_image(image.data(),
    //              image2.data(),
    //              image.width(),
    //              image.height(),
    //              channels,
    //              twostep,
    //              verbose);
    // }
    // catch(std::exception & e)  {
    //     std::cerr << "There was an error while processing image: " << std::endl << e.what() << std::endl;
    //     return 1;
    // }

    // if (channels == 3) //color
    //     //Convert back to RGB color space
    //     image2 = image2.get_channels(0,2).YCbCrtoRGB();
    // else
    //     image2 = image2.get_channel(0);
    // //Save denoised image
    // image2.save( argv[2] );

    return 0;
}
