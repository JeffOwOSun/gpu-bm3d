#include <stdlib.h>
#include <stdio.h>
#include <string>
#define cimg_display 1
#define cimg_use_png
#include "Cimg.h"
#include <getopt.h>
#include <math.h>

using namespace cimg_library;
using namespace std;

void usage(const char* progname) {
    printf("Usage: %s [options] refImage noisyImage\n", progname);
    printf("Program Options:\n");
    printf("  -r           reference image\n");
    printf("  -n           noisy image\n");
}

float get_mse(unsigned char* img1, unsigned char* img2, int width, int height) {
    float mse = 0.0f;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float a = (float)img1[i + j * width];
            float b = (float)img2[i + j * width];
            mse += (a-b) * (a-b);
        }
    }
    return mse / (width * height);
}

int main(int argc, char** argv)
{
    int opt;
    int channels = 1;
    int step = 2;
    int verbose = 0;
    int sigma = 0;
    string ref_file, noisy_file;

    while ((opt = getopt(argc, argv, "r:n:")) != EOF) {
        switch (opt) {
        case 'r':
            ref_file = optarg;
            break;
        case 'n':
            noisy_file = optarg;
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }
    if (ref_file.length() == 0 || noisy_file.length() == 0) {
        usage(argv[0]);
        return 1;
    }
    CImg<unsigned char> ref_img(ref_file.c_str());
    CImg<unsigned char> noisy_img(noisy_file.c_str());
    if (ref_img.width() != noisy_img.width() || ref_img.height() != noisy_img.height()) {
        printf("Image dimension not match\n");
        return 1;
    }
    float mse = get_mse(ref_img.data(), noisy_img.data(), ref_img.width(), ref_img.height());
    float psnr = 20 * log10(255) - 10 * log10(mse);
    printf("PSNR: %f\n", psnr);
}
