#include <stdlib.h>
#include <stdio.h>
#include <string>
#define cimg_display 1
#define cimg_use_png
#include "Cimg.h"
#include <getopt.h>

using namespace cimg_library;
using namespace std;

void usage(const char* progname) {
    printf("Usage: %s [options] refImage noisyImage\n", progname);
    printf("Program Options:\n");
    printf("  -r           reference image\n");
    printf("  -n           noisy image\n");
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
    printf("%s -> %s\n", ref_file.c_str(), noisy_file.c_str());
}
