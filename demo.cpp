#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include "bm3d.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    string test_file = "noisy/gflowersg20.avi";
    Mat frame;
    VideoCapture cap(test_file);
    if (!cap.isOpened())
        return -1;
    namedWindow("video", 1);
    for(;;) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        printf("Image size: %d, %d\n", frame.size[0], frame.size[1]);
        imshow("video", frame);
        waitKey(20);
    }
    return 0;
}

