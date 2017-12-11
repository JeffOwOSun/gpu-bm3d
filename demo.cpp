#include <opencv2/core/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include "bm3d.h"

using namespace cv;
using namespace std;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void denoise_image() {
    namedWindow("image", WINDOW_AUTOSIZE);
    namedWindow("output", WINDOW_AUTOSIZE);

    Mat frame = imread("lena_20.png", IMREAD_GRAYSCALE);
    int width = frame.size().width;
    int height = frame.size().height - 100;
    int channel = 1;
    unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
    Bm3d bm3d;
    bm3d.set_up_realtime(width, height, channel);

    printf("input type: %s\n", type2str(frame.type()).c_str() );
    bm3d.realtime_denoise(frame.data, data);
    Mat output(height, width, CV_8U, data);
    printf("output type: %s\n", type2str(output.type()).c_str() );
    
    imshow("output", output);
    imshow("image", frame);
    waitKey();
}

int main( int argc, char** argv )
{
    // denoise_image();
    // return 0;
    string test_file = "opencv_test/noisy/gstennisg20.avi";

    namedWindow("video", WINDOW_AUTOSIZE);
    namedWindow("original", WINDOW_AUTOSIZE);
    Bm3d bm3d;
    Mat frame;
    Mat gray;
    Mat output_frame(240, 352, CV_8UC1);

    bm3d.set_up_realtime(352, 240, 1);
    VideoCapture cap(test_file);
    if (!cap.isOpened()) {
        printf("cap failed\n");
        return -1;
    }
    printf("here\n");
    
    for(;;) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cvtColor(frame, gray, CV_BGR2GRAY);
        int width = gray.size().width;
        int height = gray.size().height;
        int channels = 1;

        printf("width %d, height %d\n", width, height);
        bm3d.realtime_denoise(gray.data, output_frame.data);
        imshow("video", output_frame);
        imshow("original", frame);
        waitKey(50);
    }
    return 0;
}

