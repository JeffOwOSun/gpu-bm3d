#include <opencv2/core/core.hpp>
#include <opencv2/xphoto.hpp>
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include <unistd.h>


using namespace cv;
using namespace std;

#define MAXPATHLEN 256

std::string get_working_path()
{
   char temp[MAXPATHLEN];
   return ( getcwd(temp, MAXPATHLEN) ? std::string( temp ) : std::string("") );
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    string folder = get_working_path();
    string original = folder + "/BM3D_images/Noisy/lena_20.png";
    string expect = folder + "/BM3D_images/Original/lena.png";

    Mat image = imread(original, cv::IMREAD_GRAYSCALE);   // Read the file
    Mat expected = imread(expect, cv::IMREAD_GRAYSCALE);
    Mat result;

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    cv::xphoto::bm3dDenoising(image, result, 10, 4, 16, 2500, 400, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEPALL);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Noise", image );
    imshow( "Processed", result );
    imshow("Original", expected);                     // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
