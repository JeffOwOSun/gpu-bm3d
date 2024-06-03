#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "bm3d.h" //need to include after opencv
#include <stdio.h>

namespace py = pybind11;

py::array_t<uint8_t> denoise_wrapper(py::array_t<uint8_t> src_image, float sigma_1st_step, float sigma_2nd_step, float lambda_3d, int step, bool verbose=false){
    py::buffer_info src_buf = src_image.request();

    if (src_buf.ndim != 2) {
        throw std::runtime_error("Input should be a 2D NumPy array");
    }

    // Create a cv::Mat with the same shape and data as the NumPy array
    cv::Mat src_mat(src_buf.shape[0], src_buf.shape[1], CV_8UC1, (uint8_t*)src_buf.ptr);
    cv::Mat dst_mat = cv::Mat::zeros(src_buf.shape[0], src_buf.shape[1], CV_8UC1);
    // cv::Mat dst_mat(src_buf.shape[0], src_buf.shape[1], CV_8UC1);
    
    if (verbose){
        std::cout << "Running GPU BM3D denoise..."<<std::endl;
        std::cout << "Sigma 1st step = "<<sigma_1st_step <<std::endl;
        if (step == 2)
            std::cout << "Sigma 2nd step = "<<sigma_2nd_step <<std::endl;
        std::cout << "1st step hard thresholding = "<<lambda_3d <<std::endl;
    }
    Bm3d bm3d;
    bm3d.denoise(
        (uint8_t*)src_buf.ptr,
        dst_mat.data,
        src_mat.size().width,
        src_mat.size().height,
        sigma_1st_step,
        sigma_2nd_step,
        lambda_3d,
        1,
        step, //1: first step, 2: both step
        verbose
    );

    // convert cv mat to numpy array for output
     py::buffer_info dst_buf(
        (uint8_t*)dst_mat.data,                        // Pointer to buffer
        sizeof(uint8_t),                  // Size of one scalar
        py::format_descriptor<uint8_t>::format(), // Python struct-style format descriptor
        2,                                      // Number of dimensions
        { dst_mat.rows, dst_mat.cols },                 // Buffer dimensions
        { dst_mat.step[0], dst_mat.step[1] }            // Strides (in bytes) for each index
    );
    
    return py::array_t<uint8_t>(dst_buf);
}

PYBIND11_MODULE(pyGpuBM3D, m) {
    m.doc() = "Python GPU BM3D denoising";
    m.def("denoise", &denoise_wrapper,
        R"pbdoc(
          GPU BM3D denoise function.

          Parameters
          ----------
          input : numpy.ndarray
              2D NumPy array of unsigned char (dtype=np.uint8).

          sigma_1st_step : float
              Noise standard deviation for first step 

          sigma_2nd_step : float
              Noise standard deviation for second step 

          lambda_3d: float
              Threshold in first step collaborative filtering
        
          step: int
              Perform which step of denoise, 1: first step, 2: both step (default=2)

          verbose : bool
              Verbose output. (Default=True)

          Returns
          -------
          numpy.ndarray
              Denoised 2D NumPy array with the same shape and dtype.
          )pbdoc"
    );
}