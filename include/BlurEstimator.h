//
// Created by mbed on 13.06.18.
//

#ifndef TOOPLOXBLURDEPTH_BLURESTIMATOR_H
#define TOOPLOXBLURDEPTH_BLURESTIMATOR_H


#include "IEstimator.h"

#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


class BlurEstimator : public IEstimator {
 public:
    BlurEstimator();
    bool LoadData(const Arguments &arguments);
    bool Process();
    void SetVisualization(bool is_visualized);
    void SetSave(bool is_saved);

 private:
    bool EstimateBlur(const cv::Mat& input_image, cv::Mat &laplacian);

    // implemented methods from opencv
    bool Threshold(const cv::Mat& input_matrix, cv::Mat& output_matrix, float thresh, float max);
    void Convolution(const cv::Mat &image, double **kernel, cv::Mat &result, int k_size);
    double Convolve2D(const float *image, double **kernel, int sx, int sy, int width, int height, int kernel_size, int channels);
    double Dilate2D(const float *image, int sy, int sx, int width, int height, int kernel_size);
    void RBG2GRAYSCALE(const cv::Mat &image, cv::Mat &result);
    void Dilate(const cv::Mat &image, int k_size, cv::Mat& result);

    // utilities
    std::vector<std::string> GetImagePaths();
    cv::Mat LoadImage(const std::string& path, cv::ImreadModes mode);
    void SetBlurKernel();
    void SetLaplacianKernel();
    void Replicate(const cv::Mat& mat, cv::Mat& output);
    float GetMax(const cv::Mat& mat);


    Arguments arguments_;
    bool is_visualized_, is_saved_;
    double** blur_kernel;
    double** laplacian_kernel;
};


#endif //TOOPLOXBLURDEPTH_BLURESTIMATOR_H
