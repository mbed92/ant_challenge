//
// Created by mbed on 13.06.18.
//

#include <vector>
#include <sstream>
#include <fstream>
#include "../include/BlurEstimator.h"

BlurEstimator::BlurEstimator() {
    SetBlurKernel();
    SetLaplacianKernel();
    is_visualized_ = false;
    is_saved_ = false;
}

bool BlurEstimator::LoadData(const Arguments &arguments) {
    arguments_ = arguments;
    is_visualized_ = false;
    return false;
}

bool BlurEstimator::Process() {
    std::vector<std::string> image_paths = GetImagePaths();
    float depth_weight = 1.0f;
    float depth_weight_factor = 1.0f / image_paths.size();

    if(is_visualized_){
        cv::namedWindow("depth_image", CV_WINDOW_NORMAL);
    }
    cv::Mat frames_combined, depth_combined;
    cv::Mat average_laplacian, average_frame;

    for(auto& path : image_paths)
    {
        std::cout << "Processing: " << path << std::endl;
        cv::Mat mask, laplacian;
        cv::Mat frame = LoadImage(path, cv::IMREAD_COLOR);
        bool is_estimated = EstimateBlur(frame, laplacian);
        if(!is_estimated) {
            std::cerr << "Cannot estimate depth in frame: " << path << std::endl;
        }

        // initialize 'average' arrays
        average_laplacian = average_laplacian.empty() ? laplacian : (average_laplacian + laplacian);
        average_frame = average_frame.empty() ? cv::Mat::zeros(frame.size(), frame.type()) : (average_frame + frame);

        // 1. apply mask to RGB
        cv::Mat laplacian_replicated, sharp;
        Replicate(laplacian, laplacian_replicated);
        sharp = frame.mul(laplacian_replicated);
        frames_combined = frames_combined.empty() ? cv::Mat::zeros(frame.size(), frame.type()) : (frames_combined + sharp);

        // 2. create depth map
        cv::Mat laplacian_depth_thresholded(laplacian.size(), laplacian.type(), 0.0);
        cv::Mat laplacian_depth_dilated(laplacian.size(), laplacian.type(), 0.0);
        Threshold(laplacian, laplacian_depth_thresholded, 0.2, depth_weight * depth_weight);    // square for better visualization
        depth_combined = depth_combined.empty() ? cv::Mat::zeros(frame.size(), CV_32FC1) : (depth_combined + laplacian_depth_thresholded);

        // visualize progress
        if(is_visualized_) {
            cv::imshow("depth_image", depth_combined);
            cv::waitKey(200);
        }
        depth_weight -= depth_weight_factor;
    }

    // invert laplacian to obtain blurred part of average_frame
    cv::Mat inv_laplacian = 1.0 - average_laplacian;
    Replicate(inv_laplacian, inv_laplacian);
    average_frame /= image_paths.size();

    // combine blurred average_frame with the sum of sharp parts
    cv::Mat deblurred_image = 1.5 * frames_combined + 0.8 * average_frame.mul(inv_laplacian);

    // visualize and save results
    if(is_visualized_) {
        cv::namedWindow("deblurred_image", CV_WINDOW_NORMAL);
        cv::imshow("deblurred_image", deblurred_image);
        cv::imshow("depth_image", depth_combined);
        cv::waitKey();
    }

    if(is_saved_) {
        deblurred_image *= 255.;
        depth_combined *= 255.;
        cv::imwrite("./deblurred.png", deblurred_image);
        cv::imwrite("./depth.png", depth_combined);
    }
}

void BlurEstimator::SetVisualization(bool is_visualized) {
    is_visualized_ = is_visualized;
}

void BlurEstimator::SetSave(bool is_saved) {
    is_saved_ = is_saved;
}

/// private section
std::vector<std::string> BlurEstimator::GetImagePaths() {
    std::string line;
    std::ifstream infile(arguments_.input_file_list);
    std::vector<std::string> paths;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string l;
        if (!(iss >> l)) {
            break;
        }
        paths.push_back(line);
    }
    return paths;
}

cv::Mat BlurEstimator::LoadImage(const std::string &path, cv::ImreadModes mode) {
    if(path.empty()) {
        std::cout << "Empty path was specified." << std::endl;
        return cv::Mat();
    }

    cv::Mat image = cv::imread(path, mode);
    if(image.empty()) {
        std::cerr << "Image loaded from: " << path << " is empty." << std::endl;
        return cv::Mat();
    }

    cv::Mat output;
    image.convertTo(output, CV_32F);
    output /= 255.;
    return output;
}

bool BlurEstimator::EstimateBlur(const cv::Mat &input_image, cv::Mat &laplacian) {
    cv::Mat blurred, gray_blurred, dst;
    Convolution(input_image, blur_kernel, blurred, 3);
    RBG2GRAYSCALE(input_image, gray_blurred);
    Convolution(gray_blurred, laplacian_kernel, dst, 3);
    float max = GetMax(dst);
    dst /= max;
    dst.copyTo(laplacian);
    return true;
}

double BlurEstimator::Convolve2D(const float *image, double **kernel, int sx, int sy, int width, int height, int kernel_size, int channels) {
    double result = 0.0;
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            int cx = sx + x;
            int cy = sy + y;
            float d;
            if (cx < 0 || cx >= width || cy < 0 || cy >= height) {
                d = 0;
            }
            else {
                d = *(image + width * y * channels + x * channels);
            }
            result += d * kernel[x][y];
        }
    }
    return result;
}

void BlurEstimator::Convolution(const cv::Mat &image, double **kernel, cv::Mat &result, int k_size) {

    int cols = image.cols;
    int rows = image.rows;
    int chan = image.channels();
    auto data = (float *) image.data;
    result = cv::Mat::zeros(rows, cols, image.type());
    auto *dst_data = (float *) result.data;
    int k_shift = (k_size - 1) / 2;

    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            for (int a = 0; a < chan; a++) {
                float *itData = (data + (x - k_shift) * cols * chan + (y - k_shift) * chan + a);
                *(dst_data + x * cols * chan + y * chan + a) =
                        static_cast<float>(Convolve2D(itData, kernel, y - k_shift, x - k_shift, cols, rows, k_size, chan));
            }
        }
    }
}

void BlurEstimator::Replicate(const cv::Mat &mat, cv::Mat& output) {
    // Needed to multiplicate one channel mat with 3 channels
    cv::Mat t[] = {mat, mat, mat};
    cv::merge(t, 3, output);
}

bool BlurEstimator::Threshold(const cv::Mat& input_matrix, cv::Mat& output_matrix, float thresh, float max) {
    input_matrix.copyTo(output_matrix);
    for(int i = 0; i < output_matrix.rows; i++) {
        for(int k = 0; k < output_matrix.cols; k++){
            if(output_matrix.at<float>(i, k) > thresh) {
                output_matrix.at<float>(i, k) = max;
            } else {
                output_matrix.at<float>(i, k) = 0;
            }
        }
    }
    return true;
}

void BlurEstimator::SetBlurKernel() {
    blur_kernel = new double *[3];
    for (int i = 0; i < 3; i++) {
        blur_kernel[i] = new double[3];
    }

    blur_kernel[0][0] = 1/16.;
    blur_kernel[0][1] = 2/16.;
    blur_kernel[0][2] = 1/16.;
    blur_kernel[1][0] = 2/16.;
    blur_kernel[1][1] = 4/16.;
    blur_kernel[1][2] = 2/16.;
    blur_kernel[2][0] = 1/16.;
    blur_kernel[2][1] = 2/16.;
    blur_kernel[2][2] = 1/16.;
}

void BlurEstimator::SetLaplacianKernel() {
    laplacian_kernel = new double *[3];
    for (int i = 0; i < 3; i++) {
        laplacian_kernel[i] = new double[3];
    }

    laplacian_kernel[0][0] = -1.0;
    laplacian_kernel[0][1] = -1.0;
    laplacian_kernel[0][2] = -1.0;
    laplacian_kernel[1][0] = -1.0;
    laplacian_kernel[1][1] = 8.0;
    laplacian_kernel[1][2] = -1.0;
    laplacian_kernel[2][0] = -1.0;
    laplacian_kernel[2][1] = -1.0;
    laplacian_kernel[2][2] = -1.0;
}

void BlurEstimator::RBG2GRAYSCALE(const cv::Mat &image, cv::Mat &result) {
    int cols = image.cols;
    int rows = image.rows;
    int chan = image.channels();

    auto data = (float *) image.data;
    result = cv::Mat::zeros(rows, cols, CV_32FC1);
    auto *dst_data = (float *) result.data;

    for(int i = 0; i < rows; i++) {
        for(int k = 0; k < cols; k++){
            for (int a = 0; a < chan; a++) {
                auto b = *(data + cols * i * chan + k * chan);
                auto g = *(data + cols * i * chan + k * chan + 1);
                auto r = *(data + cols * i * chan + k * chan + 2);
                *(dst_data + i * cols + k) = static_cast<float>(0.21*r + 0.72*g + 0.07*b);
            }
        }
    }
}

float BlurEstimator::GetMax(const cv::Mat &image) {
    int cols = image.cols;
    int rows = image.rows;
    int chan = image.channels();
    auto data = (float *) image.data;

    float max_tmp = -1;
    for(int i = 0; i < rows; i++) {
        for(int k = 0; k < cols; k++){
            for (int a = 0; a < chan; a++) {
                auto val = *(data + i * cols * chan + k * chan);
                if(val > max_tmp )
                    max_tmp  = val;
            }
        }
    }
    return max_tmp;
}

