//
// Created by mbed on 13.06.18.
//

#ifndef TOOPLOXBLURDEPTH_UTILITIES_H
#define TOOPLOXBLURDEPTH_UTILITIES_H

#include <iostream>

enum ESTIMATOR_TYPE
{
    BLUR
    // may be others ...
};

struct Arguments
{
    std::string input_file_list;
    std::string output_file_rgb;
    std::string output_file_depth;

    inline Arguments& operator=(const Arguments& other)
    {
        this->output_file_depth = other.output_file_depth;
        this->output_file_rgb= other.output_file_rgb;
        this->input_file_list = other.input_file_list;
        return *this;
    }
};

class Parser
{
 public:
    explicit Parser(int num_of_params);
    Arguments Parse(char** argv);

 private:
    const int num_of_params_;
};

#endif //TOOPLOXBLURDEPTH_UTILITIES_H
