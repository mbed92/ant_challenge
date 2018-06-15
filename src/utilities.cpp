//
// Created by mbed on 13.06.18.
//

#include "../include/utilities.h"

Parser::Parser(int num_of_params) : num_of_params_(num_of_params) {}

Arguments Parser::Parse(char **argv) {
    if(num_of_params_ != 4) {
        std::cerr << "Bad number of arguments. Provide paths:\n"
                "<input list of images> <output path for RGB image> <output path for DEPTH image>" << std::endl;
        return Arguments();
    }

    Arguments arguments;
    try {
        arguments.input_file_list = argv[1];
        arguments.output_file_rgb = argv[2];
        arguments.output_file_depth = argv[3];
    }
    catch(std::exception &e) {
        std::cerr << "Cannot parse arguments." << std::endl;
        return Arguments();
    }

    return arguments;
}
