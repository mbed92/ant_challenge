#include <iostream>
#include "include/utilities.h"
#include "include/IEstimator.h"


int main(int argc, char** argv) {
    // parse input parameters
    Parser parser(argc);
    Arguments arguments = parser.Parse(argv);

    // create BlurEstimator by Factory
    auto estimator = IEstimator::Create(BLUR);

    // setup & run core
    estimator->LoadData(arguments);
    estimator->SetVisualization(true);
    estimator->SetSave(true);
    estimator->Process();

    return EXIT_SUCCESS;
}