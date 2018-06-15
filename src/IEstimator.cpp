//
// Created by mbed on 13.06.18.
//

#include "../include/IEstimator.h"
#include "../include/BlurEstimator.h"

std::unique_ptr<IEstimator> IEstimator::Create(ESTIMATOR_TYPE estimator_type) {
    std::unique_ptr<IEstimator> estimator;
    switch(estimator_type)
    {
        case BLUR:
            return std::make_unique<BlurEstimator>();
        default:
            return std::make_unique<BlurEstimator>();
        // if other are specified, please implement their construction here and add to ESTIMATOR_TYPE enum
    }
}
