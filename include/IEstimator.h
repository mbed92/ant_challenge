//
// Created by mbed on 13.06.18.
//

#ifndef TOOPLOXBLURDEPTH_IESTIMATOR_H
#define TOOPLOXBLURDEPTH_IESTIMATOR_H


#include <bits/unique_ptr.h>

#include "utilities.h"

class IEstimator {
 public:
    static std::unique_ptr<IEstimator> Create(ESTIMATOR_TYPE estimator_type);
    virtual bool LoadData(const Arguments &arguments) = 0;
    virtual bool Process() = 0;
    virtual void SetVisualization(bool is_visualized) = 0;
    virtual void SetSave(bool is_saved) = 0;
};


#endif //TOOPLOXBLURDEPTH_IESTIMATOR_H
