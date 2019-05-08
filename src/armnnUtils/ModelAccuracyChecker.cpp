//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <vector>
#include <map>
#include <boost/log/trivial.hpp>
#include "ModelAccuracyChecker.hpp"

namespace armnnUtils
{

armnnUtils::ModelAccuracyChecker::ModelAccuracyChecker(const std::map<std::string, int>& validationLabels)
    : m_GroundTruthLabelSet(validationLabels){}

float ModelAccuracyChecker::GetAccuracy(unsigned int k)
{
    if(k > 10) {
        BOOST_LOG_TRIVIAL(info) << "Accuracy Tool only supports a maximum of Top 10 Accuracy. "
                                   "Printing Top 10 Accuracy result!";
        k = 10;
    }
    unsigned int total = 0;
    for (unsigned int i = k; i > 0; --i)
    {
        total += m_TopK[i];
    }
    return static_cast<float>(total * 100) / static_cast<float>(m_ImagesProcessed);
}
}