//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
# pragma once

#include <string>
#include <map>
#include "ArmnnNetworkExecutor.hpp"

namespace kws 
{

/**
* @brief Decodes quantised last layer of model output
*
*/
class Decoder 
{
private:
    int quantisationOffset;
    float quantisationScale;

public:

    Decoder(int quantisationOffset, float quantisationScale) : quantisationOffset(quantisationOffset),
                                                               quantisationScale(quantisationScale) {}

    std::pair<int, float> decodeOutput(std::vector<int8_t>& modelOutput);

};
} // namespace kws