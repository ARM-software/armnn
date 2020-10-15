//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cmath>

namespace armnnUtils
{

/**
 * Compare two floats and return true if their values are within a specified tolerance of each other.
 * @param a
 * @param b
 * @param tolerancePercentage If not supplied default will be 1% tolerance (1.0f)
 * @return true if the value of float b is within tolerancePercentage of the value for float a.
 */
inline bool within_percentage_tolerance(float a, float b, float tolerancePercent = 1.0f)
{
    float toleranceValue = std::fabs(a * (tolerancePercent / 100));
    return std::fabs(a - b) <= toleranceValue;
}

} //namespace armnn

