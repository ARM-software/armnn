//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <common/include/ProfilingGuid.hpp>

#include <array>

namespace armnn
{

using Coordinates = std::array<unsigned int, MaxNumOfTensorDimensions>;
using Dimensions  = std::array<unsigned int, MaxNumOfTensorDimensions>;

}
