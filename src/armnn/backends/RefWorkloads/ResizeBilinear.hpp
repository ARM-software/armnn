//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

void ResizeBilinear(const float* in, const TensorInfo& inputInfo, float* out, const TensorInfo& outputInfo);

} //namespace armnn
