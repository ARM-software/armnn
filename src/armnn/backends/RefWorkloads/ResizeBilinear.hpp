//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

void ResizeBilinear(const float* in, const TensorInfo& inputInfo, float* out, const TensorInfo& outputInfo);

} //namespace armnn
