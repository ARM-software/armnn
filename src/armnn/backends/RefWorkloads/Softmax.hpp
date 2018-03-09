//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

/// Computes the softmax function on some inputs, into outputs, with a shape given by tensorInfo
void Softmax(const float* in, float* out, const TensorInfo& tensorInfo, float beta);

} //namespace armnn
