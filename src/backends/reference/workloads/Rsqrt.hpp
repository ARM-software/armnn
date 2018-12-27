//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

/// Performs the reciprocal squareroot function elementwise
/// on the inputs to give the outputs.
void Rsqrt(const float* in,
           float* out,
           const TensorInfo& tensorInfo);

} //namespace armnn
