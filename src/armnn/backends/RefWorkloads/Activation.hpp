//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

/// Performs the ActivationFunction elementwise on the inputs to give the outputs.
void Activation(const float* in,
                float* out,
                const TensorInfo& tensorInfo,
                ActivationFunction function,
                float a,
                float b);

} //namespace armnn
