//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

/// Performs the ActivationFunction elementwise on the inputs to give the outputs
void Activation(const float* in,
                float* out,
                const TensorInfo& tensorInfo,
                ActivationFunction function,
                float a,
                float b);

} //namespace armnn
