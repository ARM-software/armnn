//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BaseIterator.hpp"

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

namespace armnn
{
float Activation(float in,
                 ActivationFunction function,
                 float a,
                 float b);

void Activation(Decoder<float>& in,
                Encoder<float>& out,
                const TensorInfo& tensorInfo,
                ActivationFunction function,
                float a,
                float b);

} //namespace armnn
