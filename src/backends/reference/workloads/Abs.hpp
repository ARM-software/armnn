//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

/// Performs the absolute function elementwise
/// on the inputs to give the outputs.
void Abs(Decoder<float>& in,
         Encoder<float>& out,
         const TensorInfo& tensorInfo);

} //namespace armnn
