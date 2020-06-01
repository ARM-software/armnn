//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>

namespace armnn
{

/// Computes the softmax function on some inputs, into outputs, with a shape given by tensorInfo.
void Softmax(Decoder<float>& in, Encoder<float>& out, const TensorInfo& inputTensorInfo, float beta, int axis = -1);

} //namespace armnn
