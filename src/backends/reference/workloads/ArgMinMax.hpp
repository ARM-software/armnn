//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/Tensor.hpp"
#include "armnn/Descriptors.hpp"

#include "Decoders.hpp"

namespace armnn
{

template <typename OUT>
void ArgMinMax(Decoder<float>& in, OUT *out, const TensorInfo& inputTensorInfo,
               const TensorInfo& outputTensorInfo, ArgMinMaxFunction function, int axis);

} //namespace armnn

