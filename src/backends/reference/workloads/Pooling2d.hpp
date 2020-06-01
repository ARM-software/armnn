//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

#include "BaseIterator.hpp"

namespace armnn
{
/// Computes the Pooling2d operation.
void Pooling2d(Decoder<float>& rInputDecoder,
               Encoder<float>& rOutputEncoder,
               const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const Pooling2dDescriptor& params);
} //namespace armnn
