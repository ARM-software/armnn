//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

/// Computes the Pooling2d operation
void Pooling2d(const float* in,
               float* out,
               const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const Pooling2dDescriptor& params);

} //namespace armnn
