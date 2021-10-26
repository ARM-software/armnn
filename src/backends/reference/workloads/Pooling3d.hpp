//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

#include "BaseIterator.hpp"

namespace armnn
{
/// Computes the Pooling3d operation.
void Pooling3d(Decoder<float>& rInputDecoder,
               Encoder<float>& rOutputEncoder,
               const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const Pooling3dDescriptor& params);
} //namespace armnn
