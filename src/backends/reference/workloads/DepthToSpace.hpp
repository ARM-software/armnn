//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

void DepthToSpace(const TensorInfo& inputInfo,
                  const DepthToSpaceDescriptor& descriptor,
                  const void* inputData,
                  void* outputData,
                  unsigned int dataTypeSize);

} // namespace armnn
