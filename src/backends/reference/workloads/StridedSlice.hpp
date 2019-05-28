//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

void StridedSlice(const TensorInfo& inputInfo,
                  const StridedSliceDescriptor& params,
                  const void* inputData,
                  void* outputData,
                  unsigned int dataTypeSize);

} // namespace armnn
