//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename T>
void StridedSlice(const TensorInfo& inputInfo,
                  const TensorInfo& outputInfo,
                  const StridedSliceDescriptor& params,
                  const T* inputData,
                  T* outputData);

} //namespace armnn
