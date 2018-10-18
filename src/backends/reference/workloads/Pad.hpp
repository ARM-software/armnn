//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/DescriptorsFwd.hpp"
#include "armnn/Tensor.hpp"

#include <vector>

namespace armnn
{
template <typename T>
void Pad(const TensorInfo& inputInfo,
         const TensorInfo& outputInfo,
         std::vector<std::pair<unsigned int, unsigned int>> m_PadList,
         const T* inputData,
         T* outData);
} //namespace armnn
