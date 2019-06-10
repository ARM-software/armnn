//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/DescriptorsFwd.hpp"
#include "armnn/Tensor.hpp"
#include "BaseIterator.hpp"

#include <vector>

namespace armnn
{
void Mean(const TensorInfo& inputInfo,
          const TensorInfo& outputInfo,
          const std::vector<unsigned int>& axis,
          Decoder<float>& input,
          Encoder<float>& output);
} //namespace armnn

