//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

namespace armnnUtils
{

void ProcessConcatInputTensorInfo(armnn::TensorInfo& inputTensorInfo, armnn::OriginsDescriptor& concatDescriptor,
                                  const unsigned int& concatAxis, unsigned int inputIndex,
                                  std::vector<unsigned int>& mergeDimSizes, unsigned int& mergeDim);

} // namespace armnnUtils
