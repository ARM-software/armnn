//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>

namespace armnnUtils
{

void ProcessConcatInputTensorInfo(armnn::TensorInfo& inputTensorInfo,
                                  armnn::OriginsDescriptor& concatDescriptor,
                                  const unsigned int& concatAxis,
                                  unsigned int inputIndex,
                                  unsigned int& mergeDimOrigin);

/// Creates a tensor info after reducing the dimensions mentioned in axisData.
void CalculateReducedOutputTensoInfo(const armnn::TensorInfo& inputTensorInfo, const armnn::TensorInfo& axisTensorInfo,
                                     const std::set<unsigned int>& axisSet, bool keepDims,
                                     armnn::TensorInfo& outputTensorInfo);

} // namespace armnnUtils
