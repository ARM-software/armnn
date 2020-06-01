//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/TensorFwd.hpp>

#include <set>

namespace armnnUtils
{

void ProcessConcatInputTensorInfo(armnn::TensorInfo& inputTensorInfo,
                                  armnn::OriginsDescriptor& concatDescriptor,
                                  const unsigned int& concatAxis,
                                  unsigned int inputIndex,
                                  unsigned int& mergeDimOrigin);

/// Creates a tensor info after reducing the dimensions mentioned in axisData.
void CalculateReducedOutputTensoInfo(const armnn::TensorInfo& inputTensorInfo,
                                     const std::set<unsigned int>& axisSet,
                                     bool keepDims,
                                     armnn::TensorInfo& outputTensorInfo);

/// Create output tensor info for a StridedSlice operator
void CalculateStridedSliceOutputTensorInfo(const armnn::TensorInfo& inputTensorInfo,
                                           const armnn::StridedSliceDescriptor& desc,
                                           armnn::TensorInfo& outputTensorInfo);

} // namespace armnnUtils
