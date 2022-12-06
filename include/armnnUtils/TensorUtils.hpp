//
// Copyright © 2019,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <utility>
#include <vector>

namespace armnn
{
class ITensorHandle;
}  // namespace armnn

namespace armnnUtils
{
armnn::TensorShape GetTensorShape(unsigned int numberOfBatches,
                                  unsigned int numberOfChannels,
                                  unsigned int height,
                                  unsigned int width,
                                  const armnn::DataLayout dataLayout);

armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayout dataLayout,
                                const armnn::DataType dataType);

armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int depth,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayout dataLayout,
                                const armnn::DataType dataType);

std::pair<float, float> FindMinMax(armnn::ITensorHandle* tensorHandle);

armnn::TensorShape ExpandDims(const armnn::TensorShape& tensorShape, int axis);

std::vector<unsigned int> SqueezeDims(const armnn::TensorShape& tensorShape);

unsigned int GetNumElementsBetween(const armnn::TensorShape& shape,
                                   unsigned int firstAxisInclusive,
                                   unsigned int lastAxisExclusive);

unsigned int GetUnsignedAxis(const unsigned int inputDimension, const int axis);

unsigned int GetNumElementsAfter(const armnn::TensorShape& shape, unsigned int axis);

std::pair<unsigned int, std::vector<float>> GetPerAxisParams(const armnn::TensorInfo& info);

} // namespace armnnUtils
