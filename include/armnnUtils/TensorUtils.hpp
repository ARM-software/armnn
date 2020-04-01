//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

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

std::pair<float, float> FindMinMax(armnn::ITensorHandle* tensorHandle);

armnn::TensorShape ExpandDims(const armnn::TensorShape& tensorShape, int axis);

unsigned int GetNumElementsBetween(const armnn::TensorShape& shape,
                                   unsigned int firstAxisInclusive,
                                   unsigned int lastAxisExclusive);

unsigned int GetUnsignedAxis(const unsigned int inputDimension, const int axis);

unsigned int GetNumElementsAfter(const armnn::TensorShape& shape, unsigned int axis);

std::pair<unsigned int, std::vector<float>> GetPerAxisParams(const armnn::TensorInfo& info);

} // namespace armnnUtils
