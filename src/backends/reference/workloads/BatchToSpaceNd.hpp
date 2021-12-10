//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Types.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>

namespace armnn
{

void BatchToSpaceNd(const armnnUtils::DataLayoutIndexed& dataLayout,
                    const TensorInfo& inputTensorInfo,
                    const TensorInfo& outputTensorInfo,
                    const std::vector<unsigned int>& blockShape,
                    const std::vector<std::pair<unsigned int, unsigned int>>& cropsData,
                    Decoder<float>& inputDecoder,
                    Encoder<float>& outputEncoder);
} // namespace armnn
